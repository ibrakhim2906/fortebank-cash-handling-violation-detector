"""Durable, keyed jury API; one worker, original deadlines across restarts."""
from __future__ import annotations
import asyncio,hmac,json,multiprocessing as mp,os,re,time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from src.video_api import Result,UploadBodyLimit,MAX_RESULT_BYTES,_source_timing_context

ROOT=Path(__file__).resolve().parents[1]


def atomic_json(path,value):
    payload=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
    if len(payload)>MAX_RESULT_BYTES:raise ValueError('Result exceeds size bound')
    temp=path.with_suffix('.tmp')
    fd=os.open(temp,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600)
    with os.fdopen(fd,'wb') as f:f.write(payload)
    temp.replace(path)


def _child(path,deadline,out):
    try:
        from service.pipeline.analyze import analyze
        result=analyze(Path(path),deadline=deadline)
        value=Result.model_validate(result,context=_source_timing_context(path,deadline)).model_dump()
        if time.monotonic()>=deadline:raise TimeoutError
        atomic_json(Path(out),value)
    except Exception as exc:
        atomic_json(Path(path).parent/'analysis.json',{'engine':'failed','error_code':type(exc).__name__})
        atomic_json(Path(out),{'failed':True})


async def run_analysis(path,deadline):
    out=path.parent/'worker-result.json';process=mp.get_context('spawn').Process(target=_child,args=(str(path),deadline,str(out)),daemon=True)
    try:
        process.start()
        while time.monotonic()<deadline:
            if out.exists():
                if out.stat().st_size>MAX_RESULT_BYTES:raise ValueError('Result exceeds size bound')
                value=json.loads(out.read_text())
                if value.get('failed'):raise ValueError('Analysis failed')
                return value
            if not process.is_alive():raise ValueError('Analysis process exited')
            await asyncio.sleep(.05)
        raise TimeoutError('Original analysis deadline expired')
    finally:
        if process.pid:
            if process.is_alive():process.terminate()
            stop=time.monotonic()+.3
            while process.is_alive() and time.monotonic()<stop:await asyncio.sleep(.02)
            if process.is_alive():process.kill()
            if not process.is_alive():process.join(timeout=0);process.close()
        out.unlink(missing_ok=True)


class DurableJobs:
    def __init__(self,root,runner,seconds=300,capacity=2):
        self.root=root;self.runner=runner;self.seconds=seconds;self.capacity=capacity
        self.records={};self.queue=asyncio.Queue();self.worker=None;self.started=False

    def persist(self,r):atomic_json(self.root/r['job_id']/'state.json',r)

    async def recover(self):
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        for p in sorted(self.root.glob('*/state.json')):
            if not re.fullmatch('[0-9a-f]{12}',p.parent.name):continue
            try:
                r=json.loads(p.read_text());assert r['job_id']==p.parent.name
                assert r['status'] in ['processing','done','failed']
                assert type(r['expires_epoch']) in (int,float)
            except (ValueError,KeyError,AssertionError):continue
            self.records[r['job_id']]=r
            if r['status']=='processing':
                if not r.get('upload_complete') or r['expires_epoch']<=time.time() or not (p.parent/'source.mov').exists():
                    r['status']='failed';self.persist(r)
                else:self.queue.put_nowait(r['job_id'])
        self.started=True

    def reserve(self):
        if sum(r['status']=='processing' for r in self.records.values())>=self.capacity:
            raise HTTPException(503,'Processing capacity is busy',headers={'Retry-After':'5'})
        now=time.time();identifier=uuid4().hex[:12]
        while identifier in self.records:identifier=uuid4().hex[:12]
        p=self.root/identifier;p.mkdir(mode=0o700)
        r=dict(job_id=identifier,status='processing',created_epoch=now,expires_epoch=now+self.seconds,upload_complete=False)
        self.records[identifier]=r;self.persist(r);return r

    def public(self,identifier):
        r=self.records.get(identifier)
        if r is None:raise HTTPException(404,'Job not found')
        if r['status']=='processing' and time.time()>=r['expires_epoch']:
            r['status']='failed';self.persist(r)
        value={'job_id':identifier,'status':r['status'],'verdict':None,'violations':[]}
        if r['status']=='done':value.update(r['result'])
        return value

    async def work(self):
        while True:
            identifier=await self.queue.get();r=self.records[identifier]
            try:
                remaining=r['expires_epoch']-time.time()
                if remaining<=0 or r['status']!='processing':raise TimeoutError
                deadline=time.monotonic()+remaining
                result=await asyncio.wait_for(self.runner(self.root/identifier/'source.mov',deadline),remaining)
                value=Result.model_validate(result).model_dump()
                if time.time()>=r['expires_epoch']:raise TimeoutError
                r.update(status='done',result=value);self.persist(r)
            except asyncio.CancelledError:
                # Retain processing/upload state and original wall deadline for recovery.
                self.persist(r);raise
            except Exception:
                r['status']='failed';self.persist(r)
            finally:
                if r['status']!='processing':(self.root/identifier/'source.mov').unlink(missing_ok=True)
                self.queue.task_done()


def create_app(*,api_key=None,jobs_dir=None,analysis_runner=None,deadline_seconds=300,max_upload_bytes=200*1024**2):
    key=api_key if api_key is not None else os.environ.get('API_KEY',os.environ.get('DATATHON_API_KEY',''))
    directory=Path(jobs_dir or os.environ.get('JOBS_DIR',ROOT/'data/jobs')).resolve()
    if directory.is_relative_to((ROOT/'data/raw').resolve()):raise ValueError('Jobs cannot live in raw data')
    manager=DurableJobs(directory,analysis_runner or run_analysis,deadline_seconds)
    @asynccontextmanager
    async def lifespan(app):
        await manager.recover();manager.worker=asyncio.create_task(manager.work())
        try:yield
        finally:
            manager.worker.cancel()
            try:await manager.worker
            except asyncio.CancelledError:pass
    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    app.state.jobs=manager;app.add_middleware(UploadBodyLimit,maximum=max_upload_bytes+128*1024)
    # Only presentation assets are public. Recordings/jobs and evaluation source
    # artifacts remain outside both mounts; the snapshot contains aggregates only.
    app.mount('/static',StaticFiles(directory=ROOT/'service/static'),name='frontend-static')
    app.mount('/styles',StaticFiles(directory=ROOT/'styles'),name='frontend-styles')
    def authorize(request):
        if not key:raise HTTPException(503,'API key is not configured')
        if not hmac.compare_digest(request.headers.get('X-API-Key','').encode(),key.encode()):raise HTTPException(401,'A valid X-API-Key is required')
    @app.get('/health')
    async def health():return {'ok':manager.started,'queue':manager.queue.qsize()}
    def frontend_page(filename):
        return HTMLResponse((ROOT/'service/static'/filename).read_text(),headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer','Content-Security-Policy':"default-src 'none'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; media-src blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"})
    @app.get('/',response_class=HTMLResponse)
    @app.get('/review',response_class=HTMLResponse)
    async def ui():return frontend_page('index.html')
    @app.get('/results',response_class=HTMLResponse)
    async def results_page():return frontend_page('index.html')
    @app.get('/evaluation',response_class=HTMLResponse)
    async def evaluation_page():return frontend_page('evaluation.html')
    async def upload(request,auth):
        if auth:authorize(request)
        r=manager.reserve();p=directory/r['job_id']/'source.mov'
        try:
            async with asyncio.timeout(max(0.001,r['expires_epoch']-time.time())):
                async with request.form(max_files=1,max_fields=1) as form:
                    f=form.get('file')
                    if not isinstance(f,UploadFile) or len(form.multi_items())!=1:raise HTTPException(422,'Provide one multipart video in file')
                    if not (f.filename or '').lower().endswith(('.mov','.mp4')):raise HTTPException(415,'Only MOV/MP4 are supported')
                    size=0;fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
                    with os.fdopen(fd,'wb') as dest:
                        while chunk:=await f.read(256*1024):
                            size+=len(chunk)
                            if size>max_upload_bytes:raise HTTPException(413,'Video too large')
                            dest.write(chunk)
                    if not size:raise HTTPException(422,'Video is empty')
            r['upload_complete']=True;manager.persist(r);manager.queue.put_nowait(r['job_id'])
            return {'job_id':r['job_id']}
        except BaseException as exc:
            r['status']='failed';manager.persist(r);p.unlink(missing_ok=True)
            if isinstance(exc,TimeoutError):raise HTTPException(408,'Original upload deadline expired') from exc
            raise
    @app.post('/api/v1/analyze')
    async def api_upload(request:Request):return await upload(request,True)
    @app.get('/api/v1/jobs/{job_id}')
    async def api_status(job_id:str,request:Request):
        authorize(request);return JSONResponse(manager.public(job_id),headers={'Cache-Control':'no-store'})
    @app.post('/ui/analyze')
    async def ui_upload(request:Request):return await upload(request,False)
    @app.get('/ui/jobs/{job_id}')
    async def ui_status(job_id:str):
        value=manager.public(job_id);trace=directory/job_id/'analysis.json'
        if trace.exists() and trace.stat().st_size<=MAX_RESULT_BYTES:
            try:value['analysis']=json.loads(trace.read_text())
            except (ValueError,OSError):pass
        return JSONResponse(value,headers={'Cache-Control':'no-store'})
    return app


app=create_app()
