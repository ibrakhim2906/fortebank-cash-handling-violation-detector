import asyncio,json,time
from service.app import create_app,atomic_json
from test_video_api import Client,KEY

async def clean(path,deadline):return {'verdict':'no_violation','violations':[]}


def test_contract_auth_ui_and_persistence(tmp_path):
    async def scenario():
        app=create_app(api_key=KEY,jobs_dir=tmp_path,analysis_runner=clean)
        async with Client(app) as c:
            assert (await c.request('GET','/health'))==(200,{'ok':True,'queue':0})
            assert (await c.upload(key=''))[0]==401
            assert (await c.status('unknown',key=''))[0]==401
            assert (await c.status('unknown'))[0]==404
            code,r=await c.upload();assert code==200 and set(r)=={'job_id'}
            identifier=r['job_id'];assert len(identifier)==12
            result=await c.done(identifier)
            assert result==dict(job_id=identifier,status='done',verdict='no_violation',violations=[])
            assert (await c.request('GET','/'))[0]==200
        async with Client(create_app(api_key=KEY,jobs_dir=tmp_path,analysis_runner=clean)) as c:
            assert (await c.status(identifier))[1]==result
    asyncio.run(scenario())


def test_failed_job_is_not_clean(tmp_path):
    async def fail(*args):raise ValueError('Private path/trace must not leak')
    async def scenario():
        async with Client(create_app(api_key=KEY,jobs_dir=tmp_path,analysis_runner=fail)) as c:
            _,r=await c.upload();identifier=r['job_id'];result=await c.done(identifier)
            assert result==dict(job_id=identifier,status='failed',verdict=None,violations=[])
    asyncio.run(scenario())


def test_recovery_preserves_original_deadline(tmp_path):
    identifier='a'*12;p=tmp_path/identifier;p.mkdir();(p/'source.mov').write_bytes(b'video')
    expiry=time.time()+.15
    atomic_json(p/'state.json',dict(job_id=identifier,status='processing',created_epoch=time.time()-299.85,expires_epoch=expiry,upload_complete=True))
    async def slow(*args):await asyncio.sleep(1);return {'verdict':'no_violation','violations':[]}
    async def scenario():
        async with Client(create_app(api_key=KEY,jobs_dir=tmp_path,analysis_runner=slow)) as c:
            r=await c.done(identifier);assert r['status']=='failed'
            saved=json.loads((p/'state.json').read_text());assert saved['expires_epoch']==expiry
    asyncio.run(scenario())


def test_partial_upload_cannot_resume_as_valid_video(tmp_path):
    identifier='b'*12;p=tmp_path/identifier;p.mkdir();(p/'source.mov').write_bytes(b'partial')
    atomic_json(p/'state.json',dict(job_id=identifier,status='processing',created_epoch=time.time(),expires_epoch=time.time()+300,upload_complete=False))
    async def scenario():
        async with Client(create_app(api_key=KEY,jobs_dir=tmp_path,analysis_runner=clean)) as c:
            assert (await c.status(identifier))[1]['status']=='failed'
    asyncio.run(scenario())


def test_wrong_extension_empty_and_oversize(tmp_path):
    async def scenario():
        async with Client(create_app(api_key=KEY,jobs_dir=tmp_path,analysis_runner=clean,max_upload_bytes=1024)) as c:
            assert (await c.upload(name='x.txt'))[0]==415
            assert (await c.upload(payload=b''))[0]==422
            assert (await c.upload(payload=b'v'*2048))[0]==413
    asyncio.run(scenario())
