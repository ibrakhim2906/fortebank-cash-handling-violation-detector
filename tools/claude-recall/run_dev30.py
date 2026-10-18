"""Sequential DEV30 runs for the isolated Claude candidate on the shared pod.

Writes only under /workspace/claude-recall. Takes turns with Codex through the
same config-dev30-benchmark.lock its controllers use: waits until no Codex
controller/queue/benchmark is alive, the lock is free and the model is idle,
then holds the lock for one run and always releases it.
"""
import argparse,json,os,re,signal,subprocess,sys,time,urllib.request
from pathlib import Path

ROOT=Path('/workspace/claude-recall');LIVE=Path('/workspace/datathon-kit')
LOCK=LIVE/'data/processed/mission-runtime/config-dev30-benchmark.lock'
RUNTIME=ROOT/'data/processed/claude-runtime'
BUSY=('run_isolated.py','/queue.py','scripts.benchmark_auditor','scripts.eval_video','scripts.run_mission_eval','scripts.mission_checkpoint')


def log(*parts):print(time.strftime('%H:%M:%S'),*parts,flush=True)


def proc_rows():
    rows=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            args=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            stat=(entry/'stat').read_text().rsplit(')',1)[1].split()
        except OSError:continue
        rows.append(dict(pid=int(entry.name),ppid=int(stat[1]),start=int(stat[19]),args=args))
    return rows


def owned(rows):
    mine={os.getpid()}
    for _ in rows:mine.update(r['pid'] for r in rows if r['ppid'] in mine)
    return mine


def model_busy():
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:text=opener.open('http://127.0.0.1:8002/metrics',timeout=3).read(2*1024**2).decode()
    except OSError:return True
    for name in ['num_requests_running','num_requests_waiting']:
        values=re.findall(r'^vllm:'+name+r'(?:\{[^}]*\})?\s+([0-9.eE+\-]+)\s*$',text,re.M)
        if not values or any(float(v)!=0 for v in values):return True
    return False


def codex_busy():
    rows=proc_rows();mine=owned(rows)
    others=[r for r in rows if r['pid'] not in mine and any(n in r['args'] for n in BUSY)]
    return bool(others) or LOCK.exists() or model_busy()


def acquire(name,deadline):
    clear=0
    while time.time()<deadline:
        clear=0 if codex_busy() else clear+1
        if clear>=3:
            try:LOCK.mkdir()
            except FileExistsError:clear=0;continue
            me=[r for r in proc_rows() if r['pid']==os.getpid()][0]
            (LOCK/'owner.json').write_text(json.dumps(dict(pid=os.getpid(),start=me['start'],trial='claude-'+name,owner='claude-recall'),indent=2))
            return
        time.sleep(20)
    raise TimeoutError('GPU never became free')


def release():
    owner=LOCK/'owner.json'
    if owner.exists() and json.loads(owner.read_text()).get('pid')==os.getpid():owner.unlink();LOCK.rmdir()


def environment(overrides):
    env={k:v for k,v in os.environ.items() if not k.startswith('AUDITOR_')}
    for line in (ROOT/'deploy/mission-best.env').read_text().splitlines():
        line=line.strip()
        if line and not line.startswith('#'):key,value=line.split('=',1);env[key]=value
    env.update(overrides)
    env.update(UV_PROJECT_ENVIRONMENT=str(LIVE/'.venv'),UV_NO_SYNC='1',UV_OFFLINE='1',UV_NO_PROGRESS='1',PYTHONPATH=str(ROOT),
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    return env


def run(spec,ids,deadline):
    name=spec['name'];output=ROOT/'data/processed'/('claude-'+name);terminal=RUNTIME/(name+'-terminal.json')
    if output.exists() or terminal.exists():log('skip existing',name);return
    env=environment(spec.get('env',{}))
    recipe=subprocess.run(['uv','run','--frozen','--no-dev','python','-c','from service.pipeline.analyze import recipe_fingerprint;print(recipe_fingerprint())'],
                          cwd=ROOT,env=env,check=True,capture_output=True,text=True).stdout.strip()
    log('waiting for GPU turn',name);acquire(name,deadline);log('lock acquired',name)
    started=time.time();code=None;child=None
    try:
        (RUNTIME/(name+'-freeze.json')).write_text(json.dumps(dict(trial=name,recipe_sha256=recipe,env=spec.get('env',{}),started=started),indent=2))
        child=subprocess.Popen(['uv','run','--frozen','--no-dev','python','-m','scripts.benchmark_auditor','--inputs','data/processed/mission-inputs-v1',
                                '--output',str(output),'--profile','qwen3-32b-awq','--cached','--only',*ids],cwd=ROOT,env=env,start_new_session=True)
        code=child.wait(timeout=6000)
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30)
        release()
        records=[json.loads(p.read_text()) for p in output.glob('*.json')] if output.exists() else []
        terminal.write_text(json.dumps(dict(trial=name,returncode=code,records=len(records),elapsed_sec=time.time()-started,
            new_calls=sum(r.get('new_model_calls',0) for r in records),replayed_calls=sum(r.get('replayed_calls',0) for r in records),
            budget_flags=sum(bool(r.get('requires_api_budget_replay')) for r in records),recipe_sha256=recipe),indent=2))
        log('finished',name,'returncode',code,'records',len(records))


def main():
    p=argparse.ArgumentParser();p.add_argument('queue',type=Path);p.add_argument('--hours',type=float,default=4.5);args=p.parse_args()
    RUNTIME.mkdir(parents=True,exist_ok=True);deadline=time.time()+args.hours*3600
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    ids=json.loads((ROOT/'tools/claude-recall/dev30-ids.json').read_text())
    for spec in json.loads(args.queue.read_text()):run(spec,ids,deadline)


if __name__=='__main__':main()
