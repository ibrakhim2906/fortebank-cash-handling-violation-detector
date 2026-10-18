"""Evaluation-only replay on anonymous inputs; never imported by the service.

No labels are read. Cached replies require matching model/runtime and exact
request bytes. Real API latency remains mandatory for promotion.
"""
import argparse,base64,hashlib,importlib,importlib.util,json,math,os,time,urllib.request
from contextlib import contextmanager
from pathlib import Path
import jsonschema
from src import config
from src.video_api import Result
from scripts.download_auditor import PROFILES
from service.pipeline import analyze as pipeline
from service.pipeline.frames import iter_frames


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()


@contextmanager
def pipeline_replay_clock(replayed):
    """Sequential benchmark only: advance pipeline clocks, never stdlib/network time.

    Import optional spatial verification before replacement so its lazy import
    sees the same clock. Restore every module even when inference raises.
    """
    names=['service.pipeline.analyze','service.pipeline.auditor','service.pipeline.vlm','src.video_analysis']
    for optional in ['service.pipeline.spatial','service.pipeline.desk','service.pipeline.state_ledger','service.pipeline.action_windows','service.pipeline.workspace_focus']:
        if importlib.util.find_spec(optional) is not None:names.append(optional)
    modules=[importlib.import_module(name) for name in names]
    originals=[(module,module.time) for module in modules]
    real_time=time
    class ReplayTime:
        def monotonic(self):return real_time.monotonic()+replayed()
        def __getattr__(self,name):return getattr(real_time,name)
    clock=ReplayTime()
    try:
        for module,_ in originals:module.time=clock
        yield
    finally:
        for module,original in originals:module.time=original


def runtime_signature(profile):
    selected=PROFILES[profile];manifest=json.loads((config.ROOT/selected['manifest']).read_text())
    installed=json.loads((Path(selected['directory'])/'source-manifest.json').read_text())
    if manifest!=installed:raise ValueError('Installed model differs from pinned manifest')
    commands=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():continue
        try:args=(process/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if b'serve' in args and selected['directory'].encode() in args and b'8002' in args and any(b'/opt/vllm' in a for a in args):commands.append(args)
    if len(commands)!=1:raise ValueError('Expected one matching model server')
    libraries=hashlib.sha256(Path('/opt/vllm/installed.txt').read_bytes()).hexdigest()
    return hashlib.sha256(canonical(dict(model=manifest,command=[a.decode() for a in commands[0]],libraries=libraries))).hexdigest()


def import_api_replies(inputs,runtime,cache):
    baseline=json.loads((inputs/'baseline-runtime.json').read_text())
    if baseline['runtime_signature']!=runtime:raise ValueError('Prior API observations came from a different model runtime')
    for path in (inputs/'observed-api').glob('*.json'):
        value=json.loads(path.read_text());analysis=value['analysis']
        for call in analysis.get('auditor',[]):
            if 'answer' not in call:continue
            response=dict(model='qwen-vl',choices=[dict(finish_reason='stop',message=dict(content=json.dumps(call['answer'],ensure_ascii=False)))])
            target=cache/runtime/(call['request_sha256']+'.json');target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_text(json.dumps(dict(response=response,elapsed_sec=call['elapsed_sec'],provenance='actual_prior_API_observation'),ensure_ascii=False)+'\n')


def run_one(path,input_id,inputs,cache,runtime,*,use_cache):
    started=time.monotonic();replayed=0.;fresh_calls=hits=0;budget_uncertain=False
    original=pipeline.perception
    def perception(source,deadline):
        nonlocal replayed
        cached=inputs/'detections'/(input_id+'.json');observed=inputs/'observed-api'/(input_id+'.json')
        if os.environ.get('PERCEPTION_MODE')=='frames_only' or not use_cache or not cached.exists() or not observed.exists():return original(source,deadline)
        detections=json.loads(cached.read_text());old=json.loads(observed.read_text())['analysis']
        from service.pipeline.detector import _VERSION
        if detections['engine']!=_VERSION:raise ValueError('Perception cache version differs')
        if detections['source_sha256']!=hashlib.sha256(Path(source).read_bytes()).hexdigest():raise ValueError('Perception source differs')
        before=time.monotonic();frames=[]
        for index,f in enumerate(iter_frames(source,deadline)):
            d=detections['frames'][index]
            if d['source_frame_index']!=f['source_frame_index'] or abs(d['timestamp_sec']-f['timestamp_sec'])>1e-5:raise ValueError('Cached frame sampling differs')
            frames.append(pipeline.source_sample(f,index))
        if len(frames)!=len(detections['frames']):raise ValueError('Cached frame count differs')
        replayed+=max(0,old['perception_sec']-(time.monotonic()-before))
        return detections,frames
    def request(url,payload,timeout):
        nonlocal replayed,fresh_calls,hits,budget_uncertain
        body=json.loads(payload);key=hashlib.sha256(payload).hexdigest();target=cache/runtime/(key+'.json')
        remaining=165-(time.monotonic()-started+replayed)
        if remaining<=0:
            budget_uncertain=True;raise TimeoutError('Cached policy requires real API budget replay')
        if use_cache and target.exists():
            entry=json.loads(target.read_text())
            elapsed=float(entry['elapsed_sec'])
            if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid cached duration')
            if elapsed>=remaining or elapsed>timeout:
                replayed+=min(timeout,remaining)
                budget_uncertain=True;raise TimeoutError('Cached call exceeds bounded request timeout')
            replayed+=entry['elapsed_sec'];hits+=1;return entry['response']
        req=urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'},method='POST');before=time.monotonic()
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=min(timeout,remaining)) as r:
            response=json.loads(r.read(128*1024))
        elapsed=time.monotonic()-before;fresh_calls+=1
        if response.get('model')=='qwen-vl' and response['choices'][0]['finish_reason']=='stop':
            try:
                value=json.loads(response['choices'][0]['message']['content']);jsonschema.validate(value,body['response_format']['json_schema']['schema'])
            except (ValueError,jsonschema.ValidationError):pass  # Parent validator/retry still sees the real reply.
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():target.write_text(json.dumps(dict(response=response,elapsed_sec=elapsed,provenance='actual_local_model_call'),ensure_ascii=False)+'\n')
        return response
    pipeline.perception=perception
    try:
        with pipeline_replay_clock(lambda:replayed):
            result,analysis=pipeline.analyze_detailed(path,deadline=started+185,requester=request)
        result=Result.model_validate(result).model_dump();status='done'
    except Exception as exc:result=None;analysis=dict(engine='failed',error_code=type(exc).__name__);status='failed'
    finally:pipeline.perception=original
    return dict(input_id=input_id,status=status,result=result,analysis=analysis,elapsed_sec=time.monotonic()-started,
                budget_estimate_sec=time.monotonic()-started+replayed,new_model_calls=fresh_calls,replayed_calls=hits,
                requires_api_budget_replay=budget_uncertain,mode='cached' if use_cache else 'full',runtime_signature=runtime)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--only',nargs='+',required=True);p.add_argument('--cached',action='store_true');p.add_argument('--import-api',action='store_true');args=p.parse_args(argv)
    runtime=runtime_signature(args.profile);cache=config.DATA_PROC/'mission-request-cache'
    if args.import_api:
        if args.profile!='qwen3-8b':raise ValueError('Imported observations belong only to the initial8B')
        import_api_replies(args.inputs,runtime,cache)
    manifest=json.loads((args.inputs/'manifest.json').read_text());lookup={r['input_id']:r for r in manifest['inputs']};args.output.mkdir(parents=True,exist_ok=True)
    for input_id in args.only:
        source=lookup[input_id];path=args.inputs/(input_id+'.mov')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Input source changed')
        target=args.output/(input_id+'.json')
        if target.exists():raise ValueError('Immutable benchmark output already exists')
        value=run_one(path,input_id,args.inputs,cache,runtime,use_cache=args.cached);target.write_text(json.dumps(value,ensure_ascii=False)+'\n');target.chmod(0o600)
        print(json.dumps({k:value[k] for k in ['input_id','status','elapsed_sec','new_model_calls','replayed_calls','requires_api_budget_replay']}),flush=True)


if __name__=='__main__':main()
