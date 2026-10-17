"""Run the frozen scorer and privately capture job facts for error clustering."""
import argparse,json,time
from pathlib import Path
import requests
from scripts import eval_video


def bounded_get(session,url,retries,**kwargs):
    """Retry polling only, within its original per-request time budget."""
    stop=time.monotonic()+float(kwargs.get('timeout',15))
    for attempt in range(3):
        remaining=stop-time.monotonic()
        if remaining<=0:raise requests.Timeout('Polling retry budget exhausted')
        try:
            response=session.get(url,**(kwargs|{'timeout':remaining}))
            if response.status_code not in [429,502,503,504] or attempt==2:return response
            response.close()
        except (requests.ConnectionError,requests.Timeout):
            if attempt==2:raise
        retries[0]+=1;time.sleep(min(.25*(attempt+1),max(0,stop-time.monotonic())))
    raise requests.Timeout('Polling retry budget exhausted')


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--trace-dir',type=Path,required=True)
    args,rest=parser.parse_known_args(argv);args.trace_dir.mkdir(parents=True,exist_ok=False)
    original=eval_video.api_run;protocol=json.loads(eval_video.PROTOCOL.read_text())
    original_get=requests.get;session=requests.Session();session.headers['User-Agent']='Mozilla/5.0'
    retries=[0]
    requests.get=lambda url,**kwargs:bounded_get(session,url,retries,**kwargs)
    paths={(eval_video.config.ROOT/g['path']).resolve():g['group_id'] for g in protocol['groups']}
    def capture(path,base,key,timeout=300):
        started=time.monotonic();before=retries[0]
        try:result,elapsed=original(path,base,key,timeout)
        except requests.RequestException as exc:
            target=args.trace_dir/(paths.get(Path(path).resolve(),Path(path).stem)+'.json')
            target.write_text(json.dumps(dict(status='failed',transport_error=type(exc).__name__,http_status=exc.response.status_code if exc.response is not None else None,elapsed_sec=time.monotonic()-started,poll_retries=retries[0]-before))+'\n')
            raise
        row=dict(job_id=result['job_id'],api_elapsed_sec=elapsed,status=result['status'])
        row['poll_retries']=retries[0]-before
        try:
            response=requests.get(base.rstrip('/')+'/ui/jobs/'+result['job_id'],timeout=10);response.raise_for_status()
            row['analysis']=response.json().get('analysis')
        except Exception as exc:row['trace_error']=type(exc).__name__
        identity=paths.get(Path(path).resolve(),Path(path).stem if Path(path).stem in set(paths.values()) else result['job_id'])
        target=args.trace_dir/(identity+'.json');target.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');target.chmod(0o600)
        return result,elapsed
    eval_video.api_run=capture
    try:eval_video.main(rest)
    finally:eval_video.api_run=original;requests.get=original_get;session.close()


if __name__=='__main__':main()
