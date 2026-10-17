"""Run the frozen scorer and privately capture job facts for error clustering."""
import argparse,json
from pathlib import Path
import requests
from scripts import eval_video


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--trace-dir',type=Path,required=True)
    args,rest=parser.parse_known_args(argv);args.trace_dir.mkdir(parents=True,exist_ok=False)
    original=eval_video.api_run;protocol=json.loads(eval_video.PROTOCOL.read_text())
    paths={(eval_video.config.ROOT/g['path']).resolve():g['group_id'] for g in protocol['groups']}
    def capture(path,base,key,timeout=300):
        result,elapsed=original(path,base,key,timeout)
        row=dict(job_id=result['job_id'],api_elapsed_sec=elapsed,status=result['status'])
        try:
            response=requests.get(base.rstrip('/')+'/ui/jobs/'+result['job_id'],timeout=10);response.raise_for_status()
            row['analysis']=response.json().get('analysis')
        except Exception as exc:row['trace_error']=type(exc).__name__
        identity=paths.get(Path(path).resolve(),result['job_id'])
        target=args.trace_dir/(identity+'.json');target.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');target.chmod(0o600)
        return result,elapsed
    eval_video.api_run=capture
    try:eval_video.main(rest)
    finally:eval_video.api_run=original


if __name__=='__main__':main()
