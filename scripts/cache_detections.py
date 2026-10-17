"""Cache label-blind 2fps observations for each unique source content group."""
import argparse,hashlib,json,time
from pathlib import Path
from src import config
from service.pipeline.frames import iter_frames
from service.pipeline.detector import get_detector,_VERSION
from service.pipeline.tracker import Tracker,foreground_people


def extract(path,deadline=None):
    detector=get_detector();tracker=Tracker();rows=[];started=time.monotonic()
    for f in iter_frames(path,deadline):
        detections=detector.detect(f['image'],deadline);tracked=tracker.update(detections,f['timestamp_sec'])
        rows.append({k:f[k] for k in ['timestamp_sec','source_frame_index']}|{'detections':tracked,'foreground_person_tracks':[d['track_id'] for d in foreground_people(tracked)]})
    return {'schema_version':1,'engine':_VERSION,'frames':rows,'elapsed_sec':time.monotonic()-started,'sample_fps':2,'image_shape':list(f['image'].shape[:2])}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,default=config.DATA_PROC/'fortebank-20261017/video_protocol.json');parser.add_argument('--output',type=Path,default=config.DATA_PROC/'mission-detections-v1');parser.add_argument('--limit',type=int);args=parser.parse_args(argv)
    groups=json.loads(args.protocol.read_text())['groups'];args.output.mkdir(parents=True,exist_ok=True)
    for index,g in enumerate(groups[:args.limit] if args.limit else groups):
        path=config.ROOT/g['path'];target=args.output/(g['sha256']+'.json')
        if target.exists():
            old=json.loads(target.read_text())
            if old.get('source_sha256')!=g['sha256'] or old.get('engine')!=_VERSION:raise ValueError('Detection cache differs')
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=g['sha256']:raise ValueError('Source content changed')
        value=extract(path,time.monotonic()+240);value['source_sha256']=g['sha256']
        temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(value,separators=(',',':'))+'\n');temporary.chmod(0o600);temporary.replace(target)
        print(json.dumps({'completed':index+1,'total':len(groups),'frames':len(value['frames']),'elapsed_sec':round(value['elapsed_sec'],3)}),flush=True)

if __name__=='__main__':main()
