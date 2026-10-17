"""Bounded reserve, robustness and public API checkpoint; private artifacts only."""
import argparse,datetime,hashlib,json,os,random,subprocess,time
from pathlib import Path
from scripts.eval_video import PROTOCOL,ANNOTATIONS,SPLIT,api_run,score_records,setup
from scripts.mission_validation import TRANSFORMS,augment,event_signature
from src import config


def run(args,output):
    command=['uv','run','--offline','python','-m','scripts.run_mission_eval',
             '--trace-dir',str(output.with_suffix('.traces')),*args,'--json',str(output)]
    subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
    return json.loads(output.read_text())


def quick_groups(groups,split):
    """Four predeclared DEV sources, balanced by label and spread by setup."""
    rng=random.Random(config.SEED);selected=[]
    for target in [1,0]:
        pool=sorted((g for g in groups if g['group_id'] in split['dev'] and g['target']==target),key=lambda g:g['group_id'])
        rng.shuffle(pool);ordered=[]
        for camera in sorted({setup(g) for g in pool},reverse=True):
            ordered.append(next(g for g in pool if setup(g)==camera))
        ordered.extend(g for g in pool if g not in ordered)
        selected.extend(g['group_id'] for g in ordered[:2])
    if len(set(selected))!=4 or set(selected)&set(split['reserve']):raise ValueError('Invalid bounded development selection')
    return selected


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--api',required=True);p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--honest',type=Path,required=True)
    p.add_argument('--quick',action='store_true',help='Reserve10 plus balanced DEV4 and their28transforms; no full-set product/robustness credit')
    args=p.parse_args(argv)
    root=args.run_dir.resolve()
    if not root.is_relative_to(config.DATA_PROC):raise ValueError('Checkpoint artifacts must be private derived data')
    root.mkdir(parents=True,exist_ok=False)
    counter=config.DATA_PROC/'mission-reserve-checkpoints.json'
    prior=json.loads(counter.read_text()) if counter.exists() else []
    if len(prior)>=4:raise ValueError('Four reserve checks already consumed')
    prior.append(dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_dir=str(root)))
    temp=counter.with_suffix('.tmp');temp.write_text(json.dumps(prior,indent=2));temp.replace(counter)
    # Every checkpoint report identifies the actual frozen evaluator bytes.
    summary=dict(evaluator_sha256=hashlib.sha256((config.ROOT/'scripts/eval_video.py').read_bytes()).hexdigest(),honest_report_sha256=hashlib.sha256(args.honest.read_bytes()).hexdigest(),reserve_check=len(prior),official_score=False)
    protocol=json.loads(PROTOCOL.read_text());notes=json.loads(ANNOTATIONS.read_text())
    only=quick_groups(protocol['groups'],json.loads(SPLIT.read_text())) if args.quick else []
    scope=['--only',*only] if only else []
    summary.update(scope='partial_checkpoint' if args.quick else 'full_checkpoint',dev_groups=only or 'all30',full_set_requested=not args.quick)
    (root/'scope.json').write_text(json.dumps(summary,indent=2)+'\n')
    summary['reserve']=run(['--subset','reserve','--api',args.api,'--key-file',str(args.key_file)],root/'reserve.json')
    dev_api=run(['--subset','dev','--api',args.api,'--key-file',str(args.key_file),*scope],root/'api-dev.json')
    all_rows=summary['reserve']['records']+dev_api['records']
    ids={r['group_id'] for r in all_rows}
    api_all=score_records(all_rows,[g for g in protocol['groups'] if g['group_id'] in ids],notes['events'],notes['taxonomy'])
    api_all.update(mode='api',subset='partial_checkpoint' if args.quick else 'all',records=all_rows)
    (root/'api-all.json').write_text(json.dumps(api_all,ensure_ascii=False,indent=2)+'\n');summary['api']=api_all
    transforms=[];reencoded_report=None
    for name in TRANSFORMS:
        report=run(['--subset','dev' if args.quick else 'all','--api',args.api,'--key-file',str(args.key_file),'--augment',name,*scope],root/('aug-'+name+'.json'))
        transforms.append(dict(name=name,accuracy=report['accuracy'],failed=report['failed'],n_videos=report['n_videos']))
        if name=='reencode':reencoded_report=report
    augmented=dict(accuracy=sum(r['accuracy'] for r in transforms)/len(transforms),clean_accuracy=dev_api['accuracy'] if args.quick else api_all['accuracy'],all_requested_transforms_complete=not args.quick and all(r['n_videos']==40 and r['failed']==0 for r in transforms),transforms=transforms)
    (root/'augmented.json').write_text(json.dumps(augmented,indent=2)+'\n');summary['augmented']=augmented
    longest=max(protocol['groups'],key=lambda g:g['duration_sec'])
    key=args.key_file.read_text().strip();repeats=[]
    for _ in range(2):
        result,elapsed=api_run(config.ROOT/longest['path'],args.api,key)
        repeats.append(dict(status=result['status'],elapsed_sec=elapsed))
    summary['worst_case_repeats']=repeats
    # All40 uploads reach the worker as anonymous source.mov; then re-encode
    # every source too, so a byte-fingerprint lookup cannot pass this control.
    clean={r['group_id']:r for r in api_all['records']}
    equal=complete=0
    for changed in reencoded_report['records']:
        original=clean[changed['group_id']]
        if original['status']!='done' or changed['status']!='done':continue
        complete+=1;equal+=event_signature(original['result'])==event_signature(changed['result'])
    requested=len(only) if args.quick else 40
    summary['cheat_replay']=dict(requested=requested,completed_pairs=complete,exact_event_signature_equal=equal,passed=complete==equal==requested,full_set_passed=not args.quick and complete==equal==40)
    # Keep reserve detail private. Console/report exposes only aggregates.
    public={k:{s:v.get(s) for s in ['accuracy','recall','fp','failed','hit_2s','type_accuracy','worst_sec']} if k in ['reserve','api'] else v for k,v in summary.items()}
    (root/'summary.json').write_text(json.dumps(public,indent=2)+'\n');print(json.dumps(public,indent=2))


if __name__=='__main__':main()
