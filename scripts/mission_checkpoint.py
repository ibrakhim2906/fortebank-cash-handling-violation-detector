"""Bounded reserve, robustness and public API checkpoint; private artifacts only."""
import argparse,datetime,hashlib,json,math,os,random,subprocess,time
from pathlib import Path
from scripts.eval_video import PROTOCOL,ANNOTATIONS,SPLIT,score_records,setup
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


def selected_recipe(honest,freeze=None):
    """An explicitly frozen recipe is required for primary runtime credit."""
    value=(freeze if freeze is not None else honest).get('recipe_sha256')
    if value is None and freeze is None:return None
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected a frozen serving recipe SHA-256')
    return value


def repeat_source(groups,records):
    """Select observed clean-API latency, not duration as a latency surrogate."""
    by_id={g['group_id']:g for g in groups}
    completed=[r for r in records if r.get('group_id') in by_id and r.get('status')=='done'
               and type(r.get('elapsed_sec')) in [int,float]
               and math.isfinite(r['elapsed_sec']) and r['elapsed_sec']>=0]
    if completed:
        slowest=max(completed,key=lambda r:(r['elapsed_sec'],r['group_id']))
        return by_id[slowest['group_id']],dict(basis='slowest_completed_clean_api',
                                             observed_clean_api_sec=slowest['elapsed_sec'])
    return max(groups,key=lambda g:g['duration_sec']),dict(basis='longest_duration_fallback_no_completed_clean_api')


def worst_case_repeats(longest,api,key_file,root,expected_recipe):
    """Run two real uploads through the same trace capture as the API sweep."""
    repeats=[]
    for index in range(1,3):
        output=root/f'worst-case-{index}.json'
        report=run(['--subset','all','--only',longest['group_id'],
                    '--api',api,'--key-file',str(key_file)],output)
        records=report.get('records',[])
        if len(records)!=1 or records[0].get('group_id')!=longest['group_id']:
            raise ValueError('Worst-case run did not evaluate exactly the requested recording')
        record=records[0]
        trace_path=output.with_suffix('.traces')/(longest['group_id']+'.json')
        row=dict(status=record['status'],elapsed_sec=record.get('elapsed_sec'),
                 group_id=longest['group_id'],report_path=str(output),trace_path=str(trace_path))
        try:
            trace_bytes=trace_path.read_bytes();trace=json.loads(trace_bytes)
            analysis=trace.get('analysis') or {}
            # Full facts/answers stay in the private trace, never in the console summary.
            row.update(job_id=trace.get('job_id'),analysis={k:analysis.get(k) for k in ['engine','recipe_sha256']},
                       trace_sha256=hashlib.sha256(trace_bytes).hexdigest())
            if trace.get('trace_error'):row['trace_error']=trace['trace_error']
            elapsed=row['elapsed_sec'];traced_elapsed=trace.get('api_elapsed_sec')
            same_elapsed=(type(elapsed) in [int,float] and type(traced_elapsed) in [int,float]
                          and math.isfinite(elapsed) and elapsed>=0
                          and math.isclose(elapsed,traced_elapsed,rel_tol=1e-9,abs_tol=1e-6))
            primary=(record['status']==trace.get('status')=='done'
                     and isinstance(row['job_id'],str) and bool(row['job_id'].strip())
                     and not row.get('trace_error') and same_elapsed and expected_recipe is not None
                     and analysis.get('engine')=='mission-v1'
                     and analysis.get('recipe_sha256')==expected_recipe)
        except (OSError,ValueError,TypeError,AttributeError) as exc:
            row['trace_error']=type(exc).__name__;primary=False
        row.update(primary_recipe_verified=primary,runtime_qualified=primary and row['elapsed_sec']<=200)
        repeats.append(row)
    # A cached/reused job cannot establish two consecutive actual API runs.
    passed=all(r['runtime_qualified'] for r in repeats) and len({r.get('job_id') for r in repeats})==2
    observed=[r['elapsed_sec'] for r in repeats if type(r.get('elapsed_sec')) in [int,float] and math.isfinite(r['elapsed_sec'])]
    qualification=dict(selected_recipe_sha256=expected_recipe,two_worst_case_runs_passed=passed,
                       qualified_worst_sec=max(observed) if passed else None,
                       observed_worst_sec=max(observed) if observed else None,
                       scope='Two selected clean-source API uploads only; transformed worst cases and full API scope remain separate.')
    return repeats,qualification


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.chmod(0o600);temporary.replace(path)


def save_api_stage(root,summary):
    """Persist completed API evidence before the long augmentation stage."""
    common={k:summary[k] for k in ['evaluator_sha256','honest_report_sha256','selected_recipe_sha256',
                                  'reserve_check','scope','full_set_requested']}
    if 'freeze_sha256' in summary:common['freeze_sha256']=summary['freeze_sha256']
    common.update(official_score=False,stage='clean_api_and_repeats_complete',stage_full_aug_pending=True,
                  all_requested_transforms_complete=False,worst_clean_api_sec=summary['api']['worst_sec'])
    if summary.get('api_only'):
        common.update(api_only=True,full_aug_requested=False,augmentation_stage='deferred_by_api_only',
                      robustness_credit_eligible=False)
    # Read only operational provenance; do not expose reserve predictions/facts.
    jobs=[]
    for folder in ['reserve.traces','api-dev.traces']:
        for path in sorted((root/folder).glob('*.json')):
            trace=json.loads(path.read_text());analysis=trace.get('analysis') or {}
            jobs.append({k:trace.get(k) for k in ['job_id','status','api_elapsed_sec']}|
                        {k:analysis.get(k) for k in ['engine','recipe_sha256']})
    runtime=common|summary['runtime_qualification']|dict(worst_case_repeat_selection=summary['worst_case_repeat_selection'],
                worst_case_repeats=summary['worst_case_repeats'])
    atomic_json(root/'runtime-qualification.json',runtime)
    aggregates=['mode','subset','n_videos','completed','failed','accuracy','recall','fp','hit_2s','type_accuracy','worst_sec']
    stage=common|dict(api={k:summary['api'].get(k) for k in aggregates},
                     reserve={k:summary['reserve'].get(k) for k in aggregates},
                     clean_job_provenance=jobs,clean_job_ids=[j['job_id'] for j in jobs if j['job_id']],
                     runtime_qualification=summary['runtime_qualification'],
                     worst_case_repeat_selection=summary['worst_case_repeat_selection'],
                     worst_case_repeats=summary['worst_case_repeats'])
    atomic_json(root/'api-stage-summary.json',stage)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--api',required=True);p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--honest',type=Path,required=True)
    p.add_argument('--freeze',type=Path,help='Frozen serving recipe JSON; required for runtime credit unless --honest includes recipe_sha256')
    p.add_argument('--quick',action='store_true',help='Reserve10 plus balanced DEV4 and their28transforms; no full-set product/robustness credit')
    p.add_argument('--api-only',action='store_true',help='Full40 clean public-API uploads and two slowest-source repeats; defer all280transforms with no robustness credit')
    args=p.parse_args(argv)
    if args.api_only and args.quick:p.error('--api-only requires full40 scope and cannot be combined with --quick')
    expected_recipe=selected_recipe(json.loads(args.honest.read_text()),json.loads(args.freeze.read_text()) if args.freeze else None)
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
    summary['selected_recipe_sha256']=expected_recipe
    if args.freeze:summary['freeze_sha256']=hashlib.sha256(args.freeze.read_bytes()).hexdigest()
    protocol=json.loads(PROTOCOL.read_text());notes=json.loads(ANNOTATIONS.read_text())
    only=quick_groups(protocol['groups'],json.loads(SPLIT.read_text())) if args.quick else []
    scope=['--only',*only] if only else []
    summary.update(scope='partial_checkpoint' if args.quick else 'full_checkpoint',dev_groups=only or 'all30',full_set_requested=not args.quick)
    if args.api_only:summary.update(scope='full_clean_api_only',api_only=True,full_aug_requested=False)
    (root/'scope.json').write_text(json.dumps(summary,indent=2)+'\n')
    summary['reserve']=run(['--subset','reserve','--api',args.api,'--key-file',str(args.key_file)],root/'reserve.json')
    dev_api=run(['--subset','dev','--api',args.api,'--key-file',str(args.key_file),*scope],root/'api-dev.json')
    all_rows=summary['reserve']['records']+dev_api['records']
    ids={r['group_id'] for r in all_rows}
    api_all=score_records(all_rows,[g for g in protocol['groups'] if g['group_id'] in ids],notes['events'],notes['taxonomy'])
    api_all.update(mode='api',subset='partial_checkpoint' if args.quick else 'all',records=all_rows)
    (root/'api-all.json').write_text(json.dumps(api_all,ensure_ascii=False,indent=2)+'\n');summary['api']=api_all
    selected,selection=repeat_source(protocol['groups'],all_rows)
    selection['group_id']=selected['group_id'];summary['worst_case_repeat_selection']=selection
    repeats,qualification=worst_case_repeats(selected,args.api,args.key_file,root,expected_recipe)
    summary['worst_case_repeats']=repeats
    summary['runtime_qualification']=qualification
    save_api_stage(root,summary)
    if args.api_only:
        # Preserve full clean/API provenance and both real repeats, while making
        # the unmeasured robustness stage impossible to mistake for a pass.
        deferred=dict(clean_accuracy=api_all['accuracy'],all_requested_transforms_complete=False,
                      deferred=True,stage='deferred_by_api_only',transforms=[],completed_transforms=0,
                      planned_transforms=list(TRANSFORMS),robustness_credit_eligible=False)
        atomic_json(root/'augmented.json',deferred)
        public=json.loads((root/'api-stage-summary.json').read_text())
        public.update(augmented=deferred,robustness_credit_eligible=False,
                      cheat_replay=dict(not_run=True,passed=False,full_set_passed=False,
                                        reason='Re-encode stage deferred by --api-only'))
        atomic_json(root/'summary.json',public)
        print(json.dumps(public,indent=2))
        return
    transforms=[];reencoded_report=None
    for name in TRANSFORMS:
        report=run(['--subset','dev' if args.quick else 'all','--api',args.api,'--key-file',str(args.key_file),'--augment',name,*scope],root/('aug-'+name+'.json'))
        transforms.append(dict(name=name,accuracy=report['accuracy'],failed=report['failed'],n_videos=report['n_videos']))
        if name=='reencode':reencoded_report=report
    augmented=dict(accuracy=sum(r['accuracy'] for r in transforms)/len(transforms),clean_accuracy=dev_api['accuracy'] if args.quick else api_all['accuracy'],all_requested_transforms_complete=not args.quick and all(r['n_videos']==40 and r['failed']==0 for r in transforms),transforms=transforms)
    (root/'augmented.json').write_text(json.dumps(augmented,indent=2)+'\n');summary['augmented']=augmented
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
