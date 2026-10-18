"""Frozen mission evaluator. Labels and file identities never enter inference."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src import config

DISCLOSURE = 'All 40 supplied recordings were previously reviewed. Reserve is prospective regression only; not independent. Setup metadata is team supplied.'
PROTOCOL = config.DATA_PROC / 'fortebank-20261017/video_protocol.json'
ANNOTATIONS = config.DATA_PROC / 'team-annotations-v5/v1/normalized.json'
SPLIT = config.DATA_PROC / 'mission-validation-v1.json'


def setup(group):
    numbers = {int(v.split('-')[-1]) for v in group['video_ids']}
    if numbers & {33, 40, 44, 46}: return 'c'
    if numbers & {30, 31, 32}: return 'b'
    return 'a'


def freeze_split(groups, destination=SPLIT):
    """Whole-content-group stratification with actual setup coverage, once only."""
    canonical = [{'group_id': g['group_id'], 'target': g['target'], 'setup': setup(g)} for g in groups]
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    if destination.exists():
        value = json.loads(destination.read_text())
        if value['protocol_fingerprint'] != fingerprint: raise ValueError('Frozen split input changed')
        return value
    rng = random.Random(config.SEED); reserve = []
    for target in [0, 1]:
        pool = sorted([g for g in groups if g['target'] == target], key=lambda g: g['group_id'])
        rng.shuffle(pool); chosen = []
        for camera in sorted({setup(g) for g in pool}):
            candidate = next(g for g in pool if setup(g) == camera)
            chosen.append(candidate)
        chosen.extend(g for g in pool if g not in chosen)
        reserve.extend(g['group_id'] for g in chosen[:5])
    value = dict(schema_version=1, seed=config.SEED, protocol_fingerprint=fingerprint,
                 disclosure=DISCLOSURE, reserve=reserve,
                 dev=[g['group_id'] for g in groups if g['group_id'] not in reserve],
                 folds=[dict(name='leave-setup-'+s,
                             train=[g['group_id'] for g in groups if g['group_id'] not in reserve and setup(g)!=s],
                             validation=[g['group_id'] for g in groups if g['group_id'] not in reserve and setup(g)==s])
                        for s in sorted({setup(g) for g in groups if g['group_id'] not in reserve})])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as f: json.dump(value, f, indent=2)
    return value


def match_events(truth, predictions, tolerance):
    """Maximum cardinality, then minimum time error; one prediction per truth."""
    if not truth or not predictions: return []
    delta = np.array([[abs(float(t['timecode_sec'])-float(p['timecode_sec'])) for p in predictions] for t in truth])
    # Invalid pairs cost more than all possible valid distances together.
    cost = np.where(delta <= tolerance, delta, (len(truth)+len(predictions)+1)*(tolerance+1))
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(delta[r,c])) for r,c in zip(rows,cols) if delta[r,c] <= tolerance]


def type_key(value, taxonomy):
    if value in taxonomy: return value
    for key, names in taxonomy.items():
        if value in (names['name'], names['name_en'], names['name']+' / '+names['name_en']): return key
    return None


def score_records(records, groups, events, taxonomy):
    by_id = {r['group_id']: r for r in records}
    if len(by_id) != len(records): raise ValueError('Duplicate prediction rows')
    if set(by_id) != {g['group_id'] for g in groups}: raise ValueError('Every requested group must be scored exactly once')
    tp=tn=fp=fn=failed=correct=hits1=hits2=type_hits=labelled=reported=extra=0
    times=[]; per_video=[]; per_type={}
    for g in groups:
        r=by_id[g['group_id']]; ok=r.get('status')=='done'; result=r.get('result') or {}
        pred=result.get('violations',[]) if ok else []
        actual=bool(g['target']); predicted=result.get('verdict')=='violation' if ok else None
        failed+=not ok; correct+=ok and actual==predicted
        tp+=ok and actual and predicted; tn+=ok and not actual and not predicted
        fp+=ok and not actual and predicted; fn+=actual and (not ok or not predicted)
        gold=[e for e in events if e['group_id']==g['group_id']]
        one=match_events(gold,pred,1); two=match_events(gold,pred,2)
        hits1+=len(one); hits2+=len(two); labelled+=len(gold); reported+=len(pred); extra+=len(pred)-len(two)
        for i,j,_ in two:
            hit=type_key(pred[j]['type'],taxonomy) in gold[i]['type_ids'];type_hits+=hit
        for e in gold:
            for key in e['type_ids']:
                counts=per_type.setdefault(key,dict(labelled=0,localized=0,typed=0));counts['labelled']+=1
                pair=next((p for p in two if gold[p[0]] is e),None)
                if pair:
                    counts['localized']+=1;counts['typed']+=type_key(pred[pair[1]]['type'],taxonomy)==key
        elapsed=r.get('elapsed_sec')
        if elapsed is not None and np.isfinite(elapsed) and elapsed>=0: times.append(float(elapsed))
        per_video.append(dict(group_id=g['group_id'],aliases=g['video_ids'],setup=setup(g),target=g['target'],
                              status=r.get('status'),predicted=predicted,hits_2s=len(two),labelled_events=len(gold),
                              extra_events=len(pred)-len(two),elapsed_sec=elapsed,annotation_flags=[e['flags'] for e in gold if e.get('flags')]))
    n=len(groups); positives=sum(g['target'] for g in groups)
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/positives if positives else 0
    return dict(schema_version=1,disclosure=DISCLOSURE,n_videos=n,completed=n-failed,failed=failed,
                accuracy=correct/n if n else 0,recall=recall,precision=precision,f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                tp=tp,tn=tn,fp=fp,fn=fn,labelled_events=labelled,reported_events=reported,extra_events=extra,
                hit_1s=hits1/labelled if labelled else 0,hit_2s=hits2/labelled if labelled else 0,
                event_precision=hits2/reported if reported else 0,type_accuracy=type_hits/hits2 if hits2 else 0,
                worst_sec=max(times) if len(times)==n else None,total_sec=sum(times),per_type=per_type,per_video=per_video)


def api_run(path,base,key,timeout=300):
    import requests
    started=time.monotonic();headers={'X-API-Key':key}
    with path.open('rb') as f:
        response=requests.post(base.rstrip('/')+'/api/v1/analyze',headers=headers,files={'file':('upload.mov',f,'video/quicktime')},timeout=min(90,timeout))
    response.raise_for_status();identifier=response.json()['job_id']
    while time.monotonic()-started<timeout:
        response=requests.get(base.rstrip('/')+'/api/v1/jobs/'+identifier,headers=headers,timeout=min(15,timeout-(time.monotonic()-started)))
        response.raise_for_status();value=response.json()
        if value['status']!='processing': return value,time.monotonic()-started
        time.sleep(.3)
    raise TimeoutError('API total deadline')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=PROTOCOL);p.add_argument('--annotations',type=Path,default=ANNOTATIONS)
    p.add_argument('--subset',choices=['dev','reserve','all'],default='dev');p.add_argument('--only',nargs='+')
    p.add_argument('--videos',type=Path);p.add_argument('--labels',type=Path)
    p.add_argument('--augment',choices=['flip','crop','rotate_plus','rotate_minus','dark','resolution','reencode'])
    modes=p.add_mutually_exclusive_group();modes.add_argument('--stub',action='store_true');modes.add_argument('--cached',type=Path);modes.add_argument('--full',action='store_true');modes.add_argument('--api')
    p.add_argument('--key-file',type=Path);p.add_argument('--json',type=Path,required=True)
    args=p.parse_args(argv);groups=json.loads(args.protocol.read_text())['groups'];notes=json.loads(args.annotations.read_text());split=freeze_split(groups)
    if bool(args.videos)!=bool(args.labels):raise ValueError('Supply both --videos and --labels')
    if args.videos:
        from scripts.mission_validation import external_groups
        groups=external_groups(args.videos,args.labels)
        known={g['group_id'] for g in groups};notes['events']=[e for e in notes['events'] if e['group_id'] in known]
        represented={e['group_id'] for e in notes['events']}
        notes['events'] += [dict(group_id=g['group_id'],timecode_sec=a['timestamp_sec'],type_ids=[],flags=['external_weak_time_only']) for g in groups if g['target'] and g['group_id'] not in represented for a in g['timestamp_annotations']]
    selected=groups if args.subset=='all' or args.videos else [g for g in groups if g['group_id'] in split[args.subset]]
    if args.only:selected=[g for g in selected if g['group_id'] in args.only or set(g['video_ids'])&set(args.only)]
    if not selected:raise ValueError('Empty evaluation selection')
    records=[];key=args.key_file.read_text().strip() if args.key_file else os.environ.get('API_KEY','')
    for g in selected:
        started=time.monotonic();row=dict(group_id=g['group_id'],status='failed')
        try:
            path=config.ROOT/g['path']
            if args.augment:
                from scripts.mission_validation import augment
                path=augment(path,args.augment,config.DATA_PROC/'mission-robustness'/args.augment/(g['group_id']+'.mp4'))
                # Generation is offline preparation, never included in inference latency.
                started=time.monotonic()
            if args.stub: value={'verdict':'no_violation','violations':[]}
            elif args.cached:
                cached=json.loads((args.cached/(g['group_id']+'.json')).read_text())
                row.update(cached)
                if row['group_id']!=g['group_id']:raise ValueError('Wrong prediction cache identity')
                records.append(row);continue
            elif args.api:
                response,elapsed=api_run(path,args.api,key);row.update(status=response['status'],result={k:response[k] for k in ['verdict','violations']},elapsed_sec=elapsed)
                records.append(row);continue
            else:
                from service.pipeline.analyze import analyze
                value=analyze(path,deadline=started+200)
            from src.video_api import Result
            row.update(status='done',result=Result.model_validate(value).model_dump())
        except Exception as exc:row.update(status='failed',error=type(exc).__name__)
        row['elapsed_sec']=time.monotonic()-started;records.append(row)
    value=score_records(records,selected,notes['events'],notes['taxonomy']);value['mode']='stub' if args.stub else 'cached' if args.cached else 'api' if args.api else 'full'
    value['subset']=args.subset;value['records']=records;value['evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    value['augmentation']=args.augment
    args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ['per_video','records','per_type']},ensure_ascii=False))


if __name__=='__main__':main()
