"""Re-score excluded-setup predictions; calibrations fit only training records."""
import argparse,json,statistics
from pathlib import Path
from scripts.eval_video import ANNOTATIONS,PROTOCOL,SPLIT,match_events,score_records,type_key


def fitted_offsets(records,groups,events,taxonomy):
    """Only supplied training groups contribute; insufficient support => no fit."""
    by_id={r['group_id']:r for r in records};values={}
    for g in groups:
        r=by_id[g['group_id']]
        if r.get('status')!='done':continue
        gold=[e for e in events if e['group_id']==g['group_id']];pred=r['result']['violations']
        for i,j,_ in match_events(gold,pred,8):
            key=type_key(pred[j]['type'],taxonomy)
            if key in gold[i]['type_ids']:values.setdefault(key,[]).append(gold[i]['timecode_sec']-pred[j]['timecode_sec'])
    offsets={}
    for key,rows in values.items():
        middle=statistics.median(rows)
        if len(rows)>=3 and abs(middle)>.5 and max(rows)-min(rows)<=2:offsets[key]=middle
    return offsets


def crossfit(report,protocol,annotations,split,calibrate=False):
    if report.get('mode')!='stub' and report.get('oof_eligibility_verified') is not True:
        raise ValueError('Qualify fixed-primary/fold-trained components before claiming honest scores')
    records={r['group_id']:r for r in report['records']};groups={g['group_id']:g for g in protocol['groups']}
    if not set(split['dev'])<=set(records):raise ValueError('Incomplete dev predictions')
    output=[];folds=[];seen=set()
    for fold in split['folds']:
        train=set(fold['train']);valid=set(fold['validation'])
        if train&valid or (train|valid)!=set(split['dev']) or seen&valid:raise ValueError('Invalid excluded-setup fold')
        offsets=fitted_offsets(list(records.values()),[groups[g] for g in train],annotations['events'],annotations['taxonomy']) if calibrate else {}
        fold_rows=[]
        for key in sorted(valid):
            row=json.loads(json.dumps(records[key]));row['fold']=fold['name']
            if row.get('status')=='done':
                for event in row['result']['violations']:
                    offset=offsets.get(type_key(event['type'],annotations['taxonomy']),0)
                    event['timecode_sec']=min(groups[key]['duration_sec'],max(0,event['timecode_sec']+offset))
            fold_rows.append(row)
        scores=score_records(fold_rows,[groups[g] for g in sorted(valid)],annotations['events'],annotations['taxonomy'])
        folds.append(dict(name=fold['name'],n_train=len(train),n_validation=len(valid),accuracy=scores['accuracy'],offsets=offsets));output+=fold_rows;seen|=valid
    if seen!=set(split['dev']):raise ValueError('Missing excluded predictions')
    value=score_records(output,[groups[g] for g in split['dev']],annotations['events'],annotations['taxonomy'])
    value.update(validation_kind='leave-one-setup-out',fold_integrity_verified=True,folds=folds,records=output,
                 fitting='fold-training-only offsets' if calibrate else 'fixed pretrained policy; no supervised fit',
                 model_selection_exposure='All supplied videos were historically inspected. Prompts/rules may overfit despite group exclusion.',
                 independent_test=False,mode=report['mode'])
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--calibrate',action='store_true');p.add_argument('--json',type=Path,required=True);args=p.parse_args(argv)
    read=lambda p:json.loads(p.read_text());value=crossfit(read(args.predictions),read(PROTOCOL),read(ANNOTATIONS),read(SPLIT),args.calibrate)
    args.json.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['accuracy','recall','fp','hit_1s','hit_2s','type_accuracy','folds']}))


if __name__=='__main__':main()
