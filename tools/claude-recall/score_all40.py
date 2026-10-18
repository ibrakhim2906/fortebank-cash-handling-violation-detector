"""Score raw benchmark outputs for all 40 supplied recordings with the frozen evaluator.

Usage: uv run python tools/claude-recall/score_all40.py <raw-dir> [<raw-dir> ...]
Prints aggregates only (no per-video rows), as for reserve reporting.
"""
import json,sys
from pathlib import Path
from scripts.eval_video import ANNOTATIONS,PROTOCOL,score_records,match_events,type_key
from src import config


def main(dirs):
    mapping={r['input_id']:r['group_id'] for r in json.loads((config.DATA_PROC/'mission-staging-inputs/local-eval-map.json').read_text())}
    rows={}
    for d in dirs:
        for p in Path(d).glob('*.json'):
            v=json.loads(p.read_text());gid=mapping[v['input_id']]
            if gid in rows:raise ValueError('duplicate group')
            rows[gid]=dict(group_id=gid,status=v['status'],result=v['result'] if v['status']=='done' else None,elapsed_sec=None)
    protocol=json.loads(PROTOCOL.read_text());notes=json.loads(ANNOTATIONS.read_text())
    groups=[g for g in protocol['groups'] if g['group_id'] in rows]
    if len(groups)!=40 or len(rows)!=40:raise ValueError(f'need 40 groups, got {len(rows)}')
    report=score_records(list(rows.values()),groups,notes['events'],notes['taxonomy'])
    typed=0
    for r in rows.values():
        gold=[e for e in notes['events'] if e['group_id']==r['group_id']];pred=r['result']['violations'] if r['status']=='done' else []
        typed+=sum(type_key(pred[j]['type'],notes['taxonomy']) in gold[i]['type_ids'] for i,j,_ in match_events(gold,pred,2))
    keys=['n_videos','accuracy','tp','fn','fp','tn','failed','labelled_events','hit_1s','hit_2s','type_accuracy']
    out={k:report[k] for k in keys};out['correct']=round(report['accuracy']*40);out['localized_2s']=round(report['hit_2s']*report['labelled_events']);out['typed_coverage']=typed
    print(json.dumps(out))


if __name__=='__main__':main(sys.argv[1:])
