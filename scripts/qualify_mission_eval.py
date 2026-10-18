"""Keep non-crossfit numeric fallbacks out of honest pretrained-model scores."""
import argparse,hashlib,json
from pathlib import Path
from scripts.eval_video import PROTOCOL,ANNOTATIONS,score_records


def qualify(report,traces,groups,notes,expected_recipe):
    if not expected_recipe or len(expected_recipe)!=64:raise ValueError('A frozen serving recipe is required')
    rows=[];excluded=[]
    for source in report['records']:
        row=json.loads(json.dumps(source));analysis=(traces.get(row['group_id']) or {}).get('analysis') or {}
        if row['status']=='done':
            if analysis.get('recipe_sha256') not in [None,expected_recipe]:raise ValueError('Serving recipe changed during benchmark')
            if analysis.get('engine')!='mission-v1' or analysis.get('recipe_sha256')!=expected_recipe:
                excluded.append(dict(group_id=row['group_id'],engine=analysis.get('engine','missing'),reason='not a verified fixed pretrained primary prediction'))
                row.update(status='failed',result=None,evaluation_exclusion='non_crossfit_fallback_or_missing_provenance')
        rows.append(row)
    ids={r['group_id'] for r in rows};selected=[g for g in groups if g['group_id'] in ids]
    value=score_records(rows,selected,notes['events'],notes['taxonomy'])
    value.update(mode=report['mode'],subset=report['subset'],records=rows,oof_eligibility_verified=True,
                 qualification='Fixed pretrained primary predictions only; non-crossfit fallback/missing provenance count as failures in the denominator.',
                 excluded_from_honest=excluded,recipe_sha256=expected_recipe,
                 api_status_is_separate='The original raw API report retains actual service done/failed outcomes.')
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['eval','traces','freeze','json']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv);read=lambda p:json.loads(p.read_text())
    traces={p.stem:read(p) for p in args.traces.glob('*.json')};report=read(args.eval)
    value=qualify(report,traces,read(PROTOCOL)['groups'],read(ANNOTATIONS),read(args.freeze)['recipe_sha256'])
    value['raw_report_sha256']=hashlib.sha256(args.eval.read_bytes()).hexdigest()
    args.json.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['accuracy','recall','fp','failed','qualification']}))


if __name__=='__main__':main()
