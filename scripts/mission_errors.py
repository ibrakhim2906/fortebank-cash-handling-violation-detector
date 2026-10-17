"""Inspect development error clusters; never reads reserve job errors."""
import argparse,json
from collections import Counter
from pathlib import Path
from scripts.eval_video import ANNOTATIONS,SPLIT,match_events,type_key


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--eval',type=Path,required=True);p.add_argument('--traces',type=Path,required=True);p.add_argument('--markdown',type=Path,required=True);p.add_argument('--freeze',type=Path);args=p.parse_args(argv)
    report=json.loads(args.eval.read_text());notes=json.loads(ANNOTATIONS.read_text());dev=set(json.loads(SPLIT.read_text())['dev'])
    if any(r['group_id'] not in dev for r in report['records']):raise ValueError('Reserve per-video inspection is forbidden during iteration')
    expected=json.loads(args.freeze.read_text())['recipe_sha256'] if args.freeze else None
    counts=Counter();engines=Counter();rows=[];recipes=set()
    by_record={r['group_id']:r for r in report['records']}
    for video in report['per_video']:
        key=video['group_id'];record=by_record[key];gold=[e for e in notes['events'] if e['group_id']==key]
        trace_path=args.traces/(key+'.json');trace=json.loads(trace_path.read_text()) if trace_path.exists() else {};analysis=trace.get('analysis') or {}
        engines[analysis.get('engine','missing_trace')]+=1
        if analysis.get('recipe_sha256'):recipes.add(analysis['recipe_sha256'])
        if expected and analysis.get('recipe_sha256') not in [None,expected]:raise ValueError('Mixed serving recipe in evaluation')
        predictions=(record.get('result') or {}).get('violations',[])
        if video['status']!='done':counts['processing_failure']+=1
        elif video['target'] and not video['predicted']:
            for kind in sorted({t for e in gold for t in e['type_ids']}):counts['missed:'+kind]+=1
        elif not video['target'] and video['predicted']:
            for kind in sorted({type_key(e['type'],notes['taxonomy']) or 'unmapped' for e in predictions}):counts['false_alarm:'+kind]+=1
        if video['target'] and video['predicted']:
            matches=match_events(gold,predictions,2)
            counts['localization_miss']+=len(gold)-len(matches)
            counts['type_mismatch']+=sum(type_key(predictions[j]['type'],notes['taxonomy']) not in gold[i]['type_ids'] for i,j,_ in matches)
        answers=[str(c.get('kind') or '')+':'+str((c.get('answer') or {}).get('support',(c.get('answer') or {}).get('visibility',''))) for c in analysis.get('auditor',[])]
        found=','.join(type_key(e['type'],notes['taxonomy']) or 'unmapped' for e in predictions) or 'none'
        truth=','.join(sorted({t for e in gold for t in e['type_ids']})) or 'clean'
        rows.append('|'+','.join(video['aliases'])+'|'+truth+'|'+str(video['status'])+' / '+found+'|'+str(video['hits_2s'])+'/'+str(video['labelled_events'])+'|'+analysis.get('engine','missing')+'|'+('; '.join(answers) or '; '.join(analysis.get('warnings',[]))).replace('|','/')+'|')
    header='# Development errors — measured system results\n\n'+report['disclosure']+'\n\n'
    header+='Engines: '+json.dumps(engines)+'. Recipe hashes: '+json.dumps(sorted(recipes))+'.\n\n'
    header+='| Video | Human target types | Status / predicted types | Time hits | Engine | Auditor / failure |\n|---|---|---|---|---|---|\n'
    args.markdown.write_text(header+'\n'.join(rows)+'\n\nClusters (counts can overlap for multi-type events): '+json.dumps(counts.most_common(),ensure_ascii=False)+'\n')
    print(json.dumps(dict(engines=engines,clusters=counts.most_common(),mixed_recipes=len(recipes)>1),ensure_ascii=False))


if __name__=='__main__':main()
