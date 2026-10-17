"""Combine qualified physical observations; preserve distinct event episodes."""


def combine(rule_events,observations,rules):
    result=[]
    for event in sorted(rule_events+observations,key=lambda r:(-r['confidence'],r['timecode_sec'])):
        spec=rules['types'].get(event['type_id'])
        if not spec:continue
        if event['origin']!='temporal_rule' and not spec.get('auditor_review_enabled',False):continue
        if any(old['type_id']==event['type_id'] and abs(old['timecode_sec']-event['timecode_sec'])<=4 for old in result):continue
        result.append(dict(event))
    if len(result)>64:raise ValueError('Distinct events exceed the documented result-size bound')
    return result
