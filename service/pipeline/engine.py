"""Small declarative temporal engine with explicit unknown/absence semantics."""
import math,re


def known_window(rows,start,end,fact,equals,min_observed=0,max_gap=.6):
    samples=[r for r in rows if start-1e-6<=r['timestamp_sec']<=end+1e-6]
    if not samples:return False
    times=[r['timestamp_sec'] for r in samples]
    if times[-1]-times[0]+1e-6<min_observed or times[0]-start>max_gap or end-times[-1]>max_gap:return False
    if any(b-a>max_gap for a,b in zip(times,times[1:])):return False
    return all(r['facts'].get(fact) is equals for r in samples)


def duration_match(rows,start,end,fact,equals,min_sec):
    onset=previous=None
    for row in rows:
        t=row['timestamp_sec']
        if not start<=t<=end:continue
        if row['facts'].get(fact) is not equals or previous is not None and t-previous>.6:onset=None
        if row['facts'].get(fact) is equals:
            if onset is None:onset=t
            if t-onset+1e-6>=min_sec:return True
        previous=t
    return False


def conditions(rows,index,logic):
    row=rows[index];t=row['timestamp_sec'];facts=row['facts'];end=rows[-1]['timestamp_sec'];begin=rows[0]['timestamp_sec']
    if any(facts.get(k) is not True for k in logic.get('require_all',[])):return False
    if logic.get('require_any') and not any(facts.get(k) is True for k in logic['require_any']):return False
    if any(facts.get(k) is not False for k in logic.get('require_none',[])):return False
    trigger=logic['trigger']
    minimum=float(logic.get('min_duration_sec',0))
    if minimum>0 and (t+minimum>end or not known_window(rows,t,t+minimum,trigger,True,minimum)):return False
    for direction in ['before','after']:
        for prefix in ['require_','absent_']:
            spec=logic.get(prefix+direction)
            if not spec:continue
            span=float(spec.get('within_sec',10));start,stop=(max(begin,t-span),t) if direction=='before' else (t,min(end,t+span))
            if prefix=='absent_':
                if not known_window(rows,start,stop,spec['fact'],False,float(spec.get('min_observed_sec',3))):return False
            elif not duration_match(rows,start,stop,spec['fact'],spec.get('equals',True),float(spec.get('min_sec',0))):return False
    return True


def evaluate(rows,rules,*,include_disabled=False):
    if not rows:return []
    times=[r['timestamp_sec'] for r in rows]
    if any(not isinstance(t,(int,float)) or not math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Fact timeline must have increasing finite source times')
    result=[]
    for key,rule in rules['types'].items():
        if not rule.get('enabled') and not include_disabled:continue
        logic=rule.get('logic')
        if not logic:continue
        branches=[logic]+([logic['alt']] if 'alt' in logic else [])
        last=-math.inf
        for i,row in enumerate(rows):
            for branch in branches:
                trigger=branch['trigger'];active=row['facts'].get(trigger) is True
                if not active or i>0 and rows[i-1]['facts'].get(trigger) is True:continue
                if not conditions(rows,i,branch) or row['timestamp_sec']-last<float(branch.get('cooldown_sec',10)):continue
                offset=branch.get('timecode','trigger');match=re.fullmatch(r'trigger([+-]\d+(?:\.\d+)?)?',offset)
                if not match:raise ValueError('Unsupported rule time expression')
                t=row['timestamp_sec']+float(match[1] or 0)+float(rule.get('timecode_offset_sec',0));t=max(times[0],min(times[-1],t))
                result.append(dict(type_id=key,timecode_sec=t,trigger_sec=row['timestamp_sec'],confidence=rule.get('confidence',.75),origin='temporal_rule',enabled=bool(rule.get('enabled'))));last=row['timestamp_sec'];break
    return sorted(result,key=lambda r:(-r['confidence'],r['timecode_sec'],r['type_id']))
