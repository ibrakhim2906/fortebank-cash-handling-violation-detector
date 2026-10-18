from service.pipeline.engine import evaluate


def timeline(**facts):
    n=max(len(v) for v in facts.values())
    return [dict(timestamp_sec=i*.5,facts={k:v[i] for k,v in facts.items()}) for i in range(n)]


def rule(logic,enabled=True):return {'types':{'concern':dict(enabled=enabled,logic=logic)}}


def test_leaving_requires_cash_not_just_a_bag():
    logic=dict(trigger='leaving',require_all=['cash_visible'],require_none=['active_handling'])
    rows=timeline(leaving=[False,True,False],cash_visible=[None,True,None],active_handling=[None,False,None])
    assert evaluate(rows,rule(logic))[0]['timecode_sec']==.5
    rows[1]['facts']['cash_visible']=None;rows[1]['facts']['bag_visible']=True
    assert evaluate(rows,rule(logic))==[]


def test_unknown_does_not_satisfy_absence_or_negative_prerequisite():
    logic=dict(trigger='received',absent_before=dict(fact='id_shown',within_sec=5,min_observed_sec=3))
    rows=timeline(received=[False]*8+[True],id_shown=[False]*9)
    assert len(evaluate(rows,rule(logic)))==1
    rows[3]['facts']['id_shown']=None
    assert evaluate(rows,rule(logic))==[]
    assert evaluate(timeline(leaving=[True,False],open=[None,False]),rule(dict(trigger='leaving',require_none=['open'])))==[]


def test_disabled_rules_stay_silent_and_states_need_continuous_duration():
    rows=timeline(open=[True,False,True,True,True])
    rules=rule(dict(trigger='open',min_duration_sec=1),False)
    assert evaluate(rows,rules)==[]
    preview=evaluate(rows,rules,include_disabled=True)
    assert len(preview)==1 and preview[0]['timecode_sec']==1 and not preview[0]['enabled']


def test_cooldown_offset_and_distinct_episodes():
    rows=timeline(leaving=[False,True,False,True,False,False,True])
    rules=rule(dict(trigger='leaving',cooldown_sec=2,timecode='trigger-0.5'))
    result=evaluate(rows,rules)
    assert [e['timecode_sec'] for e in result]==[0,2.5]


def test_after_requirement_does_not_bridge_unknown_gaps():
    rows=timeline(left=[True]+[False]*6,counting=[False,True,None,True,True,False,False])
    rules=rule(dict(trigger='left',require_after=dict(fact='counting',min_sec=1,within_sec=3)))
    assert evaluate(rows,rules)==[]


def test_alt_does_not_remove_primary_prerequisites():
    rows=timeline(a=[True,False],b=[True,False],cash=[None,None])
    rules=rule(dict(trigger='a',require_all=['cash'],alt=dict(trigger='b',require_all=['cash'])))
    assert evaluate(rows,rules)==[]
