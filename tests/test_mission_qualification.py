from scripts.qualify_mission_eval import qualify


def test_non_crossfit_fallback_cannot_acquire_honest_credit():
    recipe='a'*64
    groups=[dict(group_id='g',target=1,video_ids=['X-1'])]
    records=[dict(group_id='g',status='done',result=dict(verdict='violation',violations=[]),elapsed_sec=1)]
    source=dict(records=records,mode='api',subset='dev')
    notes=dict(events=[],taxonomy={})
    trace={'g':dict(analysis=dict(engine='mission-v1-degraded',recipe_sha256=recipe))}
    value=qualify(source,trace,groups,notes,recipe)
    assert value['accuracy']==0 and value['failed']==1 and value['fn']==1
    assert source['records'][0]['status']=='done'  # Raw operational evidence preserved.
    trace['g']['analysis']['engine']='mission-v1'
    assert qualify(source,trace,groups,notes,recipe)['accuracy']==1
