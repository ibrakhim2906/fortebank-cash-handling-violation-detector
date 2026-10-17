import json,time
import numpy as np,pytest
from service.pipeline import auditor as a
from service.pipeline import analyze as p
from src.video_api import Result


def frames(n=20):
    image=p.encode_jpeg(np.zeros((80,60,3),np.uint8))
    return [dict(timestamp_sec=i*.5,source_frame_index=i*15,image=image) for i in range(n)]


def reply(value):return dict(model='qwen-vl',choices=[dict(finish_reason='stop',message=dict(content=json.dumps(value)))])


def candidate(kind='cash_left_out'):
    return dict(kind=kind,onset_frame=6,evidence_frame=8,support='clear',counterexamples_excluded=True,complete_phase_visible=True,visible_fact='На столе остаются банкноты после ухода сотрудника.')


def test_confirmed_concern_uses_actual_source_evidence_and_bilingual_type():
    def request(url,payload,timeout):
        body=json.loads(payload);schema=body['response_format']['json_schema']['schema']
        if 'observations' in schema['properties']:return reply(dict(visibility='clear',observations=[candidate()]))
        return reply(dict(support='clear',onset_frame=2,evidence_frame=4,complete_phase_visible=False,visible_fact='Банкноты остаются на рабочей поверхности после ухода.'))
    samples=frames();observed,log=a.confirmed_observations(samples,time.monotonic()+60,requester=request)
    assert len(observed)==1 and len(log)==2
    rules={'types':{'cash_left_out':dict(name='Наличные вне хранения',name_en='Cash outside storage',context_source=None)}}
    result=Result.model_validate(p.build_result(observed,samples,rules)).model_dump()
    event=result['violations'][0]
    assert event['type']=='Наличные вне хранения / Cash outside storage'
    assert event['timecode_sec']==event['evidence_frames'][0]['timestamp_sec']
    assert event['timecode_sec'] in [f['timestamp_sec'] for f in samples]


def test_sparse_images_cannot_establish_omission_even_if_model_claims_complete():
    calls=[]
    def request(*args):calls.append(1);return reply(dict(visibility='clear',observations=[candidate('no_id_check')]))
    assert a.confirmed_observations(frames(),time.monotonic()+60,requester=request)[0]==[]
    assert len(calls)==1


def test_unusable_visibility_is_not_a_clean_verdict():
    def request(*args):return reply(dict(visibility='insufficient',observations=[]))
    with pytest.raises(a.UnusableVisibilityError):a.confirmed_observations(frames(),time.monotonic()+60,requester=request)


@pytest.mark.parametrize('error',[TimeoutError,a.UnusableVisibilityError])
def test_model_outage_uses_existing_cpu_fallback(monkeypatch,tmp_path,error):
    monkeypatch.setattr(p,'perception',lambda *args:({'frames':[]},frames()))
    monkeypatch.setattr(a,'confirmed_observations',lambda *args,**kw:(_ for _ in ()).throw(error()))
    monkeypatch.setattr(p,'cpu_fallback',lambda *args:dict(verdict='no_violation',violations=[]))
    result,meta=p.analyze_detailed(tmp_path/'anonymous.mov',time.monotonic()+60)
    assert result['verdict']=='no_violation' and meta['engine']=='mission-v1-degraded' and meta['warnings']


def test_boolean_cannot_be_a_source_frame_index():
    from pydantic import ValidationError
    value=candidate();value['onset_frame']=True
    with pytest.raises(ValidationError):a.Candidate.model_validate(value)


def test_different_model_profiles_cannot_share_recipe_provenance(monkeypatch):
    monkeypatch.setenv('AUDITOR_PROFILE','qwen3-8b');first=p.recipe_fingerprint()
    monkeypatch.setenv('AUDITOR_PROFILE','qwen25-7b');second=p.recipe_fingerprint()
    monkeypatch.setenv('AUDITOR_PROFILE','qwen3-32b-awq');third=p.recipe_fingerprint()
    assert len({first,second,third})==3
    monkeypatch.setenv('AUDITOR_PROFILE','unverified-model')
    with pytest.raises(ValueError):p.recipe_fingerprint()


def test_frame_only_mode_preserves_source_images_and_does_not_claim_absence(monkeypatch):
    from service.pipeline import frames as decoder,detector
    image=np.arange(80*60*3,dtype=np.uint8).reshape(80,60,3)
    source=[dict(timestamp_sec=1.25,source_frame_index=37,image=image)]
    monkeypatch.setenv('PERCEPTION_MODE','frames_only')
    monkeypatch.setattr(decoder,'iter_frames',lambda *args:iter(source))
    monkeypatch.setattr(detector,'get_detector',lambda:(_ for _ in ()).throw(AssertionError('Unused detector must not run')))
    observations,samples=p.perception('anonymous.mov',time.monotonic()+20)
    assert observations['frames']==[]
    assert samples==[dict(timestamp_sec=1.25,source_frame_index=37,image=p.encode_jpeg(image))]


def test_frame_only_cannot_silently_disable_an_enabled_object_rule(monkeypatch,tmp_path):
    monkeypatch.setenv('PERCEPTION_MODE','frames_only');monkeypatch.setattr(p.config,'ROOT',tmp_path)
    (tmp_path/'service').mkdir();(tmp_path/'service/rules.json').write_text(json.dumps(dict(types=dict(cash_left_out=dict(enabled=True)))))
    with pytest.raises(ValueError,match='enabled object rule'):p.perception('anonymous.mov',time.monotonic()+20)


def test_temporal_windows_cover_every_source_sample_with_overlap():
    source=frames(110);windows=list(a.temporal_windows(source))
    assert {f['source_frame_index'] for w in windows for f in w}=={f['source_frame_index'] for f in source}
    assert all(w[-1]['timestamp_sec']-w[0]['timestamp_sec']<=12 for w in windows)
    assert windows[0][-1]['timestamp_sec']>=windows[1][0]['timestamp_sec']


def test_tiled_late_failure_preserves_confirmed_earlier_evidence(monkeypatch):
    count=[0]
    def check(window,*args,**kwargs):
        assert kwargs['overview_count']==8 and kwargs['allow_omissions'] is False
        count[0]+=1
        if count[0]==2:raise TimeoutError
        return ([dict(type_id='food_drink',timecode_sec=1)] if count[0]==1 else []),[]
    monkeypatch.setattr(a,'confirmed_observations',check)
    observed,log=a.tiled_observations(frames(60),time.monotonic()+120)
    assert observed==[dict(type_id='food_drink',timecode_sec=1)] and any(e['stage']=='audit_incomplete' for e in log)


def test_incomplete_negative_temporal_scan_cannot_claim_primary_clean(monkeypatch):
    monkeypatch.setattr(a,'confirmed_observations',lambda *args,**kw:(_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError):a.tiled_observations(frames(60),time.monotonic()+120)


def test_short_subwindow_cannot_prove_a_transaction_omission():
    def request(*args):return reply(dict(visibility='clear',observations=[candidate('no_id_check')]))
    assert a.confirmed_observations(frames(12),time.monotonic()+60,requester=request,allow_omissions=False)[0]==[]


def test_tiny_detail_second_confirmation_checks_its_own_frame_bounds():
    count=[0]
    def request(url,payload,timeout):
        count[0]+=1
        if count[0]==1:return reply(dict(visibility='clear',observations=[candidate('keys_left_in_safe')]))
        return reply(dict(support='clear',onset_frame=0 if count[0]==2 else 15,evidence_frame=1,complete_phase_visible=True,visible_fact='Видимый ключ в замке.'))
    assert a.confirmed_observations(frames(),time.monotonic()+60,requester=request)[0]==[] and count[0]==3


def test_workflow_context_preserves_unknown_cash_and_client_role_boundary(monkeypatch):
    assert a.predicate_catalogue()==a.TYPE_PREDICATES
    monkeypatch.setenv('AUDITOR_WORKFLOW','storage_context');catalogue=a.predicate_catalogue()
    assert 'client departure alone is NOT cashier departure' in catalogue
    assert 'whose cash contents were actually visible earlier' in catalogue
    assert 'even if the cashier remains seated' in catalogue
    assert catalogue.split('safe_left_open:',1)[1]==a.TYPE_PREDICATES.split('safe_left_open:',1)[1]


def test_proposal_policy_preserves_actual_overview_time_and_skips_confirmation(monkeypatch):
    monkeypatch.setenv('AUDITOR_CONFIRMATION','proposal');calls=[]
    def request(*args):
        calls.append(1)
        return reply(dict(visibility='clear',observations=[candidate()]))
    samples=frames();observed,log=a.confirmed_observations(samples,time.monotonic()+60,requester=request)
    assert len(calls)==1 and len(observed)==1
    overview=a.choose(samples)
    assert observed[0]['timecode_sec']==overview[6]['timestamp_sec']
    assert observed[0]['evidence_sec']==overview[8]['timestamp_sec']
    assert observed[0]['origin']=='single_pass_self_hosted_observation'


@pytest.mark.parametrize('kind',sorted(a.TINY_OR_AMBIGUOUS|a.OMISSIONS))
def test_proposal_policy_cannot_bypass_detail_or_omission_safeguards(monkeypatch,kind):
    monkeypatch.setenv('AUDITOR_CONFIRMATION','proposal');calls=[]
    def request(*args):
        calls.append(1)
        if len(calls)==1:return reply(dict(visibility='clear',observations=[candidate(kind)]))
        return reply(dict(support='uncertain',onset_frame=0,evidence_frame=1,complete_phase_visible=False,visible_fact='Деталь неразличима.'))
    observed,_=a.confirmed_observations(frames(),time.monotonic()+60,requester=request)
    assert observed==[]
    assert len(calls)==(1 if kind in a.OMISSIONS else 2)


def test_late_confirmation_timeout_does_not_erase_earlier_supported_event():
    calls=[]
    def request(*args):
        calls.append(1)
        if len(calls)==1:return reply(dict(visibility='clear',observations=[candidate('cash_left_out'),candidate('food_drink')]))
        if len(calls)==3:raise TimeoutError('Later candidate timed out')
        return reply(dict(support='clear',onset_frame=1,evidence_frame=2,complete_phase_visible=False,visible_fact='Банкноты остаются на столе после ухода.'))
    observed,log=a.confirmed_observations(frames(),time.monotonic()+120,requester=request)
    assert len(observed)==1 and observed[0]['type_id']=='cash_left_out'
    assert any(e['stage']=='audit_incomplete' and e['kind']=='food_drink' for e in log)


def test_onset_refinement_sees_full_timeline_before_dense_interval(monkeypatch):
    monkeypatch.setenv('AUDITOR_CONFIRMATION','proposal_refine');calls=[]
    def request(url,payload,timeout):
        body=json.loads(payload);calls.append(body)
        if len(calls)==1:return reply(dict(visibility='clear',observations=[candidate('door_left_open')]))
        return reply(dict(located=True,onset_frame=12 if len(calls)==2 else 1,evidence_frame=13 if len(calls)==2 else 2,visible_transition='Сотрудник уходит, дверь остаётся открытой.'))
    source=frames(100);observed,log=a.confirmed_observations(source,time.monotonic()+180,requester=request)
    assert len(observed)==1 and len(calls)==3
    assert calls[0]['messages'][1]['content'][1:]==calls[1]['messages'][1]['content'][1:]
    assert log[1]['stage']=='global_onset' and log[2]['stage']=='dense_onset'
    assert observed[0]['timecode_sec'] in log[2]['source_times']
    assert observed[0]['timecode_sec']>source[50]['timestamp_sec']


@pytest.mark.parametrize('outcome',['uncertain','timeout','out_of_bounds'])
def test_unusable_timing_refinement_preserves_proposal_without_inventing_a_time(monkeypatch,outcome):
    monkeypatch.setenv('AUDITOR_CONFIRMATION','proposal_refine');calls=[]
    def request(*args):
        calls.append(1)
        if len(calls)==1:return reply(dict(visibility='clear',observations=[candidate()]))
        if outcome=='timeout':raise TimeoutError
        return reply(dict(located=outcome!='uncertain',onset_frame=15,evidence_frame=15,visible_transition='Момент перехода не установлен.'))
    source=frames(12)
    proposed,_=a.confirmed_observations(source,time.monotonic()+120,requester=request)
    assert len(proposed)==1 and proposed[0]['timecode_sec']==a.choose(source)[6]['timestamp_sec']


def test_wide_input_can_observe_and_recheck_a_complete_longer_transaction(monkeypatch):
    monkeypatch.setenv('AUDITOR_FRAMES','64');calls=[]
    def request(url,payload,timeout):
        body=json.loads(payload);calls.append(body)
        if len(calls)==1:
            c=candidate('no_receipt');c.update(onset_frame=45,evidence_frame=47)
            return reply(dict(visibility='clear',observations=[c]))
        return reply(dict(support='clear',onset_frame=45,evidence_frame=47,complete_phase_visible=True,visible_fact='Завершённая видимая передача наличных, документ не передан.'))
    source=frames(52);observed,log=a.confirmed_observations(source,time.monotonic()+120,requester=request)
    assert len(observed)==1 and observed[0]['timecode_sec']==22.5
    assert len(calls)==2 and log[0]['source_times']==log[1]['source_times']==[f['timestamp_sec'] for f in source]


@pytest.mark.parametrize('too_many,gapped',[(True,False),(False,True)])
def test_wide_input_still_rejects_omissions_when_coverage_is_incomplete(monkeypatch,too_many,gapped):
    monkeypatch.setenv('AUDITOR_FRAMES','64');calls=[]
    def request(*args):calls.append(1);return reply(dict(visibility='clear',observations=[candidate('no_id_check')]))
    source=frames(66 if too_many else 40)
    if gapped:
        for f in source:f['timestamp_sec']*=2
    assert a.confirmed_observations(source,time.monotonic()+120,requester=request)[0]==[] and len(calls)==1
