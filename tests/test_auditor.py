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


def test_model_outage_uses_existing_cpu_fallback(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'perception',lambda *args:({'frames':[]},frames()))
    monkeypatch.setattr(a,'confirmed_observations',lambda *args,**kw:(_ for _ in ()).throw(TimeoutError()))
    monkeypatch.setattr(p,'cpu_fallback',lambda *args:dict(verdict='no_violation',violations=[]))
    result,meta=p.analyze_detailed(tmp_path/'anonymous.mov',time.monotonic()+60)
    assert result['verdict']=='no_violation' and meta['engine']=='mission-v1-degraded' and meta['warnings']


def test_boolean_cannot_be_a_source_frame_index():
    from pydantic import ValidationError
    value=candidate();value['onset_frame']=True
    with pytest.raises(ValidationError):a.Candidate.model_validate(value)
