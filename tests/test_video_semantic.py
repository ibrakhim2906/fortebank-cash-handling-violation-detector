import json,time
from pathlib import Path
import cv2,numpy as np,pytest
from src import video_semantic as s
from src.video_api import Result


def response(value):
    return {'model':'local-action-review','choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}], 'usage':{'prompt_tokens':100}}


def candidate(kind='cash_left_out',**updates):
    value=dict(type=kind,start_frame=0,end_frame=1,evidence_frame=1,visible_fact='На рабочей поверхности остались банкноты после выхода сотрудника.',certainty='clear',counterexamples_excluded=True,complete_phase_visible=False)
    return value|updates


def observation(candidates):
    return dict(scene_visibility='clear',context='Видны рабочее место и выход.',candidates=candidates)


def test_unknowns_and_invisible_omissions_cannot_be_promoted():
    value=observation([candidate('no_id_check',complete_phase_visible=True),candidate(certainty='uncertain'),candidate(counterexamples_excluded=False),candidate('keys_left_in_safe')])
    assert s.validate_candidates(value,[{},{}])==[]


def test_source_indices_and_episode_bounds_are_enforced():
    with pytest.raises(ValueError):s.validate_candidates(observation([candidate(evidence_frame=2)]),[{},{}])
    with pytest.raises(ValueError):s.validate_candidates(observation([candidate(start_frame=1,evidence_frame=0)]),[{},{}])
    with pytest.raises(ValueError):s.validate_candidates(observation([candidate(start_frame=True)]),[{},{}])


def test_deduplication_preserves_distinct_types_and_episodes():
    values=[dict(type='cash_left_out',start_sec=0,end_sec=3),dict(type='cash_left_out',start_sec=4,end_sec=6),dict(type='personal_items',start_sec=1,end_sec=2),dict(type='cash_left_out',start_sec=12,end_sec=14)]
    result=s.deduplicate(values)
    assert len(result)==3 and result[0]['end_sec']==6


def video(tmp_path):
    p=tmp_path/'source.avi';writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64));assert writer.isOpened()
    for i in range(30):writer.write(np.full((64,64,3),i*5,np.uint8))
    writer.release();return p


def test_full_flow_keeps_source_evidence_and_fixed_citations(tmp_path):
    path=video(tmp_path);requests=[]
    def requester(route,body,**kwargs):
        requests.append(body)
        if 'scene_visibility' in body['response_format']['json_schema']['schema']['properties']:
            return response(observation([candidate()]))
        return response(dict(supported=True,counterexample=False,visible_fact='Банкноты находятся на столе после ухода сотрудника.',evidence_frame=3,complete_phase_visible=False))
    result,meta=s.analyze(path,requester=requester,check_runtime=False)
    validated=Result.model_validate(result,context={'duration_sec':3.,'tolerance_sec':.25})
    assert validated.verdict=='violation'
    e=result['violations'][0]
    assert s.CASH_URL in e['description'] and e['timecode_sec']==e['evidence_frames'][0]['timestamp_sec']
    assert meta['source_sha256'] and meta['recipe_sha256'] and not meta['production_qualified']
    assert path.name not in json.dumps(requests)
    assert all('filename' not in json.dumps(body) for body in requests)


def test_insufficient_view_is_failure_not_clean_result(tmp_path):
    def requester(*args,**kwargs):return response(dict(scene_visibility='insufficient',context='',candidates=[]))
    with pytest.raises(ValueError,match='visibility'):s.analyze(video(tmp_path),requester=requester,check_runtime=False)


def test_confirmation_rejects_counterexample():
    assert not s.validate_confirmation(dict(supported=True,counterexample=True,visible_fact='Идёт активная работа.',evidence_frame=0,complete_phase_visible=False),[{}])


def test_expired_deadline_prevents_runtime_and_decode(tmp_path):
    with pytest.raises(TimeoutError):s.analyze(tmp_path/'absent.mov',deadline=time.monotonic()-1)


def test_dispatch_opt_in_never_unpickles_missing_artifact(monkeypatch,tmp_path):
    from src import video_inference
    monkeypatch.setenv('DATATHON_VIDEO_ENGINE','semantic-v1')
    monkeypatch.setattr(s,'analyze_video',lambda path,deadline=None:{'verdict':'no_violation','violations':[]})
    assert video_inference.analyze_video(tmp_path/'new.mov',model_path=tmp_path/'absent.joblib')['verdict']=='no_violation'
