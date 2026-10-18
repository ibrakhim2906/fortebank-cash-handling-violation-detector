"""Dynamic storage/door verification, using the actual service modules."""
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from service.pipeline import spatial as s
from service.pipeline import auditor as a
from src import config

@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv('AUDITOR_FRAMES','64');monkeypatch.setenv('AUDITOR_CONFIRMATION','proposal');monkeypatch.delenv('AUDITOR_SPATIAL',raising=False)

def source(n=50):
    image=np.zeros((200,320,3),dtype=np.uint8);image[50:150,120:220]=(0,0,255)
    ok,jpeg=cv2.imencode('.jpg',image);assert ok
    return [dict(timestamp_sec=i*.5,source_frame_index=100+i,image=jpeg.tobytes()) for i in range(n)]

def box(**updates):
    data=dict(located=True,object_kind='lockable_storage',x0=350,y0=200,x1=750,y1=800,visible_fact='Шкаф.');data.update(updates);return data

def answer(**updates):
    data=dict(support='clear',same_object_visible=True,actual_opening_visible=True,operator_departure_visible=True,
              object_access_finished_visible=False,ongoing_access=False,unsafe_interval_visible=True,normal_access_closure=False,final_state='open',
              onset_frame=4,evidence_frame=6,visible_fact='Сотрудник уходит, открытый проём шкафа остаётся виден.');data.update(updates);return data

def candidate(kind='safe_left_open'):
    return a.WideCandidate(kind=kind,onset_frame=4,evidence_frame=6,support='clear',counterexamples_excluded=True,complete_phase_visible=False,visible_fact='Возможный открытый шкаф.')

def verify(monkeypatch,*,ground=None,state=None,deadline=None,frames=None):
    calls=[]
    def ask(frames,prompt,schema,deadline,**kwargs):
        calls.append((frames,prompt));value=(ground or box()) if len(calls)==1 else (state or answer());return schema.model_validate(value),dict(request_sha256='synthetic-test',source_times=[f['timestamp_sec'] for f in frames])
    monkeypatch.setattr(s,'ask',ask)
    result,logs=s.verify_open_state(candidate(),source() if frames is None else frames,2,3,deadline or time.monotonic()+150)
    return result,logs,calls

@pytest.mark.parametrize('updates',[dict(located=False),dict(object_kind='other'),dict(x1=351),dict(y1=201),dict(x0=0,y0=0,x1=1000,y1=1000),dict(x0=900,x1=800)])
def test_ambiguous_or_invalid_grounding_never_becomes_a_verdict(monkeypatch,updates):
    result,logs,calls=verify(monkeypatch,ground=box(**updates));assert result is None and len(calls)==1
    assert logs[-1]['stage']=='audit_incomplete'

@pytest.mark.parametrize('updates',[dict(x0=-1),dict(y1=1001),dict(x0=True)])
def test_normalized_box_schema_rejects_unbounded_or_boolean_coordinates(updates):
    with pytest.raises(ValidationError):s.GroundedBox.model_validate(box(**updates))

def test_actual_crop_enlarges_dynamic_region_and_keeps_source_identity():
    original=source(1)[0];copy=dict(original);cropped=s.crop_pair(original,(375,250,688,750))
    assert original==copy and cropped['timestamp_sec']==original['timestamp_sec'] and cropped['source_frame_index']==original['source_frame_index']
    image=cv2.imdecode(np.frombuffer(cropped['image'],np.uint8),cv2.IMREAD_COLOR);original_image=cv2.imdecode(np.frombuffer(original['image'],np.uint8),cv2.IMREAD_COLOR)
    assert image.shape[0]==512 and image.shape[1]>original_image.shape[1]
    # The same red object appears larger in the detail panel than in full context.
    width=round(320*512/200);left=image[:,:width];right=image[:,width:]
    red=lambda im:int(((im[:,:,2]>150)&(im[:,:,1]<60)&(im[:,:,0]<60)).sum())
    assert red(right)>red(left)*1.3

def test_temporal_inspection_includes_actual_proposal_neighbors_and_finalframe(monkeypatch):
    monkeypatch.setenv('AUDITOR_FRAMES','16');frames=source(100);selected=s.state_samples(frames,17,18)
    assert selected[0] is frames[0] and selected[-1] is frames[-1] and len(selected)<=16
    assert {15,17,18,19}<={f['timestamp_sec'] for f in selected}
    assert [f['timestamp_sec'] for f in selected]==sorted(f['timestamp_sec'] for f in selected)

@pytest.mark.parametrize('updates',[dict(ongoing_access=True),dict(unsafe_interval_visible=False,normal_access_closure=True,final_state='closed'),dict(support='counterexample')])
def test_ordinary_closure_or_still_in_use_object_rejects_initial_positive_proposal(monkeypatch,updates):
    result,logs,calls=verify(monkeypatch,state=answer(**updates));assert result is None and len(calls)==2
    assert logs[-1]['stage']=='spatial_open_state'

@pytest.mark.parametrize('updates',[dict(support='uncertain'),dict(same_object_visible=False),dict(actual_opening_visible=False),dict(operator_departure_visible=False),dict(onset_frame=63),dict(unsafe_interval_visible=False)])
def test_unproven_opening_departure_or_frame_index_marks_scan_incomplete(monkeypatch,updates):
    result,logs,calls=verify(monkeypatch,state=answer(**updates));assert result is None
    assert logs[-1]['stage']=='audit_incomplete'

def test_confirmed_departure_returns_real_source_times_without_replacing_evidence(monkeypatch):
    frames=source();saved=[f['image'] for f in frames];result,logs,calls=verify(monkeypatch,frames=frames)
    selected=s.state_samples(frames,2,3)
    assert result['timecode_sec']==selected[4]['timestamp_sec'] and result['evidence_sec']==selected[6]['timestamp_sec']
    assert result['origin']=='spatially_confirmed_self_hosted_observation'
    assert [f['image'] for f in frames]==saved
    assert calls[0][0][0]['image']==frames[6]['image'] and calls[1][0][-1]['timestamp_sec']==frames[-1]['timestamp_sec']

def test_budget_guard_makes_no_new_model_request(monkeypatch):
    result,logs,calls=verify(monkeypatch,deadline=time.monotonic()+34);assert result is None and not calls
    assert logs[-1]['error_code']=='SpatialBudget'

def test_default_off_preserves_onepass_proposal_and_makes_no_spatial_call(monkeypatch):
    calls=[]
    def ask(frames,prompt,schema,deadline,**kwargs):
        calls.append(prompt);return schema.model_validate(dict(visibility='clear',observations=[candidate().model_dump()])),dict(request_sha256='synthetic')
    monkeypatch.setattr(a,'ask',ask);result,logs=a.confirmed_observations(source(),time.monotonic()+150)
    assert len(result)==1 and len(calls)==1 and result[0]['origin']=='single_pass_self_hosted_observation'

def test_optin_gates_only_open_state_types_and_preserves_other_proposals(monkeypatch):
    monkeypatch.setenv('AUDITOR_SPATIAL','open_state')
    kinds=[]
    def check(candidate,*args,**kwargs):kinds.append(candidate.kind);return None,[dict(stage='spatial_open_state',kind=candidate.kind)]
    monkeypatch.setattr(s,'verify_open_state',check)
    def ask(frames,prompt,schema,deadline,**kwargs):
        return schema.model_validate(dict(visibility='clear',observations=[candidate().model_dump(),candidate('food_drink').model_dump()])),dict(request_sha256='synthetic')
    monkeypatch.setattr(a,'ask',ask);result,logs=a.confirmed_observations(source(),time.monotonic()+150)
    assert kinds==['safe_left_open'] and [r['type_id'] for r in result]==['food_drink']


def test_public_evidence_contains_original_full_source_not_the_inspection_crop(monkeypatch):
    import base64,json
    from service.pipeline.analyze import build_result
    frames=source();event,logs,calls=verify(monkeypatch,frames=frames)
    rules=json.loads((config.ROOT/'service/rules.json').read_text())
    result=build_result([event],frames,rules)
    original={f['timestamp_sec']:f['image'] for f in frames}
    assert result['verdict']=='violation'
    for evidence in result['violations'][0]['evidence_frames']:
        assert base64.b64decode(evidence['image_base64'])==original[evidence['timestamp_sec']]


@pytest.mark.parametrize('normal_closure',[False,True])
def test_real_unsafe_interval_is_retained_even_if_object_is_closed_later(monkeypatch,normal_closure):
    result,logs,calls=verify(monkeypatch,state=answer(final_state='closed',unsafe_interval_visible=True,normal_access_closure=normal_closure))
    assert result is not None and result['type_id']=='safe_left_open'
    assert logs[-1]['answer']['final_state']=='closed' and logs[-1]['answer']['unsafe_interval_visible']
    assert len(calls)==2

def test_normal_attended_open_close_without_unsafe_interval_is_rejected(monkeypatch):
    result,logs,calls=verify(monkeypatch,state=answer(final_state='closed',unsafe_interval_visible=False,normal_access_closure=True,operator_departure_visible=False))
    assert result is None and len(calls)==2
    assert not any(entry['stage']=='audit_incomplete' for entry in logs)


@pytest.mark.parametrize('final_state',['uncertain','not_visible'])
def test_clear_past_unsafe_interval_survives_later_occlusion(monkeypatch,final_state):
    result,logs,calls=verify(monkeypatch,state=answer(final_state=final_state,unsafe_interval_visible=True))
    assert result is not None and result['type_id']=='safe_left_open'
    assert logs[-1]['answer']['final_state']==final_state
    assert not any(entry['stage']=='audit_incomplete' for entry in logs)


def detailed_source():
    image=np.full((1200,2000,3),128,dtype=np.uint8)
    for x in range(900,1100,2):image[500:700,x:x+2]=255 if (x//2)%2 else 0
    return dict(timestamp_sec=3.25,source_frame_index=97,image=image)


def test_high_resolution_inspection_preserves_real_roi_detail_lost_in_base_jpeg(monkeypatch):
    from service.pipeline import analyze as pipeline
    monkeypatch.setenv('AUDITOR_SPATIAL','open_state');frame=pipeline.source_sample(detailed_source(),0)
    assert isinstance(frame['inspection_image'],bytes) and len(frame['inspection_image'])<=1024**2
    inspection=cv2.imdecode(np.frombuffer(frame['inspection_image'],np.uint8),cv2.IMREAD_COLOR)
    assert max(inspection.shape[:2])<=2048 and inspection.shape[1]>720
    box=(450,416,550,584)
    detailed=s.crop_pair(frame,box);low=s.crop_pair({key:value for key,value in frame.items() if key!='inspection_image'},box)
    detailed=cv2.imdecode(np.frombuffer(detailed['image'],np.uint8),cv2.IMREAD_COLOR)
    low=cv2.imdecode(np.frombuffer(low['image'],np.uint8),cv2.IMREAD_COLOR)
    # The two-pixel physical stripe pattern is present in the source, blurred
    # irreversibly by720-edge downsampling, and recoverable in the real ROI.
    context_width=round(2000*512/1200)
    high_detail=detailed[:,context_width:];low_detail=low[:,context_width:]
    assert float(high_detail.std())>float(low_detail.std())*1.7
    assert frame['timestamp_sec']==3.25 and frame['source_frame_index']==97


def test_default_off_retains_no_inspection_and_coarse_payload_is_identical(monkeypatch):
    import json
    from service.pipeline import analyze as pipeline,vlm
    raw=detailed_source();monkeypatch.setenv('AUDITOR_SPATIAL','off')
    plain=pipeline.source_sample(raw,0)
    assert set(plain)=={'timestamp_sec','source_frame_index','image'}
    assert plain['image']==pipeline.encode_jpeg(raw['image'])
    monkeypatch.setenv('AUDITOR_SPATIAL','open_state');inspected=pipeline.source_sample(raw,0)
    assert inspected['image']==plain['image']
    payloads=[]
    def request(url,payload,timeout):
        payloads.append(payload)
        return dict(model='qwen-vl',choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(visibility='clear',observations=[]))))])
    for frame in [plain,inspected]:vlm.ask([frame],'Fixed probe',a.WideSweep,time.monotonic()+30,requester=request)
    assert len(payloads)==2 and payloads[0]==payloads[1]
    assert 'inspection_image' not in payloads[0].decode()


def test_perception_only_retains_extra_jpeg_for_explicit_spatial_mode(monkeypatch):
    from service.pipeline import analyze as pipeline,frames as decoder
    monkeypatch.setenv('PERCEPTION_MODE','frames_only');raw=detailed_source()
    monkeypatch.setattr(decoder,'iter_frames',lambda *args:iter([raw]))
    monkeypatch.setenv('AUDITOR_SPATIAL','off');_,plain=pipeline.perception('anonymous.mov',time.monotonic()+30)
    assert 'inspection_image' not in plain[0]
    monkeypatch.setenv('AUDITOR_SPATIAL','open_state');_,inspected=pipeline.perception('anonymous.mov',time.monotonic()+30)
    assert isinstance(inspected[0]['inspection_image'],bytes)
    assert all(not isinstance(value,np.ndarray) for value in inspected[0].values())
    assert {key:value for key,value in inspected[0].items() if key!='inspection_image'}==plain[0]


def test_inspection_retention_has_explicit_sample_bound_without_changing_default(monkeypatch):
    from service.pipeline import analyze as pipeline
    frame=dict(timestamp_sec=300,source_frame_index=9000,image=np.zeros((16,16,3),np.uint8))
    monkeypatch.setenv('AUDITOR_SPATIAL','open_state')
    with pytest.raises(ValueError,match='sample budget'):pipeline.source_sample(frame,602)
    monkeypatch.setenv('AUDITOR_SPATIAL','off')
    assert 'inspection_image' not in pipeline.source_sample(frame,602)


def test_large_inspection_jpeg_is_bounded_and_preserves_source_aspect(monkeypatch):
    from service.pipeline import analyze as pipeline
    image=np.full((2400,4000,3),128,dtype=np.uint8);jpeg=pipeline.encode_inspection_jpeg(image)
    decoded=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
    assert len(jpeg)<=1024**2 and max(decoded.shape[:2])<=2048
    assert abs(decoded.shape[1]/decoded.shape[0]-4000/2400)<.002
