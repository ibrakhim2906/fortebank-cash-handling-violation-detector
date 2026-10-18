"""Grounded desk rescan: synthetic geometry and mocked inference only."""
import time,base64,json
import cv2,numpy as np,pytest
from pydantic import ValidationError
from service.pipeline import desk as d, auditor as a, analyze as p


@pytest.fixture(autouse=True)
def isolated_auditor_settings(monkeypatch):
    monkeypatch.setenv('AUDITOR_FRAMES','16')
    monkeypatch.setenv('AUDITOR_RESOLUTION','standard')
    monkeypatch.setenv('AUDITOR_SPATIAL','off')
    monkeypatch.setenv('AUDITOR_DESK_RESCAN','off')
    monkeypatch.setenv('AUDITOR_WORKFLOW','original')


def frames():
    image=np.zeros((848,1400,3),np.uint8);image[100:600,700:1050]=(20,180,240)
    return [dict(timestamp_sec=float(i),source_frame_index=i,image=p.encode_jpeg(image),inspection_image=p.encode_inspection_jpeg(image)) for i in range(20)]


def region(**updates):
    return d.WorkRegion(**(dict(located=True,consistent_across_views=True,x0=450,y0=80,x1=820,y1=850)|updates))


def candidate(**updates):
    return d.DeskCandidate(**(dict(kind='food_drink',onset_frame=2,evidence_frame=3,support='clear',counterexamples_excluded=True,visible_fact='Синтетический пример.')|updates))


def run(monkeypatch,answer,ground=None):
    calls=[];source=frames()
    def ask(samples,prompt,schema,deadline,**kwargs):
        calls.append((samples,prompt,schema,deadline))
        return (ground or region(),{}) if schema is d.WorkRegion else (answer,{})
    monkeypatch.setattr(d,'ask',ask)
    deadline=time.monotonic()+160
    result,logs=d.desk_rescan(source,deadline)
    return result,logs,calls,source,deadline


def test_real_crops_and_source_time_mapping_preserve_original_api_evidence(monkeypatch):
    answer=d.DeskSweep(visibility='clear',assessment='concern',observations=[candidate()])
    events,logs,calls,source,deadline=run(monkeypatch,answer)
    assert len(calls)==2 and [f['timestamp_sec'] for f in calls[0][0]]==[0.,9.,19.]
    assert all(call[3]==deadline for call in calls)
    assert len(calls[1][0])==16
    crop=calls[1][0][0];decoded=cv2.imdecode(np.frombuffer(crop['image'],np.uint8),cv2.IMREAD_COLOR)
    assert decoded.shape[0]==512 and decoded.shape[1]>600
    # Known synthetic marker center, transformed into the padded ROI panel.
    context_width=round(1400*512/848)
    left=int((450-(820-450)*.35)*1400/1000)
    marker_x=context_width+round((875-left)*512/848);marker_y=round(350*512/848)
    assert np.mean(decoded[marker_y-10:marker_y+10,marker_x-10:marker_x+10,2])>220
    assert crop['source_frame_index']==source[0]['source_frame_index'] and crop['image']!=source[0]['image']
    assert events[0]['timecode_sec']==calls[1][0][2]['timestamp_sec']
    rules=json.loads((p.config.ROOT/'service/rules.json').read_text())
    result=p.build_result(events,source,rules)
    originals={f['timestamp_sec']:f['image'] for f in source}
    for evidence in result['violations'][0]['evidence_frames']:
        assert base64.b64decode(evidence['image_base64'])==originals[evidence['timestamp_sec']]
    assert 'cash_left_out:' in calls[1][1] and 'no_receipt:' not in calls[1][1]


def test_explicit_clear_negative_remains_primary_eligible(monkeypatch):
    events,logs,*_=run(monkeypatch,d.DeskSweep(visibility='clear',assessment='clear_negative',observations=[]))
    assert not events and not any(row['stage']=='audit_incomplete' for row in logs)


@pytest.mark.parametrize('visibility,assessment',[('limited','clear_negative'),('insufficient','clear_negative'),('clear','uncertain'),('clear','concern')])
def test_unknown_or_contradictory_empty_answer_is_incomplete(monkeypatch,visibility,assessment):
    events,logs,*_=run(monkeypatch,d.DeskSweep(visibility=visibility,assessment=assessment,observations=[]))
    assert not events and logs[-1]['stage']=='audit_incomplete'


def test_partial_supported_evidence_survives_uncertain_sibling(monkeypatch):
    answer=d.DeskSweep(visibility='limited',assessment='uncertain',observations=[candidate(),candidate(kind='personal_items',support='uncertain')])
    events,logs,*_=run(monkeypatch,answer)
    assert [event['type_id'] for event in events]==['food_drink'] and logs[-1]['stage']=='audit_incomplete'


def test_unknown_grounding_stops_before_rescan(monkeypatch):
    events,logs,calls,*_=run(monkeypatch,d.DeskSweep(visibility='clear',assessment='clear_negative',observations=[]),region(consistent_across_views=False))
    assert not events and len(calls)==1 and logs[-1]['stage']=='audit_incomplete'


def test_budget_makes_zero_model_calls(monkeypatch):
    monkeypatch.setattr(d,'ask',lambda *args,**kwargs:pytest.fail('No remaining budget'))
    events,logs=d.desk_rescan(frames(),time.monotonic()+20)
    assert not events and logs[-1]['error_code']=='DeskRescanBudget'


@pytest.mark.parametrize('kind',['no_receipt','no_id_check','workstation_unlocked','keys_left_in_safe','safe_left_open'])
def test_schema_excludes_omissions_tiny_states_and_other_types(kind):
    with pytest.raises(ValidationError):candidate(kind=kind)


@pytest.mark.parametrize('enabled,positive,expected',[('off',False,0),('on',True,0),('on',False,1)])
def test_rescan_only_when_enabled_and_no_existing_positive(monkeypatch,enabled,positive,expected):
    monkeypatch.setenv('AUDITOR_DESK_RESCAN',enabled);monkeypatch.setenv('AUDITOR_SPATIAL','off');monkeypatch.setenv('AUDITOR_CONFIRMATION','proposal');monkeypatch.setenv('AUDITOR_FRAMES','16')
    calls=[]
    monkeypatch.setattr(d,'desk_rescan',lambda *args,**kwargs:(calls.append(True) or [],[]))
    observed=[a.Candidate(**candidate().model_dump(),complete_phase_visible=False)] if positive else []
    monkeypatch.setattr(a,'ask',lambda *args,**kwargs:(a.Sweep(visibility='clear',observations=observed),{}))
    events,logs=a.confirmed_observations(frames(),time.monotonic()+160)
    assert len(calls)==expected and bool(events)==positive


def test_auxiliary_retention_default_off_and_coarse_bytes_unchanged(monkeypatch):
    monkeypatch.setenv('AUDITOR_SPATIAL','off');monkeypatch.setenv('AUDITOR_DESK_RESCAN','off')
    source=dict(timestamp_sec=0.,source_frame_index=0,image=np.full((848,1400,3),90,np.uint8))
    plain=p.source_sample(source,0)
    monkeypatch.setenv('AUDITOR_DESK_RESCAN','on');detail=p.source_sample(source,0)
    assert 'inspection_image' not in plain and isinstance(detail['inspection_image'],bytes)
    assert detail['image']==plain['image']
