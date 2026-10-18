"""Synthetic motion and mocked temporal-action checks; no case footage."""
import time
import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from service.pipeline import action_windows as m, auditor as a

def frames(kind='motion'):
    result=[]
    for i in range(80):
        image=np.full((128,128,3),40 if kind!='brightness' or i<30 else 180,np.uint8)
        if kind=='motion':
            if 10<=i<16:image[20:60,10+(i-10)*12:35+(i-10)*12]=200
            if 55<=i<61:image[60:100,10+(i-55)*12:35+(i-55)*12]=200
        ok,jpg=cv2.imencode('.jpg',image);assert ok
        result.append(dict(timestamp_sec=i*.5,source_frame_index=i,image=jpg.tobytes()))
    return result

def action(**changes):
    return m.Action(**(dict(kind='food_drink',before_frame=1,after_frame=2,
        before_fact='Контейнер ниже лица.',after_fact='Контейнер поднесён ко рту.',support='clear',
        prerequisites_visible=True,counterexamples_excluded=True)|changes))

def answer(**changes):
    return m.ActionSweep(**(dict(visibility='clear',assessment='concern',observations=[action()])|changes))

def test_motion_selects_two_disjoint_local_intervals_preserving_originals():
    source=frames();windows=m.motion_windows(source,time.monotonic()+10)
    assert len(windows)==2
    assert windows[0][-1]['timestamp_sec']<windows[1][0]['timestamp_sec']
    for window in windows:
        assert 2<=len(window)<=16
        assert window[-1]['timestamp_sec']-window[0]['timestamp_sec']<=8
        assert all(frame is source[frame['source_frame_index']] for frame in window)
    assert windows[0][0]['timestamp_sec']<=6<=windows[0][-1]['timestamp_sec']
    assert windows[1][0]['timestamp_sec']<=29<=windows[1][-1]['timestamp_sec']

@pytest.mark.parametrize('kind',['static','brightness'])
def test_static_and_global_brightness_do_not_propose_motion(kind):
    assert m.motion_windows(frames(kind),time.monotonic()+10)==[]

def test_selection_ignores_identity_fields_and_absolute_time_origin():
    source=frames();first=m.motion_windows(source,time.monotonic()+10)
    other=[dict(f,timestamp_sec=f['timestamp_sec']+100,filename='unrelated',label='unused') for f in source]
    second=m.motion_windows(other,time.monotonic()+10)
    assert [[f['source_frame_index'] for f in w] for w in first]==[[f['source_frame_index'] for f in w] for w in second]

def test_real_window_payload_bound_and_evidence_source_mapping(monkeypatch):
    calls=[];deadline=time.monotonic()+160;source=frames()
    def ask(selected,prompt,schema,limit,**kwargs):
        calls.append(selected)
        assert limit==deadline and len(selected)<=16 and kwargs['max_tokens']==512
        assert 'hand_to_hand_transfer:' in prompt and 'no_receipt:' not in prompt
        assert 'bypassed transfer device' in prompt
        return answer(),{}
    monkeypatch.setattr(m,'ask',ask)
    events,logs=m.action_rescan(source,deadline)
    assert len(calls)==2 and len(events)==2
    for event,selected in zip(events,calls):
        assert event['timecode_sec']==selected[2]['timestamp_sec']
        assert event['evidence_sec']==selected[2]['timestamp_sec']
    assert all(frame['image']==source[frame['source_frame_index']]['image'] for selected in calls for frame in selected)

@pytest.mark.parametrize('changes',[{'prerequisites_visible':False},{'counterexamples_excluded':False},
    {'support':'uncertain'},{'before_frame':2,'after_frame':1},{'before_frame':2,'after_frame':2}])
def test_unsupported_or_reversed_transition_never_positive(monkeypatch,changes):
    monkeypatch.setattr(m,'ask',lambda *args,**kw:(answer(observations=[action(**changes)]),{}))
    events,logs=m.action_rescan(frames(),time.monotonic()+160)
    assert not events and any(row['stage']=='action_window_incomplete' for row in logs)
    assert not any(row['stage']=='audit_incomplete' for row in logs)

def test_positive_survives_later_timeout(monkeypatch):
    calls=[]
    def ask(*args,**kw):
        calls.append(1)
        if len(calls)==2:raise TimeoutError()
        return answer(),{}
    monkeypatch.setattr(m,'ask',ask)
    events,logs=m.action_rescan(frames(),time.monotonic()+160)
    assert len(events)==1 and logs[-1]['error_code']=='TimeoutError'

def test_budget_skips_without_model_call_or_mandatory_failure(monkeypatch):
    monkeypatch.setattr(m,'ask',lambda *a,**kw:pytest.fail('No call budget'))
    events,logs=m.action_rescan(frames(),time.monotonic()+20)
    assert not events and logs[-1]['stage']=='action_window_incomplete'

def test_budget_rechecked_after_first_call(monkeypatch):
    now=[0.];calls=[]
    monkeypatch.setattr(m,'time',type('Clock',(),{'monotonic':staticmethod(lambda:now[0])}))
    def ask(*args,**kw):
        calls.append(1);now[0]=140.
        return answer(assessment='clear_negative',observations=[]),{}
    monkeypatch.setattr(m,'ask',ask)
    events,logs=m.action_rescan(frames(),165.)
    assert len(calls)==1 and not events and logs[-1]['error_code']=='ActionWindowBudget'

def test_schema_excludes_omissions_and_context_heavy_labels():
    for kind in ['no_receipt','cash_left_out','counting_without_client','safe_left_open']:
        with pytest.raises(ValidationError):action(kind=kind)

@pytest.mark.parametrize('enabled,positive,expected',[('off',False,0),('on',True,0),('on',False,1)])
def test_integration_only_runs_after_no_accepted_concern(monkeypatch,enabled,positive,expected):
    monkeypatch.setenv('AUDITOR_ACTION_WINDOWS',enabled)
    monkeypatch.setenv('AUDITOR_DESK_RESCAN','off');monkeypatch.setenv('AUDITOR_SPATIAL','off')
    monkeypatch.setenv('AUDITOR_CONFIRMATION','proposal');monkeypatch.setenv('AUDITOR_FRAMES','16')
    candidate=a.Candidate(kind='food_drink',onset_frame=1,evidence_frame=2,support='clear',counterexamples_excluded=True,complete_phase_visible=False,visible_fact='Факт.')
    monkeypatch.setattr(a,'ask',lambda *args,**kw:(a.Sweep(visibility='clear',observations=[candidate] if positive else []),{}))
    calls=[]
    def rescan(*args,**kw):calls.append(1);return [],[]
    monkeypatch.setattr(m,'action_rescan',rescan)
    result,_=a.confirmed_observations(frames(),time.monotonic()+160)
    assert len(calls)==expected
    if positive:assert result[0]['origin']=='single_pass_self_hosted_observation'

def test_original_coarse_arguments_identical_between_flag_values(monkeypatch):
    monkeypatch.setenv('AUDITOR_DESK_RESCAN','off');monkeypatch.setenv('AUDITOR_SPATIAL','off')
    monkeypatch.setenv('AUDITOR_FRAMES','16');monkeypatch.setenv('AUDITOR_WORKFLOW','original')
    calls=[];source=frames();deadline=time.monotonic()+160
    def ask(*args,**kwargs):calls.append((args,kwargs));return a.Sweep(visibility='clear',observations=[]),{}
    monkeypatch.setattr(a,'ask',ask);monkeypatch.setattr(m,'action_rescan',lambda *a,**kw:([],[]))
    for value in ['off','on']:
        monkeypatch.setenv('AUDITOR_ACTION_WINDOWS',value);a.confirmed_observations(source,deadline)
    assert calls[0]==calls[1]
