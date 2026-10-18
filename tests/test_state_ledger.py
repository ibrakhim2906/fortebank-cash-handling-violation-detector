"""Synthetic chronological states only; no bank footage or model calls."""
from types import SimpleNamespace
import time
import pytest
from service.pipeline import state_ledger as s


def ledger(*states):
    return s.StateLedger(frames=[s.FrameState(frame=i,state=state,operator=operator) for i,(state,operator) in enumerate(states)])


@pytest.mark.parametrize('kind,active',[('safe_left_open','accessing'),('door_left_open','passing')])
def test_actual_access_to_clear_open_is_onset(kind,active):
    value=ledger(('closed',active),('open',active),('open','clear'))
    assert s.ledger_interval(value,3,kind)==('event',2)


@pytest.mark.parametrize('later',[('closed','clear'),('unknown','unknown'),('open','accessing')])
def test_earlier_observed_unsafe_interval_survives_later_states(later):
    value=ledger(('open','accessing'),('open','clear'),later)
    assert s.ledger_interval(value,3,'safe_left_open')==('event',1)


def test_closed_before_departure_is_counterexample():
    value=ledger(('open','accessing'),('closed','accessing'),('closed','clear'))
    assert s.ledger_interval(value,3,'safe_left_open')==('counterexample',None)


@pytest.mark.parametrize('middle',[('unknown','accessing'),('open','unknown'),('unknown','unknown')])
def test_unknown_never_bridges_access_to_departure(middle):
    value=ledger(('open','accessing'),middle,('open','clear'))
    assert s.ledger_interval(value,3,'safe_left_open')==('unknown',None)


def test_open_while_passing_is_not_finished_passage():
    value=ledger(('open','passing'),('open','passing'))
    assert s.ledger_interval(value,2,'door_left_open')==('counterexample',None)


def test_unsupported_initial_open_state_remains_unknown():
    assert s.ledger_interval(ledger(('open','clear'),('open','clear')),2,'safe_left_open')==('unknown',None)


def test_safe_handling_does_not_establish_door_passage():
    assert s.ledger_interval(ledger(('open','accessing'),('open','clear')),2,'door_left_open')==('unknown',None)


@pytest.mark.parametrize('indices',[[0,0],[1,0],[0,2],[0]])
def test_missing_duplicate_or_reordered_states_are_unknown(indices):
    value=s.StateLedger(frames=[s.FrameState(frame=i,state='open',operator='accessing' if n==0 else 'clear') for n,i in enumerate(indices)])
    assert s.ledger_interval(value,2,'safe_left_open')==('unknown',None)


def test_two_call_adapter_preserves_source_times_and_original_frames(monkeypatch):
    monkeypatch.setenv('AUDITOR_FRAMES','16')
    frames=[dict(timestamp_sec=t,source_frame_index=i,image=b'original') for i,t in enumerate([0.,8.,21.])]
    before=[dict(frame) for frame in frames];calls=[]
    response=ledger(('open','accessing'),('open','clear'),('closed','clear'))
    def ask(samples,prompt,schema,deadline,**kwargs):
        calls.append((samples,schema,deadline,kwargs['max_tokens']))
        if schema is s.GroundedBox:
            return s.GroundedBox(located=True,object_kind='safe',x0=100,y0=100,x1=500,y1=600,visible_fact='synthetic'),{}
        assert schema is s.StateLedger
        return response,{}
    monkeypatch.setattr(s,'ask',ask)
    monkeypatch.setattr(s,'crop_pair',lambda frame,box:dict(frame,image=b'inspection'))
    deadline=time.monotonic()+160
    event,logs=s.verify_open_state(SimpleNamespace(kind='safe_left_open'),frames,8.,8.,deadline)
    assert len(calls)==2 and all(call[2]==deadline for call in calls)
    assert calls[1][3]==768
    assert event['timecode_sec']==event['evidence_sec']==8.
    assert frames==before
    assert 'support' not in s.StateLedger.model_json_schema()['properties']


def test_unknown_adapter_logs_incomplete_instead_of_clean(monkeypatch):
    monkeypatch.setenv('AUDITOR_FRAMES','16')
    frames=[dict(timestamp_sec=float(i),source_frame_index=i,image=b'original') for i in range(2)]
    def ask(samples,prompt,schema,deadline,**kwargs):
        if schema is s.GroundedBox:
            return s.GroundedBox(located=True,object_kind='door',x0=100,y0=100,x1=500,y1=600,visible_fact='synthetic'),{}
        return ledger(('open','passing'),('unknown','unknown')),{}
    monkeypatch.setattr(s,'ask',ask);monkeypatch.setattr(s,'crop_pair',lambda frame,box:frame)
    event,logs=s.verify_open_state(SimpleNamespace(kind='door_left_open'),frames,0.,0.,time.monotonic()+160)
    assert event is None and logs[-1]['stage']=='audit_incomplete'
