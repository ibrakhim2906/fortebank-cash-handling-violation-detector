"""Normal service imports; isolated synthetic fixtures only."""
import time
import numpy as np
import pytest
from service.pipeline import auditor,analyze,spatial,state_ledger
from scripts import benchmark_auditor

@pytest.fixture(autouse=True)
def modes(monkeypatch):
    monkeypatch.setenv('AUDITOR_FRAMES','16');monkeypatch.setenv('AUDITOR_CONFIRMATION','proposal');monkeypatch.setenv('AUDITOR_DESK_RESCAN','off')


def proposal(kind='safe_left_open'):
    return dict(kind=kind,onset_frame=0,evidence_frame=1,support='clear',counterexamples_excluded=True,complete_phase_visible=False,visible_fact='Synthetic visible fact.')


def frames():return [dict(timestamp_sec=float(i),source_frame_index=i,image=b'original') for i in range(3)]


def sweep(*args,**kwargs):return auditor.Sweep(visibility='clear',observations=[proposal(),proposal('food_drink')]),{}


def forbidden(*args,**kwargs):raise AssertionError('Wrong verifier selected')


def test_state_ledger_mode_dispatches_only_open_state_candidates(monkeypatch):
    monkeypatch.setenv('AUDITOR_SPATIAL','state_ledger');monkeypatch.setattr(auditor,'ask',sweep);monkeypatch.setattr(spatial,'verify_open_state',forbidden)
    seen=[]
    def verify(candidate,*args,**kwargs):seen.append(candidate.kind);return None,[dict(stage='spatial_ledger_decision',decision='counterexample')]
    monkeypatch.setattr(state_ledger,'verify_open_state',verify)
    events,logs=auditor.confirmed_observations(frames(),time.monotonic()+150)
    assert seen==['safe_left_open'] and [event['type_id'] for event in events]==['food_drink']


def test_open_state_mode_keeps_original_verifier(monkeypatch):
    monkeypatch.setenv('AUDITOR_SPATIAL','open_state');monkeypatch.setattr(auditor,'ask',sweep);monkeypatch.setattr(state_ledger,'verify_open_state',forbidden)
    seen=[]
    def verify(candidate,*args,**kwargs):seen.append(candidate.kind);return None,[]
    monkeypatch.setattr(spatial,'verify_open_state',verify)
    events,logs=auditor.confirmed_observations(frames(),time.monotonic()+150)
    assert seen==['safe_left_open'] and [event['type_id'] for event in events]==['food_drink']


def test_off_mode_calls_neither_verifier(monkeypatch):
    monkeypatch.setenv('AUDITOR_SPATIAL','off');monkeypatch.setattr(auditor,'ask',sweep)
    monkeypatch.setattr(state_ledger,'verify_open_state',forbidden);monkeypatch.setattr(spatial,'verify_open_state',forbidden)
    events,logs=auditor.confirmed_observations(frames(),time.monotonic()+150)
    assert [event['type_id'] for event in events]==['safe_left_open','food_drink']


@pytest.mark.parametrize('mode,retained',[('off',False),('open_state',True),('state_ledger',True)])
def test_new_mode_retains_source_detail_without_changing_base_evidence(monkeypatch,mode,retained):
    monkeypatch.setenv('AUDITOR_SPATIAL',mode);monkeypatch.setattr(analyze,'encode_jpeg',lambda image:b'base')
    monkeypatch.setattr(analyze,'encode_inspection_jpeg',lambda image:b'detail')
    value=analyze.source_sample(dict(timestamp_sec=2.5,source_frame_index=75,image=np.zeros((2,2,3),np.uint8)),0)
    assert value['image']==b'base' and ('inspection_image' in value)==retained
    if retained:assert value['inspection_image']==b'detail'
    assert value['timestamp_sec']==2.5 and value['source_frame_index']==75


def test_replay_clock_reaches_ledger_and_restores_it_without_changing_network_clock():
    original=state_ledger.time;real_before=time.monotonic()
    with benchmark_auditor.pipeline_replay_clock(lambda:37):
        assert real_before+37<=state_ledger.time.monotonic()<real_before+38
        assert time.monotonic()<real_before+1
    assert state_ledger.time is original


def test_ledger_reuses_existing_grounding_and_crop_helpers():
    assert state_ledger.GroundedBox is spatial.GroundedBox
    assert state_ledger.bounded_box is spatial.bounded_box
    assert state_ledger.crop_pair is spatial.crop_pair
    assert state_ledger.state_samples is spatial.state_samples
    assert 'unsafe_interval_visible' in spatial.OpenState.model_fields
