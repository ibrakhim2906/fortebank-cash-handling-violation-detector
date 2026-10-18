"""Storage-only negative authority; reuse existing ledger request path."""
import time
import pytest
from service.pipeline import state_ledger as s,auditor as a

def candidate(kind='safe_left_open'):
    return a.Candidate(kind=kind,onset_frame=1,evidence_frame=2,support='clear',counterexamples_excluded=True,complete_phase_visible=False,visible_fact='Исходное наблюдение.')

@pytest.mark.parametrize('event,details,veto',[
    (None,[dict(stage='spatial_ledger_decision',decision='counterexample')],True),
    (None,[dict(stage='spatial_ledger_decision',decision='unknown'),dict(stage='audit_incomplete',error_code='SpatialLedgerUnresolved')],False),
    ({'type_id':'safe_left_open'},[dict(stage='spatial_ledger_decision',decision='event')],False),
    (None,[dict(stage='audit_incomplete',error_code='SpatialObjectUnknown')],False),
    (None,[],False),
    (None,[dict(stage='spatial_ledger_decision',decision='counterexample'),dict(stage='audit_incomplete',error_code='BudgetLimit')],False),
])
def test_only_explicit_complete_counterexample_has_veto_authority(monkeypatch,event,details,veto):
    calls=[];frames=[{'timestamp_sec':0}];deadline=123.
    def verify(*args,**kwargs):calls.append((args,kwargs));return event,details
    monkeypatch.setattr(s,'verify_open_state',verify)
    proposed=candidate();requester=object()
    actual,logs=s.storage_counterexample(proposed,frames,1.,2.,deadline,requester=requester)
    assert actual is veto
    assert calls==[((proposed,frames,1.,2.,deadline),{'requester':requester})]
    assert not any(row['stage']=='audit_incomplete' for row in logs)

def test_timeout_has_no_veto_authority(monkeypatch):
    def timeout(*args,**kwargs):raise TimeoutError()
    monkeypatch.setattr(s,'verify_open_state',timeout)
    veto,logs=s.storage_counterexample(candidate(),[],0,0,0)
    assert not veto and logs[0]['stage']=='storage_veto_incomplete'

def test_nonstorage_never_calls_ledger(monkeypatch):
    monkeypatch.setattr(s,'verify_open_state',lambda *args,**kwargs:pytest.fail('No door veto'))
    assert s.storage_counterexample(candidate('door_left_open'),[],0,0,0)==(False,[])

@pytest.mark.parametrize('veto',[False,True])
def test_auditor_preserves_other_events_and_original_storage_payload(monkeypatch,veto):
    monkeypatch.setenv('AUDITOR_SPATIAL','storage_veto');monkeypatch.setenv('AUDITOR_CONFIRMATION','proposal')
    monkeypatch.setenv('AUDITOR_DESK_RESCAN','off');monkeypatch.setenv('AUDITOR_ACTION_WINDOWS','off')
    monkeypatch.setenv('AUDITOR_FRAMES','16')
    kinds=['safe_left_open','door_left_open','food_drink'];source=[dict(timestamp_sec=float(i),image=b'not decoded') for i in range(8)]
    monkeypatch.setattr(a,'ask',lambda *args,**kwargs:(a.Sweep(visibility='clear',observations=[candidate(k) for k in kinds]),{}))
    seen=[]
    def check(proposed,*args,**kwargs):seen.append(proposed.kind);return veto,[dict(stage='storage_veto_decision',veto=veto)]
    monkeypatch.setattr(s,'storage_counterexample',check)
    result,logs=a.confirmed_observations(source,time.monotonic()+160)
    assert seen==['safe_left_open']
    assert [event['type_id'] for event in result]==(kinds[1:] if veto else kinds)
    for event in result:
        assert event['timecode_sec']==1. and event['evidence_sec']==2.
        assert event['visible_fact']=='Исходное наблюдение.'
        assert event['origin']=='single_pass_self_hosted_observation'
