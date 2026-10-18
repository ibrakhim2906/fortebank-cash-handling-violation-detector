import json,time
import pytest
from service.pipeline import analyze as p,desk as d,auditor

@pytest.mark.parametrize('ground_fails,stage',[(True,'desk_grounding'),(False,'desk_rescan')])
def test_timeout_stage_is_recorded_without_image_payload(monkeypatch,ground_fails,stage):
    monkeypatch.setenv('AUDITOR_FRAMES','16')
    monkeypatch.setattr(d,'crop_pair',lambda frame,box:frame)
    def ask(samples,prompt,schema,deadline,**kwargs):
        if ground_fails or schema is d.DeskSweep:raise TimeoutError('Do not copy this private message')
        return d.WorkRegion(located=True,consistent_across_views=True,x0=100,y0=100,x1=800,y1=800),{'request_sha256':'ground-request'}
    monkeypatch.setattr(d,'ask',ask)
    frames=[dict(timestamp_sec=float(i),source_frame_index=i,image=b'private-image') for i in range(3)]
    events,logs=d.desk_rescan(frames,time.monotonic()+160)
    assert not events and logs[-1]['failed_stage']==stage and logs[-1]['error_code']=='TimeoutError'
    assert 'private-image' not in json.dumps(logs) and 'private message' not in json.dumps(logs)
    if not ground_fails:assert logs[0]['stage']=='desk_grounding'


@pytest.mark.parametrize('underlying',['TimeoutError','DeskRescanUnknown','DeskRegionUnknown'])
def test_fallback_retains_audit_and_underlying_cause(monkeypatch,underlying):
    result={'verdict':'no_violation','violations':[]}
    audits=[dict(stage='sweep',answer={'visibility':'clear','observations':[]}),dict(stage='desk_grounding',request_sha256='ground-request'),dict(stage='audit_incomplete',failed_stage='desk_rescan',error_code=underlying)]
    monkeypatch.setattr(p,'perception',lambda *args:({'frames':[]},[dict(timestamp_sec=0.,source_frame_index=0,image=b'private-image')]))
    monkeypatch.setattr(auditor,'confirmed_observations',lambda *args,**kwargs:([],audits))
    monkeypatch.setattr(p,'cpu_fallback',lambda *args:result)
    monkeypatch.setattr(p,'recipe_fingerprint',lambda:'synthetic-recipe')
    public,metadata=p.analyze_detailed('synthetic-no-file')
    assert public==result and set(public)=={'verdict','violations'}
    assert metadata['engine']=='mission-v1-degraded'
    assert metadata['auditor']==audits and metadata['failed_stage']=='desk_rescan'
    assert metadata['error_code']==underlying and metadata['fallback_exception_type']=='TimeoutError'
    assert 'private-image' not in json.dumps(metadata) and 'image_base64' not in json.dumps(metadata)


def test_dispatch_exception_does_not_invent_completed_stage(monkeypatch):
    monkeypatch.setattr(p,'perception',lambda *args:({'frames':[]},[]))
    def fail(*args,**kwargs):raise ValueError('private detail')
    monkeypatch.setattr(auditor,'confirmed_observations',fail)
    monkeypatch.setattr(p,'cpu_fallback',lambda *args:{'verdict':'no_violation','violations':[]})
    monkeypatch.setattr(p,'recipe_fingerprint',lambda:'synthetic')
    result,metadata=p.analyze_detailed('synthetic')
    assert metadata['auditor']==[] and metadata['failed_stage']=='auditor_dispatch'
    assert metadata['error_code']=='ValueError' and 'private detail' not in json.dumps(metadata)
