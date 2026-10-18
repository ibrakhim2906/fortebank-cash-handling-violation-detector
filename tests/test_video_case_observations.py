import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import check_case_observations as diagnostic
from src import config, video_case_observations as observer


def states():
    return {'cabinet_state':'closed','banknotes_visible':'no',
            'banknotes_on_working_table':'no','person_state':'absent',
            'coat_or_handbag_on_surface':'no','visibility':'clear'}


def response():
    return {'model':observer.MODEL_ALIAS,'choices':[{'finish_reason':'stop',
             'message':{'role':'assistant','content':json.dumps(states())}}],
            'usage':{'prompt_tokens':700,'completion_tokens':70,'total_tokens':770},
            'timings':{'cache_n':0}}


def images():
    frame=np.zeros((848,464,3),np.uint8)
    frame[300:450,100:300]=(200,100,0)
    return observer.image_views(frame)


def protocol():
    return {'groups':[{'group_id':f'g{i}','video_ids':[name],'path':f'p{i}',
                       'sha256':str(i)*64,'duration_sec':10.,'target':i%2,
                       'timestamp_annotations':[{'timestamp_sec':9.}]} for i,name in enumerate(diagnostic.NAMES)]}


def test_views_preserve_native_aspect_and_workspace_not_face_crop():
    decoded=[cv2.imdecode(np.frombuffer(v,np.uint8),cv2.IMREAD_COLOR) for v in images()]
    assert decoded[0].shape==(848,464,3)
    assert decoded[1].shape==(round(848*.72)-round(848*.18),464,3)
    assert len(images())==2


def test_request_has_fixed_prompt_cache_policy_and_no_case_context():
    value=observer.request_body(images())
    assert value['seed']==config.SEED and value['temperature']==0
    assert value['cache_prompt'] is False and value['stream'] is False
    assert value['messages'][0]['content'][0]['text']==observer.PROMPT
    assert value['response_format']['json_schema']['schema']==observer.SCHEMA
    assert len(value['messages'])==1 and len(value['messages'][0]['content'])==3
    assert 'filename' not in json.dumps(value) and 'official_target' not in json.dumps(value)


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(model='another-model'),
    lambda r:r['choices'][0].update(finish_reason='length'),
    lambda r:r.update(truncated=True),
    lambda r:r['usage'].update(prompt_tokens=4096,total_tokens=4166),
    lambda r:r['usage'].update(total_tokens=1),
    lambda r:r['choices'][0]['message'].update(content=json.dumps(states()|{'cabinet_state':'unlocked'})),
    lambda r:r['choices'][0]['message'].update(content=json.dumps(states()|{'identity':'employee'})),
    lambda r:r['choices'][0]['message'].update(content='{"cabinet_state":"closed","cabinet_state":"open"}'),
])
def test_invalid_completions_are_rejected_without_model_text(mutate):
    value=response();mutate(value)
    with pytest.raises(ValueError,match='model, completion, context or state'):
        observer.validate_response(value)


def test_observe_uses_bounded_fixed_loopback_transport_contract():
    calls=[]
    def fake(path,body,*,timeout):
        calls.append((path,body,timeout));return response()
    result=observer.observe(images(),timeout=12,requester=fake)
    assert result['states']==states()
    assert calls[0][0]=='/v1/chat/completions' and calls[0][2]==12
    assert result['body_sha256']==observer.fingerprint(calls[0][1])


@pytest.mark.parametrize('timeout',[True,0,46,float('nan')])
def test_invalid_timeout_never_calls_requester(timeout):
    with pytest.raises(ValueError):
        observer.observe([],timeout=timeout,requester=lambda *a,**k:pytest.fail('network used'))


def test_corrupt_or_extra_images_fail_closed():
    with pytest.raises(ValueError):observer.request_body([b'\xff\xd8bad',b'\xff\xd8bad'])
    with pytest.raises(ValueError):observer.request_body(images()+images())


def test_selection_is_independent_of_labels_anchors_and_input_order():
    original=protocol();expected=diagnostic.select(original)
    changed=copy.deepcopy(original);changed['groups'].reverse()
    for group in changed['groups']:
        group['target']=999;group['timestamp_annotations']=[]
    assert diagnostic.select(changed)==expected
    assert all('target' not in x and 'timestamp_annotations' not in x for x in expected)
    changed['groups'].append(copy.deepcopy(changed['groups'][0]))
    with pytest.raises(ValueError):diagnostic.select(changed)


def test_prior_comparison_keeps_unknown_and_not_accuracy():
    rows=[{'assistant_prior_states':{'cabinet_state':'open','person_state':'absent','banknotes_visible':'yes'},
           'states':states()|{'banknotes_visible':'unknown'}}]
    metrics=diagnostic.comparison(rows)
    assert metrics['attributed_prior_checks']==3
    assert metrics['prior_state_agreements']==1 and metrics['prior_state_disagreements']==1
    assert metrics['prior_state_unknowns']==1 and metrics['semantic_accuracy'] is None


def test_private_outputs_are_exclusive_and_mode600(tmp_path):
    path=tmp_path/'record.json';diagnostic.private_write(path,{'private':True})
    assert path.stat().st_mode&0o777==0o600
    with pytest.raises(FileExistsError):diagnostic.private_write(path,{'private':False})
    assert json.loads(path.read_text())=={'private':True}


def test_prepare_freezes_sources_and_rejects_tampered_manifest_or_source(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'ROOT',tmp_path)
    monkeypatch.setattr(config,'DATA_PROC',tmp_path/'data/processed')
    source_root=config.DATA_PROC/'fortebank-20261017/open/videos';source_root.mkdir(parents=True)
    p=protocol()
    for i,group in enumerate(p['groups']):
        video=source_root/f'{i}.mov';video.write_bytes(bytes([i]))
        group['path']=str(video.relative_to(tmp_path));group['sha256']=diagnostic.digest(video)
    protocol_path=config.DATA_PROC/'protocol.json';protocol_path.write_text(json.dumps(p))
    monkeypatch.setattr(diagnostic,'PROTOCOL',protocol_path)
    for source in diagnostic.SOURCES:
        path=tmp_path/source;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
    report=tmp_path/'reports/case_video_review.md';report.parent.mkdir();report.write_text('attributed review')
    fake_model={k:k for k in ('model_id','model_revision','weights_sha256','projector_sha256','runtime_release','alias')}
    monkeypatch.setattr(diagnostic,'verify_ready',lambda:fake_model)
    monkeypatch.setattr(diagnostic,'command',lambda *a:['verified-server'])
    class Capture:
        def __init__(self,*a):self.index=-1
        def isOpened(self):return True
        def get(self,key):
            return {cv2.CAP_PROP_FPS:10.,cv2.CAP_PROP_FRAME_COUNT:100,
                    cv2.CAP_PROP_POS_MSEC:max(0,self.index)*100.}.get(key,0)
        def grab(self):self.index+=1;return True
        def retrieve(self):return True,np.zeros((848,464,3),np.uint8)
        def release(self):pass
    monkeypatch.setattr(diagnostic.cv2,'VideoCapture',Capture)
    output=config.DATA_PROC/'probe';manifest=diagnostic.prepare(output)
    assert len(manifest['samples'])==18 and (output/'manifest.json').stat().st_mode&0o777==0o600
    assert output.stat().st_mode&0o777==0o700
    diagnostic.validate_manifest(output)
    original=(output/'manifest.json').read_bytes()
    tampered=copy.deepcopy(manifest);tampered['samples'][0]['assistant_prior_states']={'cabinet_state':'open'}
    (output/'manifest.json').write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match='samples'):diagnostic.validate_manifest(output)
    (output/'manifest.json').write_bytes(original)
    (tmp_path/diagnostic.SOURCES[0]).write_text('changed')
    with pytest.raises(ValueError,match='source changed'):diagnostic.validate_manifest(output)


def test_default_cli_does_not_prepare_decode_or_call_model(monkeypatch,capsys):
    monkeypatch.setattr(diagnostic,'prepare',lambda *a:pytest.fail('prepared'))
    monkeypatch.setattr(diagnostic,'run',lambda *a:pytest.fail('inference'))
    assert diagnostic.main([])==0
    assert json.loads(capsys.readouterr().out)['will_execute'] is False


def test_failed_synthetic_stops_before_any_target_and_preserves_private_failure(tmp_path,monkeypatch,capsys):
    manifest={'model':{'alias':observer.MODEL_ALIAS},'expected_server_command':[],
              'synthetic_views':[], 'source_hashes':{}, 'total_requests_max':20,
              'samples':[{'ordinal':i,'views':[],'assistant_prior_states':{}} for i in range(18)]}
    monkeypatch.setattr(diagnostic,'validate_manifest',lambda output:manifest)
    monkeypatch.setattr(diagnostic,'digest',lambda path:'fixed-digest')
    monkeypatch.setattr(diagnostic,'verify_ready',lambda:{'alias':observer.MODEL_ALIAS})
    monkeypatch.setattr(diagnostic,'ready_server',lambda expected:nullcontext({'pid':1}))
    monkeypatch.setattr(diagnostic,'attest_server',lambda expected:{'pid':1})
    monkeypatch.setattr(diagnostic,'_images',lambda *a:[])
    calls=[]
    def fake(*a,**k):
        calls.append(1)
        return {'states':states()|{'banknotes_visible':'yes','person_state':'at_desk'},
                'body_sha256':'fixed','response':response()}
    monkeypatch.setattr(observer,'observe',fake)
    assert diagnostic.run(tmp_path)==2 and len(calls)==1
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['status']=='failed' and summary['supports_expansion'] is False
    assert summary['completed_requests']==1 and summary['failure_stage']=='synthetic'
    assert not (tmp_path/'response-02.json').exists()
    assert (tmp_path/'failure.json').stat().st_mode&0o777==0o600
    assert 'banknotes_visible' not in capsys.readouterr().out
