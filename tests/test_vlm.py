import json,time
import pytest
from pydantic import BaseModel,ConfigDict,StrictBool
from service.pipeline import vlm


class Reply(BaseModel):
    model_config=ConfigDict(extra='forbid')
    visible:StrictBool


@pytest.mark.parametrize('url',['https://example.com','http://localhost:8001','http://127.0.0.1@evil.test','http://127.0.0.1:8001/path','http://127.0.0.1:8001?x=1'])
def test_hosted_or_ambiguous_endpoint_rejected(monkeypatch,url):
    monkeypatch.setenv('VLLM_URL',url)
    with pytest.raises(ValueError):vlm.endpoint()


def test_request_never_contains_file_identity_and_enforces_json():
    calls=[]
    def request(url,payload,timeout):
        calls.append(json.loads(payload));assert url=='http://127.0.0.1:8001/v1/chat/completions';assert timeout<=10
        return dict(model='qwen-vl',choices=[dict(finish_reason='stop',message=dict(content='{"visible":false}'))])
    result,meta=vlm.ask([dict(timestamp_sec=0,image=b'jpeg')],'Visible person?',Reply,time.monotonic()+20,requester=request)
    assert result.visible is False and len(meta['request_sha256'])==64
    assert 'filename' not in json.dumps(calls) and calls[0]['seed']==42


def test_invalid_schema_retried_once_then_rejected():
    calls=[]
    def request(*args):
        calls.append(1);return dict(model='qwen-vl',choices=[dict(finish_reason='stop',message=dict(content='{"visible":"yes"}'))])
    with pytest.raises(ValueError):vlm.ask([dict(timestamp_sec=0,image=b'jpeg')],'Visible person?',Reply,time.monotonic()+20,requester=request)
    assert len(calls)==2


def test_network_timeout_is_not_repeated_until_job_exhaustion():
    calls=[]
    def request(*args):calls.append(1);raise TimeoutError
    with pytest.raises(TimeoutError):vlm.ask([dict(timestamp_sec=0,image=b'jpeg')],'?',Reply,time.monotonic()+20,requester=request)
    assert len(calls)==1


@pytest.mark.parametrize('configured',[60,90])
def test_larger_model_call_still_respects_remaining_video_budget(monkeypatch,configured):
    monkeypatch.setenv('AUDITOR_CALL_TIMEOUT',str(configured));limits=[]
    def request(url,payload,timeout):
        limits.append(timeout)
        return dict(model='qwen-vl',choices=[dict(finish_reason='stop',message=dict(content='{"visible":false}'))])
    frame=[dict(timestamp_sec=0,image=b'jpeg')]
    vlm.ask(frame,'?',Reply,time.monotonic()+200,requester=request)
    vlm.ask(frame,'?',Reply,time.monotonic()+5,requester=request)
    assert limits[0]==configured and 0<limits[1]<=4.5


@pytest.mark.parametrize('value',['91','0','nan','inf'])
def test_invalid_larger_model_timeout_never_reaches_network(monkeypatch,value):
    monkeypatch.setenv('AUDITOR_CALL_TIMEOUT',value)
    def forbidden(*args):raise AssertionError('Invalid runtime configuration must not send a request')
    with pytest.raises(ValueError):vlm.ask([dict(timestamp_sec=0,image=b'jpeg')],'?',Reply,time.monotonic()+200,requester=forbidden)
