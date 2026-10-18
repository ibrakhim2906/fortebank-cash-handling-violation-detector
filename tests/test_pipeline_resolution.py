import json,time
import cv2,numpy as np,pytest
from pydantic import BaseModel,StrictBool
from service.pipeline.analyze import encode_jpeg
from service.pipeline.vlm import ask


class Reply(BaseModel):
    visible:StrictBool


def test_source_resolution_preserves_supplied_pixels_without_upsampling(monkeypatch):
    image=np.zeros((848,464,3),np.uint8)
    monkeypatch.setenv('AUDITOR_RESOLUTION','standard')
    standard=cv2.imdecode(np.frombuffer(encode_jpeg(image),np.uint8),cv2.IMREAD_COLOR)
    monkeypatch.setenv('AUDITOR_RESOLUTION','source')
    source=cv2.imdecode(np.frombuffer(encode_jpeg(image),np.uint8),cv2.IMREAD_COLOR)
    assert standard.shape[:2]==(720,394) and source.shape==image.shape
    assert cv2.imdecode(np.frombuffer(encode_jpeg(image[:40,:40]),np.uint8),cv2.IMREAD_COLOR).shape[:2]==(40,40)


def test_source_pixels_reach_the_model_without_the_old_processor_shrink(monkeypatch):
    monkeypatch.setenv('AUDITOR_RESOLUTION','source');seen=[]
    def request(url,payload,timeout):
        seen.append(json.loads(payload))
        return dict(model='qwen-vl',choices=[dict(finish_reason='stop',message=dict(content='{"visible":false}'))])
    ask([dict(timestamp_sec=0,image=b'jpeg')],'Check image.',Reply,time.monotonic()+20,requester=request)
    assert seen[0]['mm_processor_kwargs']==dict(min_pixels=3136,max_pixels=524288)
    monkeypatch.setenv('AUDITOR_RESOLUTION','standard')
    ask([dict(timestamp_sec=0,image=b'jpeg')],'Check image.',Reply,time.monotonic()+20,requester=request)
    assert 'mm_processor_kwargs' not in seen[1]


def test_unknown_resolution_fails_before_encoding_or_model_io(monkeypatch):
    monkeypatch.setenv('AUDITOR_RESOLUTION','invalid')
    with pytest.raises(ValueError):encode_jpeg(np.zeros((2,2,3),np.uint8))
