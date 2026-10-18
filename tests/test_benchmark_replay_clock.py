"""Replay must make the same deadline decisions as elapsed live inference."""
import hashlib,json,sys,time,types
from importlib.machinery import ModuleSpec
import pytest
from scripts import benchmark_auditor as bench
from service.pipeline import analyze,auditor,vlm
from src import video_analysis


class Clock:
    now=1000.
    def monotonic(self):return self.now


def setup_run(tmp_path,monkeypatch,elapsed=None):
    clock=Clock();monkeypatch.setattr(bench,'time',clock)
    payload=json.dumps({'response_format':{'json_schema':{'schema':{'type':'object'}}}}).encode()
    response={'model':'qwen-vl','choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
    if elapsed is not None:
        target=tmp_path/'cache'/'runtime'/(hashlib.sha256(payload).hexdigest()+'.json')
        target.parent.mkdir(parents=True);target.write_text(json.dumps({'elapsed_sec':elapsed,'response':response}))
    def run():return bench.run_one(tmp_path/'unused.mov','synthetic',tmp_path,tmp_path/'cache','runtime',use_cache=True)
    return clock,payload,response,run


def test_cached_sweep_advances_lazy_spatial_admission(tmp_path,monkeypatch):
    clock,payload,response,run=setup_run(tmp_path,monkeypatch,145)
    from service.pipeline.spatial import crop_pair
    spatial=types.ModuleType('service.pipeline.spatial');spatial.__spec__=ModuleSpec(spatial.__name__,loader=None);spatial.time=time
    # Lazy desk import needs the real crop helper; do not poison its retained binding.
    spatial.crop_pair=crop_pair
    monkeypatch.setitem(sys.modules,spatial.__name__,spatial)
    def infer(path,deadline,requester):
        requester('http://127.0.0.1',payload,150)
        audit_deadline=deadline-20
        assert audit_deadline-spatial.time.monotonic()==20
        assert spatial.time.monotonic()+35>=audit_deadline
        for module in (analyze,auditor,vlm,video_analysis):assert module.time.monotonic()==1145
        assert sys.modules['service.pipeline.desk'].time.monotonic()==1145
        return {'verdict':'no_violation','violations':[]},{'spatial_skipped':True}
    monkeypatch.setattr(analyze,'analyze_detailed',infer)
    result=run()
    assert result['status']=='done' and result['analysis']['spatial_skipped']
    assert result['budget_estimate_sec']==145 and result['elapsed_sec']==0
    assert spatial.time is time
    assert sys.modules['service.pipeline.desk'].time is time
    assert all(module.time is time for module in (analyze,auditor,vlm,video_analysis))


def test_cached_call_respects_smaller_call_timeout(tmp_path,monkeypatch):
    clock,payload,response,run=setup_run(tmp_path,monkeypatch,60)
    monkeypatch.setattr(analyze,'analyze_detailed',lambda path,deadline,requester:requester('http://127.0.0.1',payload,30))
    result=run()
    assert result['status']=='failed' and result['analysis']['error_code']=='TimeoutError'
    assert result['requires_api_budget_replay'] and result['budget_estimate_sec']==30
    assert result['replayed_calls']==0
    assert all(module.time is time for module in (analyze,auditor,vlm,video_analysis))


def test_clock_and_perception_restore_after_exception(tmp_path,monkeypatch):
    clock,payload,response,run=setup_run(tmp_path,monkeypatch)
    original=analyze.perception
    def fail(*args,**kwargs):raise RuntimeError('synthetic')
    monkeypatch.setattr(analyze,'analyze_detailed',fail)
    assert run()['analysis']['error_code']=='RuntimeError'
    assert analyze.perception is original
    assert all(module.time is time for module in (analyze,auditor,vlm,video_analysis))


def test_fresh_network_elapsed_is_not_double_counted(tmp_path,monkeypatch):
    clock,payload,response,run=setup_run(tmp_path,monkeypatch)
    class Reply:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):clock.now+=12;return json.dumps(response).encode()
    class Opener:
        def open(self,*args,**kwargs):return Reply()
    monkeypatch.setattr(bench.urllib.request,'build_opener',lambda *args:Opener())
    def infer(path,deadline,requester):
        requester('http://127.0.0.1',payload,30)
        assert auditor.time.monotonic()==1012
        return {'verdict':'no_violation','violations':[]},{}
    monkeypatch.setattr(analyze,'analyze_detailed',infer)
    result=run()
    assert result['status']=='done'
    assert result['elapsed_sec']==result['budget_estimate_sec']==12
    assert result['new_model_calls']==1 and result['replayed_calls']==0
