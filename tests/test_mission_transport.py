from types import SimpleNamespace
import pytest,requests
from scripts import run_mission_eval as runner


def test_transient_get_failures_retry_within_original_budget(monkeypatch):
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    closed=[];timeouts=[]
    replies=iter([SimpleNamespace(status_code=503,close=lambda:closed.append(1)),requests.Timeout(),SimpleNamespace(status_code=200)])
    def get(url,**kwargs):
        timeouts.append(kwargs['timeout']);item=next(replies)
        if isinstance(item,Exception):raise item
        return item
    retries=[0]
    result=runner.bounded_get(SimpleNamespace(get=get),'https://our-pod/jobs/id',retries,timeout=4)
    assert result.status_code==200 and retries==[2] and closed==[1]
    assert len(timeouts)==3 and all(0<t<=4 for t in timeouts) and timeouts==sorted(timeouts,reverse=True)


def test_authentication_errors_are_never_retried():
    calls=[];retries=[0]
    def get(*args,**kwargs):calls.append(1);return SimpleNamespace(status_code=401)
    assert runner.bounded_get(SimpleNamespace(get=get),'url',retries).status_code==401
    assert calls==[1] and retries==[0]


def test_exhausted_get_budget_never_starts_another_request():
    def forbidden(*args,**kwargs):raise AssertionError('No request fits the original budget')
    with pytest.raises(requests.Timeout):runner.bounded_get(SimpleNamespace(get=forbidden),'url',[0],timeout=0)
