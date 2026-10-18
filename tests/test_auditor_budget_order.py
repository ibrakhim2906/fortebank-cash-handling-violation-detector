"""Late confirmation must not discard supported later proposals."""
import time
from service.pipeline import auditor


def test_budget_does_not_discard_later_ready_proposal(monkeypatch):
    monkeypatch.setenv('AUDITOR_CONFIRMATION','proposal')
    monkeypatch.setenv('AUDITOR_FRAMES','64')
    frames=[dict(timestamp_sec=i/2,image=b'synthetic',source_frame_index=i) for i in range(20)]
    def candidate(kind):
        return auditor.WideCandidate(kind=kind,onset_frame=6,evidence_frame=8,support='clear',counterexamples_excluded=True,complete_phase_visible=False,visible_fact='Synthetic test')
    for kinds in [('workstation_unlocked','food_drink'),('food_drink','workstation_unlocked')]:
        sweep=auditor.WideSweep(visibility='clear',observations=[candidate(k) for k in kinds])
        monkeypatch.setattr(auditor,'ask',lambda *args,**kwargs:(sweep,{}))
        events,logs=auditor.confirmed_observations(frames,time.monotonic()+5)
        assert [event['type_id'] for event in events]==['food_drink']
        assert any(entry['stage']=='audit_incomplete' for entry in logs)
