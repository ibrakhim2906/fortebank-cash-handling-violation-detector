"""Opt-in recall: shifted coverage, confirmed rescue and dense onset timing."""
import time
import pytest
from service.pipeline import auditor as a
from service.pipeline import state_ledger as s


def source(seconds=40):
    return [dict(timestamp_sec=i/2,image=b'synthetic',source_frame_index=i) for i in range(seconds*2)]


def candidate(kind='food_drink',support='clear',excluded=True,onset=3,evidence=4):
    return a.Candidate(kind=kind,onset_frame=onset,evidence_frame=evidence,support=support,counterexamples_excluded=excluded,
                       complete_phase_visible=False,visible_fact='Синтетическое наблюдение.')


def confirmation(support='clear',onset=2,evidence=3):
    return a.Confirmation(support=support,onset_frame=onset,evidence_frame=evidence,complete_phase_visible=False,visible_fact='Подтверждено.')


class Model:
    """Synthetic model: answers by request kind and records every request."""
    def __init__(self,primary=(),shifted=(),confirm='clear',onset=None,fail=()):
        self.primary=list(primary);self.shifted=list(shifted);self.confirm=confirm;self.onset=onset;self.fail=set(fail);self.calls=[]

    def __call__(self,frames,prompt,model,deadline,*,requester=None,max_tokens=768):
        times=[f['timestamp_sec'] for f in frames]
        if model is a.Sweep:
            stage='sweep' if not any(c[0]=='sweep' for c in self.calls) else 'shifted_sweep'
            observations=self.primary if stage=='sweep' else self.shifted
            answer=a.Sweep(visibility='clear',observations=observations)
        elif model is a.Confirmation:
            stage='focused';answer=confirmation(self.confirm)
        elif model is a.Onset:
            stage='onset';answer=self.onset or a.Onset(located=False,onset_frame=0,evidence_frame=0,visible_transition='Нет.')
        else:raise AssertionError(model)
        self.calls.append((stage,times))
        if stage in self.fail:raise TimeoutError('synthetic')
        return answer,dict(source_times=times)

    def stages(self):return [c[0] for c in self.calls]


@pytest.fixture
def env(monkeypatch):
    for key,value in dict(AUDITOR_CONFIRMATION='proposal',AUDITOR_SPATIAL='off',AUDITOR_DESK_RESCAN='off',
                          AUDITOR_ACTION_WINDOWS='off',AUDITOR_FRAMES='16',AUDITOR_WORKFLOW='original').items():
        monkeypatch.setenv(key,value)
    return monkeypatch


def run(env,mode,model,frames=None):
    env.setenv('AUDITOR_RECALL',mode);env.setattr(a,'ask',model)
    return a.confirmed_observations(frames or source(),time.monotonic()+160)


def test_interleaved_uses_actual_source_samples_strictly_between_primary_samples():
    frames=source();overview=a.choose(frames,16);shifted=a.interleaved(frames,overview)
    assert len(shifted)==15
    for left,middle,right in zip(overview,shifted,overview[1:]):
        assert left['timestamp_sec']<middle['timestamp_sec']<right['timestamp_sec'] and middle in frames
    dense=source(4);assert a.interleaved(dense,a.choose(dense,16))==[]


def test_off_mode_is_the_selected_single_sweep_recipe(env):
    model=Model(primary=[candidate(support='uncertain')])
    events,logs=run(env,'off',model)
    assert events==[] and model.stages()==['sweep']


def test_unknown_recall_mode_is_rejected(env):
    with pytest.raises(ValueError):run(env,'aggressive',Model())


def test_uncertain_primary_candidate_needs_independent_dense_confirmation(env):
    model=Model(primary=[candidate(support='uncertain')])
    events,logs=run(env,'rescue',model)
    assert model.stages()==['sweep','shifted_sweep','focused']
    focused=model.calls[2][1]
    # The recheck is denser than the sparse sweep that left it uncertain.
    sweep=model.calls[0][1]
    assert max(b-a for a,b in zip(focused,focused[1:]))<min(b-a for a,b in zip(sweep,sweep[1:]))
    assert [(e['type_id'],e['origin']) for e in events]==[('food_drink','recall_confirmed_self_hosted_observation')]
    assert events[0]['timecode_sec']==focused[2] and events[0]['visible_fact']=='Подтверждено.'
    assert not any(row['stage']=='audit_incomplete' for row in logs)


def test_unconfirmed_rescue_reports_nothing(env):
    events,_=run(env,'rescue',Model(primary=[candidate(support='uncertain')],confirm='uncertain'))
    assert events==[]


def test_shifted_sweep_finds_a_concern_between_primary_samples(env):
    model=Model(shifted=[candidate('hand_to_hand_transfer',onset=6,evidence=6)])
    events,_=run(env,'rescue',model)
    shifted=model.calls[1][1]
    assert model.stages()==['sweep','shifted_sweep','focused']
    assert shifted==[f['timestamp_sec'] for f in a.interleaved(source(),a.choose(source(),16))]
    assert [e['type_id'] for e in events]==['hand_to_hand_transfer']
    assert abs(events[0]['timecode_sec']-shifted[6])<=4


def test_already_accepted_or_settled_primary_concern_is_not_rechecked(env):
    model=Model(primary=[candidate()],shifted=[candidate(onset=3,evidence=4)])
    events,_=run(env,'rescue',model)
    assert model.stages()==['sweep','shifted_sweep']
    assert [e['origin'] for e in events]==['single_pass_self_hosted_observation']


def test_two_agreeing_views_are_confirmed_once(env):
    model=Model(primary=[candidate(support='uncertain')],shifted=[candidate(support='uncertain',onset=3,evidence=3)])
    events,_=run(env,'rescue',model)
    assert model.stages().count('focused')==1 and len(events)==1


def test_confirmations_are_bounded(env):
    kinds=['food_drink','personal_items','hand_to_hand_transfer','door_left_open','cash_left_out']
    model=Model(primary=[candidate(k,support='uncertain',onset=i,evidence=i) for i,k in enumerate(kinds)])
    events,logs=run(env,'rescue',model)
    assert model.stages().count('focused')==a.RECALL_CONFIRMATIONS==len(events)
    assert any(row.get('error_code')=='RecallConfirmationLimit' for row in logs)


@pytest.mark.parametrize('kind',sorted(a.OMISSIONS))
def test_omissions_are_never_rescued_from_sparse_views(env,kind):
    model=Model(primary=[candidate(kind,support='uncertain')],shifted=[candidate(kind)])
    events,_=run(env,'rescue',model)
    assert events==[] and 'focused' not in model.stages()


def test_not_visible_candidates_are_not_rescued(env):
    model=Model(primary=[candidate(support='not_visible')])
    events,_=run(env,'rescue',model)
    assert events==[] and 'focused' not in model.stages()


def test_tiny_details_still_need_a_second_subsample(env):
    model=Model(primary=[candidate('keys_left_in_safe',support='uncertain')])
    events,_=run(env,'rescue',model)
    assert model.stages()==['sweep','shifted_sweep','focused','focused'] and len(events)==1


@pytest.mark.parametrize('failing,reported',[('shifted_sweep',['food_drink']),('focused',[])])
def test_recall_failure_never_makes_primary_negative_incomplete(env,failing,reported):
    model=Model(primary=[candidate(support='uncertain')],fail={failing})
    events,logs=run(env,'rescue',model)
    # A failed shifted sweep does not block confirming the primary candidate;
    # a failed confirmation never reports it.
    assert [e['type_id'] for e in events]==reported
    assert any(row['stage']=='recall_incomplete' for row in logs)
    assert not any(row['stage']=='audit_incomplete' for row in logs)


def test_storage_veto_still_governs_rescued_storage(env):
    env.setenv('AUDITOR_SPATIAL','storage_veto');seen=[]
    def veto(proposed,*args,**kwargs):seen.append(proposed.kind);return True,[dict(stage='storage_veto_decision',veto=True)]
    env.setattr(s,'storage_counterexample',veto)
    events,_=run(env,'rescue',Model(primary=[candidate('safe_left_open',support='uncertain')]))
    assert seen==['safe_left_open'] and events==[]


def test_dense_onset_moves_sparse_proposal_time_inside_neighbouring_samples(env):
    frames=source();overview=a.choose(frames,16)
    model=Model(primary=[candidate(onset=3,evidence=4)],onset=a.Onset(located=True,onset_frame=5,evidence_frame=7,visible_transition='Подносит стакан ко рту.'))
    events,logs=run(env,'localize',model,frames)
    assert model.stages()==['sweep','onset']
    dense=model.calls[1][1]
    assert dense[0]==overview[2]['timestamp_sec'] and dense[-1]==overview[4]['timestamp_sec']
    assert events[0]['timecode_sec']==dense[5] and events[0]['evidence_sec']==dense[7]
    assert events[0]['visible_fact']=='Подносит стакан ко рту.'
    assert [row['stage'] for row in logs if 'onset' in row['stage']]==['dense_onset']


def test_unlocated_onset_keeps_the_proposal_time(env):
    frames=source();overview=a.choose(frames,16)
    events,_=run(env,'localize',Model(primary=[candidate(onset=3,evidence=4)]),frames)
    assert events[0]['timecode_sec']==overview[3]['timestamp_sec']


def test_full_mode_localizes_then_recalls(env):
    model=Model(primary=[candidate()],shifted=[candidate('personal_items',onset=10,evidence=10)])
    events,_=run(env,'on',model)
    assert model.stages()==['sweep','onset','shifted_sweep','focused']
    assert [e['type_id'] for e in events]==['food_drink','personal_items']


def test_tiled_windows_do_not_multiply_recall_calls(env):
    env.setenv('AUDITOR_RECALL','on');model=Model(primary=[candidate(support='uncertain',onset=1,evidence=1)]);env.setattr(a,'ask',model)
    a.confirmed_observations(source(12),time.monotonic()+160,overview_count=8,allow_omissions=False)
    assert model.stages()==['sweep']
