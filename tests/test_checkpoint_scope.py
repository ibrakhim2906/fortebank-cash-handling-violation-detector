import json
import pytest
from scripts import mission_checkpoint as checkpoint
from scripts.eval_video import PROTOCOL,SPLIT
from scripts.mission_checkpoint import quick_groups
from scripts.rubric_score import calculate


def test_bounded_checkpoint_excludes_reserve_and_covers_both_classes():
    groups=json.loads(PROTOCOL.read_text())['groups'];split=json.loads(SPLIT.read_text())
    selected=quick_groups(groups,split)
    assert selected==quick_groups(list(reversed(groups)),split)
    assert set(selected)<=set(split['dev']) and not set(selected)&set(split['reserve'])
    assert sorted(g['target'] for g in groups if g['group_id'] in selected)==[0,0,1,1]


def test_partial_api_and_robustness_cannot_receive_full_set_rubric_credit():
    honest=dict(mode='cached',validation_kind='leave-one-setup-out',fold_integrity_verified=True,accuracy=1)
    partial_api=dict(mode='api',subset='partial_checkpoint',n_videos=14,completed=14,failed=0,worst_sec=12)
    partial_robustness=dict(accuracy=1,clean_accuracy=1,all_requested_transforms_complete=False)
    result=calculate(honest,api=partial_api,augmented=partial_robustness,gates=dict(pytest_green=True))
    assert result['components']['product']==result['components']['robustness']==0


def traced_repeats(monkeypatch,tmp_path,*,engine='mission-v1',recipe=None,elapsed=18,
                   status='done',job_id=None,missing_trace=False,traced_elapsed=None):
    recipe=recipe or 'a'*64
    calls=[]
    def fake_run(args,output):
        calls.append(args)
        trace=dict(job_id=job_id or f'actual-job-{len(calls)}',status=status,
                   api_elapsed_sec=elapsed if traced_elapsed is None else traced_elapsed,
                   analysis=dict(engine=engine,recipe_sha256=recipe,facts={'private':'not in summary'}))
        folder=output.with_suffix('.traces');folder.mkdir()
        if not missing_trace:(folder/'longest.json').write_text(json.dumps(trace))
        value=dict(records=[dict(group_id='longest',status=status,elapsed_sec=elapsed)])
        output.write_text(json.dumps(value))
        return value
    monkeypatch.setattr(checkpoint,'run',fake_run)
    return calls


def test_longest_repeats_capture_distinct_actual_jobs_and_selected_primary_recipe(monkeypatch,tmp_path):
    calls=traced_repeats(monkeypatch,tmp_path)
    rows,qualification=checkpoint.worst_case_repeats(dict(group_id='longest'),'https://public.example',tmp_path/'key',tmp_path,'a'*64)
    assert len(calls)==2 and calls[0][:4]==['--subset','all','--only','longest']
    assert all('--api' in call and '--key-file' in call for call in calls)
    assert [r['job_id'] for r in rows]==['actual-job-1','actual-job-2']
    assert all(r['primary_recipe_verified'] and r['runtime_qualified'] for r in rows)
    assert all(len(r['trace_sha256'])==64 and r['analysis']==dict(engine='mission-v1',recipe_sha256='a'*64) for r in rows)
    assert qualification['two_worst_case_runs_passed'] is True
    assert qualification['qualified_worst_sec']==qualification['observed_worst_sec']==18


@pytest.mark.parametrize('kwargs,selected',[
    ({'engine':'mission-v1-degraded'},'a'*64),
    ({'recipe':'b'*64},'a'*64),
    ({'status':'failed'},'a'*64),
    ({'missing_trace':True},'a'*64),
    ({'traced_elapsed':17},'a'*64),
    ({'elapsed':201},'a'*64),
    ({'elapsed':float('nan')},'a'*64),
    ({'job_id':'reused-job'},'a'*64),
    ({},None),
])
def test_unproven_or_slow_runs_never_qualify_repeated_runtime(monkeypatch,tmp_path,kwargs,selected):
    traced_repeats(monkeypatch,tmp_path,**kwargs)
    rows,qualification=checkpoint.worst_case_repeats(dict(group_id='longest'),'https://public.example',tmp_path/'key',tmp_path,selected)
    assert len(rows)==2 and all('status' in r and 'elapsed_sec' in r for r in rows)
    assert qualification['two_worst_case_runs_passed'] is False
    assert qualification['qualified_worst_sec'] is None


def test_checkpoint_freeze_is_explicit_and_missing_recipe_cannot_qualify():
    assert checkpoint.selected_recipe({}) is None
    assert checkpoint.selected_recipe(dict(recipe_sha256='a'*64))=='a'*64
    assert checkpoint.selected_recipe(dict(recipe_sha256='a'*64),dict(recipe_sha256='b'*64))=='b'*64
    for malformed in [{},dict(recipe_sha256='not-a-hash'),dict(recipe_sha256=3)]:
        with pytest.raises(ValueError,match='frozen serving recipe'):
            checkpoint.selected_recipe({},malformed)
