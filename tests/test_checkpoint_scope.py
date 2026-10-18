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


def test_repeat_selection_uses_slowest_actual_clean_upload_even_for_short_video():
    groups=[dict(group_id='long',duration_sec=140),dict(group_id='short',duration_sec=15)]
    records=[dict(group_id='long',status='done',elapsed_sec=45),
             dict(group_id='short',status='done',elapsed_sec=75)]
    selected,basis=checkpoint.repeat_source(groups,records)
    assert selected['group_id']=='short'
    assert basis==dict(basis='slowest_completed_clean_api',observed_clean_api_sec=75)


def test_repeat_selection_falls_back_only_without_usable_completed_clean_latency():
    groups=[dict(group_id='long',duration_sec=140),dict(group_id='short',duration_sec=15)]
    records=[dict(group_id='short',status='failed',elapsed_sec=300),
             dict(group_id='long',status='done',elapsed_sec=float('nan')),
             dict(group_id='unrequested',status='done',elapsed_sec=350)]
    selected,basis=checkpoint.repeat_source(groups,records)
    assert selected['group_id']=='long'
    assert basis['basis']=='longest_duration_fallback_no_completed_clean_api'


@pytest.mark.parametrize('api_only',[False,True])
def test_checkpoint_preserves_clean_api_and_repeats_before_optional_augmentation(monkeypatch,tmp_path,api_only,capsys):
    recipe='a'*64
    groups=[dict(group_id=f'g{i:02}',target=0,video_ids=[f'X-{i+1}'],duration_sec=150-i)
            for i in range(40)]
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(dict(groups=groups)))
    annotations=tmp_path/'annotations.json';annotations.write_text(json.dumps(dict(events=[],taxonomy={})))
    honest=tmp_path/'honest.json';honest.write_text(json.dumps(dict(recipe_sha256=recipe)))
    monkeypatch.setattr(checkpoint.config,'DATA_PROC',tmp_path)
    monkeypatch.setattr(checkpoint,'PROTOCOL',protocol);monkeypatch.setattr(checkpoint,'ANNOTATIONS',annotations)
    calls=[]
    def fake_run(args,output):
        if '--augment' in args:
            calls.append('augmentation')
            raise RuntimeError('Interrupted during first augmentation')
        if '--only' in args:
            selected=[g for g in groups if g['group_id']==args[args.index('--only')+1]]
            calls.append('repeat')
        else:
            subset=args[args.index('--subset')+1];calls.append(subset)
            selected=groups[:10] if subset=='reserve' else groups[10:]
        folder=output.with_suffix('.traces');folder.mkdir()
        records=[]
        for g in selected:
            elapsed=75 if g['group_id']=='g39' else 12
            records.append(dict(group_id=g['group_id'],status='done',elapsed_sec=elapsed,
                                result=dict(verdict='no_violation',violations=[])))
            trace=dict(job_id=f'actual-job-{len(calls)}-{g["group_id"]}',status='done',api_elapsed_sec=elapsed,
                       analysis=dict(engine='mission-v1',recipe_sha256=recipe,facts={'private':'not in summary'}))
            (folder/(g['group_id']+'.json')).write_text(json.dumps(trace))
        report=checkpoint.score_records(records,selected,[],{})
        report.update(mode='api',subset=args[args.index('--subset')+1],records=records)
        output.write_text(json.dumps(report));return report
    monkeypatch.setattr(checkpoint,'run',fake_run)
    root=tmp_path/'interrupted-checkpoint'
    args=['--api','https://public.example','--key-file',str(tmp_path/'key'),
          '--run-dir',str(root),'--honest',str(honest)]
    if api_only:
        checkpoint.main(args+['--api-only'])
        assert calls==['reserve','dev','repeat','repeat']
    else:
        with pytest.raises(RuntimeError,match='Interrupted during first augmentation'):
            checkpoint.main(args)
        assert calls==['reserve','dev','repeat','repeat','augmentation']
    stage=json.loads((root/'api-stage-summary.json').read_text())
    runtime=json.loads((root/'runtime-qualification.json').read_text())
    assert stage['api']['n_videos']==stage['api']['completed']==40
    assert len(stage['clean_job_ids'])==len(set(stage['clean_job_ids']))==40
    assert all(j['recipe_sha256']==recipe and j['engine']=='mission-v1' for j in stage['clean_job_provenance'])
    assert stage['stage_full_aug_pending'] is True and stage['all_requested_transforms_complete'] is False
    assert stage['selected_recipe_sha256']==runtime['selected_recipe_sha256']==recipe
    assert len(stage['evaluator_sha256'])==64 and stage['evaluator_sha256']==runtime['evaluator_sha256']
    assert stage['worst_clean_api_sec']==runtime['worst_clean_api_sec']==75
    assert runtime['two_worst_case_runs_passed'] is True
    assert runtime['worst_case_repeat_selection']['group_id']=='g39'
    assert len({r['job_id'] for r in runtime['worst_case_repeats']})==2
    assert 'facts' not in json.dumps(stage)
    if api_only:
        final=json.loads((root/'summary.json').read_text())
        augmented=json.loads((root/'augmented.json').read_text())
        assert stage['scope']=='full_clean_api_only' and stage['api_only'] is True
        assert stage['full_set_requested'] is True and stage['full_aug_requested'] is False
        assert final['api']['mode']=='api' and final['api']['subset']=='all'
        assert final['api']['n_videos']==final['api']['completed']==40
        assert final['stage_full_aug_pending'] is True and final['robustness_credit_eligible'] is False
        assert augmented['deferred'] is True and augmented['all_requested_transforms_complete'] is False
        assert augmented['transforms']==[] and augmented['completed_transforms']==0
        assert 'accuracy' not in augmented  # No invented unmeasured robustness score.
        assert final['cheat_replay']['not_run'] is True and final['cheat_replay']['passed'] is False
        assert not list(root.glob('aug-*.json'))
        rubric=calculate(dict(mode='cached',validation_kind='leave-one-setup-out',fold_integrity_verified=True,accuracy=1),
                        api=final['api'],augmented=augmented,gates=dict(pytest_green=True))
        assert rubric['components']['robustness']==0
        assert 'facts' not in capsys.readouterr().out
        counter=json.loads((tmp_path/'mission-reserve-checkpoints.json').read_text())
        assert len(counter)==1 and counter[0]['run_dir']==str(root)
    else:
        assert not (root/'summary.json').exists() and not (root/'augmented.json').exists()
    assert not list(root.glob('*.tmp'))


def test_api_only_preserves_four_reserve_check_guard_without_any_upload(monkeypatch,tmp_path):
    monkeypatch.setattr(checkpoint.config,'DATA_PROC',tmp_path)
    honest=tmp_path/'honest.json';honest.write_text('{}')
    counter=tmp_path/'mission-reserve-checkpoints.json'
    original=[dict(started_utc='previous',run_dir=str(tmp_path/f'check-{i}')) for i in range(4)]
    counter.write_text(json.dumps(original))
    before=counter.read_bytes()
    monkeypatch.setattr(checkpoint,'run',lambda *a,**k:pytest.fail('Exhausted reserve guard allowed an upload'))
    with pytest.raises(ValueError,match='Four reserve checks already consumed'):
        checkpoint.main(['--api','https://public.example','--key-file',str(tmp_path/'key'),
                         '--run-dir',str(tmp_path/'fifth'),'--honest',str(honest),'--api-only'])
    assert counter.read_bytes()==before
    assert not list(tmp_path.glob('*.tmp'))


def test_api_only_cannot_silently_use_quick_partial_scope(monkeypatch,tmp_path):
    monkeypatch.setattr(checkpoint.config,'DATA_PROC',tmp_path)
    monkeypatch.setattr(checkpoint,'run',lambda *a,**k:pytest.fail('Conflicting scope flags allowed an upload'))
    with pytest.raises(SystemExit) as exc:
        checkpoint.main(['--api','https://public.example','--key-file',str(tmp_path/'key'),
                         '--run-dir',str(tmp_path/'invalid'),'--honest',str(tmp_path/'not-read'),
                         '--api-only','--quick'])
    assert exc.value.code==2
    assert not (tmp_path/'invalid').exists()
    assert not (tmp_path/'mission-reserve-checkpoints.json').exists()
