import json
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
