"""Provenance compatibility checks with synthetic source/config only."""
import copy

import pytest

from scripts.migrate_qwen_decoder_cache import (OLD_PROPOSAL_SHA256, POLICY,
                                               checked_configs, unchanged_ast)


def configs():
    old = {"model": "pinned", "sampling": [0.125, 0.375, 0.625, 0.875],
           "inherited_sources": {"src/video_action_proposals.py": OLD_PROPOSAL_SHA256,
                                 "src/video_actions.py": "a" * 64}}
    new = copy.deepcopy(old)
    new["failed_seek_fallback"] = POLICY
    new["inherited_sources"]["src/video_action_proposals.py"] = "b" * 64
    return old, new


def test_only_decoder_policy_and_equivalent_proposal_source_may_change():
    old, new = configs()
    assert checked_configs(old, new) == old["inherited_sources"]
    assert "failed_seek_fallback" not in old


@pytest.mark.parametrize("kind", ["model", "sampling", "helper", "policy", "old_source"])
def test_unrelated_cache_changes_are_rejected(kind):
    old, new = configs()
    if kind == "model": new["model"] = "other-model"
    elif kind == "sampling": new["sampling"] = [0.5]
    elif kind == "helper": new["inherited_sources"]["src/video_actions.py"] = "c" * 64
    elif kind == "policy": new["failed_seek_fallback"] = "silently-use-zero"
    elif kind == "old_source": old["inherited_sources"]["src/video_action_proposals.py"] = "d" * 64
    with pytest.raises(ValueError):
        checked_configs(old, new)


def test_non_decoder_code_must_be_ast_identical():
    before = "LIMIT=4\ndef sampler(): return 1\ndef classifier(): return 7\n"
    repaired = "LIMIT=4\ndef sampler(): return 2\ndef classifier(): return 7\n"
    unchanged_ast(before, repaired, {"sampler"})
    with pytest.raises(ValueError):
        unchanged_ast(before, repaired.replace("return 7", "return 8"), {"sampler"})
    with pytest.raises(ValueError):
        unchanged_ast(before, repaired.replace("LIMIT=4", "LIMIT=2"), {"sampler"})


def test_config_assignment_allowed_only_with_separate_checked_config_guard():
    old = "FEATURE_CONFIG={'decoder':0}\ndef score(): return 1\n"
    new = "FEATURE_CONFIG={'decoder':1}\ndef score(): return 1\n"
    unchanged_ast(old, new, set(), config_assignment=True)
    with pytest.raises(ValueError):
        unchanged_ast(old, new, set())
