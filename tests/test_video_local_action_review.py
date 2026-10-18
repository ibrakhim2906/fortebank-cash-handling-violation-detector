"""Synthetic tests only: no bank media, model inference or network requests."""
import copy
import json
from unittest.mock import Mock

import pytest

from src import video_local_action_review as review
from src import video_local_observations as observations
from src.video_actions import ACTION_IDS


def proposal(present=(), visibility="clear"):
    return {"actions": [{"action_id": identifier,
                         "presence": "present" if identifier in present else "unknown"}
                        for identifier in ACTION_IDS], "visibility": visibility}


def states(result):
    return {item["action_id"]: item["presence"] for item in result["actions"]}


def task(identifier="task-one", group="development-group"):
    return {"task_id": identifier, "group_id": group, "source_sha256": "a" * 64,
            "partition": "development", "path": "synthetic.mov",
            "interval": {"start_sec": 0, "end_sec": 4}}


def test_consensus_requires_both_passes_present_and_preserves_unknowns():
    left, right, shared = ACTION_IDS[:3]
    result = review.consensus(proposal((left, shared)), proposal((right, shared)))
    merged = states(result)
    assert merged[shared] == "present"
    assert merged[left] == merged[right] == "unknown"
    assert all(merged[identifier] == "unknown" for identifier in ACTION_IDS[3:])


def test_unknown_action_never_becomes_a_verified_observation():
    result = review.consensus(proposal(("unknown_observable_action",)),
                              proposal(("unknown_observable_action",)))
    assert states(result)["unknown_observable_action"] == "unknown"
    assert set(result) == {"actions", "visibility"}


@pytest.mark.parametrize("side", ["draft", "critique"])
def test_unobservable_either_pass_forces_every_action_unknown(side):
    draft = proposal(ACTION_IDS, "unobservable" if side == "draft" else "clear")
    critique = proposal(ACTION_IDS, "unobservable" if side == "critique" else "clear")
    result = review.consensus(draft, critique)
    assert result["visibility"] == "unobservable"
    assert set(states(result).values()) == {"unknown"}


@pytest.mark.parametrize("side", ["draft", "critique"])
@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown_id"])
def test_consensus_requires_exact_vocab_coverage(side, defect):
    draft, critique = proposal(), proposal()
    value = draft if side == "draft" else critique
    if defect == "missing":
        value["actions"].pop()
    elif defect == "duplicate":
        value["actions"][-1] = copy.deepcopy(value["actions"][0])
    else:
        value["actions"][-1]["action_id"] = "invented-action"
    with pytest.raises(ValueError):
        review.consensus(draft, critique)


def test_consensus_matches_ids_independently_of_input_order():
    draft, critique = proposal((ACTION_IDS[0],)), proposal((ACTION_IDS[0],))
    critique["actions"].reverse()
    result = review.consensus(draft, critique)
    assert [item["action_id"] for item in result["actions"]] == list(ACTION_IDS)
    assert states(result)[ACTION_IDS[0]] == "present"


@pytest.fixture
def local_run(tmp_path, monkeypatch):
    events = []
    verify = Mock(side_effect=lambda: events.append("verified") or {
        "model_id": "synthetic-local-model", "weights_sha256": "b" * 64,
        "projector_sha256": "c" * 64})
    queue = {"tasks": [task()]}
    load_history = Mock(return_value=[])
    sample = Mock(side_effect=lambda *a: events.append("sampled") or
                  ([b"synthetic-jpeg-one", b"synthetic-jpeg-two", b"synthetic-jpeg-three"],
                   [0.0, 2.0, 4.0]))
    request = Mock(side_effect=lambda *a, **k: proposal((ACTION_IDS[0],)))
    append = Mock()
    monkeypatch.setattr(review, "verify_ready", verify)
    monkeypatch.setattr(review, "review_recipe", lambda: ({"version": 3}, "f" * 64))
    monkeypatch.setattr(review, "load_queue", lambda *a: queue)
    monkeypatch.setattr(review, "load_protocol", lambda *a: {"holdout_group_ids": ["reserved-group"]})
    monkeypatch.setattr(review, "load_reviews", load_history)
    monkeypatch.setattr(review, "sample_interval", sample)
    monkeypatch.setattr(observations, "request_action_review", request)
    monkeypatch.setattr(review, "append_proposal", append)
    return {"directory": tmp_path / "actions", "queue": queue, "history": load_history,
            "verify": verify, "sample": sample, "request": request, "append": append,
            "events": events}


def run(local_run, **kwargs):
    return review.review_tasks(out_dir=local_run["directory"], protocol_path="synthetic-protocol", **kwargs)


def test_two_pass_output_remains_unverified_proposal_without_human_notes(local_run):
    result = run(local_run)
    assert local_run["events"] == ["verified", "sampled"]
    assert local_run["request"].call_count == 2
    first, second = local_run["request"].call_args_list
    assert "prior_actions" not in first.kwargs
    assert second.kwargs["prior_actions"] == proposal((ACTION_IDS[0],))["actions"]
    assert 0 < first.kwargs["timeout"] <= 45
    assert 0 < second.kwargs["timeout"] <= 45
    local_run["append"].assert_called_once()
    call = local_run["append"].call_args
    assert call.args[1] == "task-one"
    provenance = call.args[4]
    assert provenance["kind"] == "local_model_proposal"
    assert len(provenance["frame_sha256s"]) == 3
    assert "reviewer_id" not in provenance and "human_notes" not in provenance
    audits = list((local_run["directory"] / "local-review-v3").glob("*.json"))
    assert len(audits) == 1
    audit = json.loads(audits[0].read_text())
    assert audit["verified"] is False
    assert "reviewer_id" not in audit and "human_notes" not in audit
    assert audit["provenance"]["kind"] == "local_model_proposal"
    assert result["verified_labels_created"] == 0
    assert result["semantic_accuracy_verified"] is False
    assert result["machine_rechecked_proposals"] == 1


def test_append_boundary_uses_proposed_status_and_no_human_identity(local_run, monkeypatch):
    from src import video_actions

    # Exercise the actual proposal payload builder, intercepting its write.
    append_record = Mock()
    monkeypatch.setattr(video_actions, "load_queue", lambda *a: local_run["queue"])
    monkeypatch.setattr(video_actions, "append_review", append_record)
    local_run["append"].side_effect = video_actions.append_proposal
    run(local_run)
    payload = append_record.call_args.args[0]
    assert payload["status"] == "proposed"
    assert payload["reviewer_id"] is None
    assert payload["provenance"]["kind"] == "local_model_proposal"
    assert "human_notes" not in payload


@pytest.mark.parametrize("human_status", ["verified", "ambiguous", "unknown"])
def test_any_human_reviewed_task_is_skipped(local_run, human_status):
    local_run["history"].return_value = [{"task_id": "task-one", "status": human_status,
                                          "provenance": {"kind": "human_review"}}]
    result = run(local_run)
    assert result["tasks_attempted"] == 0
    local_run["sample"].assert_not_called()
    local_run["request"].assert_not_called()
    local_run["append"].assert_not_called()


def test_human_review_arriving_during_inference_prevents_model_append(local_run):
    human = {"task_id": "task-one", "provenance": {"kind": "human_review"}}
    local_run["history"].side_effect = [[], [human]]
    result = run(local_run)
    assert local_run["request"].call_count == 2
    local_run["append"].assert_not_called()
    assert result["machine_rechecked_proposals"] == 0
    assert result["verified_labels_created"] == 0


def test_reserved_holdout_rejected_before_sampling(local_run):
    local_run["queue"]["tasks"] = [task(group="reserved-group")]
    with pytest.raises(ValueError, match="Reserved footage"):
        run(local_run)
    local_run["sample"].assert_not_called()
    local_run["request"].assert_not_called()
    local_run["append"].assert_not_called()


@pytest.mark.parametrize("pilot_count,present,recipe", [(0, 0, "f" * 64), (2, 1, "f" * 64),
                                                       (3, 0, "f" * 64), (3, 1, "e" * 64)])
def test_bulk_expansion_requires_three_matching_pilot_audits_with_observations(
        local_run, pilot_count, present, recipe):
    directory = local_run["directory"] / "local-review-v3"
    directory.mkdir(parents=True)
    for index in range(pilot_count):
        (directory / f"pilot-{index}.json").write_text(json.dumps({
            "task_id": f"pilot-{index}", "recipe_sha256": recipe,
            "consensus_present_count": present, "verified": False}))
    with pytest.raises(ValueError, match="three completed pilot audits"):
        run(local_run, limit=4)
    local_run["sample"].assert_not_called()
    local_run["request"].assert_not_called()
    local_run["append"].assert_not_called()


def test_bulk_expansion_allowed_after_three_matching_nonempty_pilot_audits(local_run):
    directory = local_run["directory"] / "local-review-v3"
    directory.mkdir(parents=True)
    for index in range(3):
        (directory / f"pilot-{index}.json").write_text(json.dumps({
            "task_id": f"pilot-{index}", "recipe_sha256": "f" * 64,
            "consensus_present_count": int(index == 0), "verified": False}))
    result = run(local_run, limit=4)
    assert result["machine_rechecked_proposals"] == 1
    assert result["verified_labels_created"] == 0


def test_sampler_failure_is_explicit_and_creates_no_proposal(local_run):
    local_run["sample"].side_effect = ValueError("synthetic decode failure")
    result = run(local_run)
    assert result["tasks_failed"] == 1
    assert result["machine_rechecked_proposals"] == 0
    local_run["request"].assert_not_called()
    local_run["append"].assert_not_called()
