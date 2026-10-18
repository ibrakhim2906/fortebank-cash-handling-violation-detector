"""Synthetic metadata only: action truth, provenance and holdout safeguards."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.video_actions import (
    ACTION_VOCABULARY, aggregate_summary, append_proposal, append_review,
    load_queue, load_reviews, prepare_queue, verified_training_labels,
)
from src.video_eval import _fingerprint, create_or_load_protocol


@pytest.fixture
def action_queue(tmp_path):
    rows = [{"video_id": f"synthetic-{index}", "path": f"local/not-real-{index}.mov",
             "sha256": hashlib.sha256(f"synthetic-{index}".encode()).hexdigest(),
             "target": int(index < 21), "timestamp_sec": 10.0 if index < 21 else np.nan,
             "duration_sec": 80.0, "localization_eligible": bool(index < 21)}
            for index in range(40)]
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    protocol_path = tmp_path / "protocol.json"
    protocol = create_or_load_protocol(inventory, protocol_path)
    # An ambiguous duplicate exists in development; preserve both source anchors.
    group = next(item for item in protocol["groups"]
                 if item["group_id"] in protocol["development_group_ids"] and item["target"])
    group["video_ids"].append("synthetic-duplicate")
    group["paths"].append("local/not-real-duplicate.mov")
    group["timestamp_annotations"].append({"video_id": "synthetic-duplicate", "timestamp_sec": 68.0})
    group["timestamp_annotations"][0]["timestamp_sec"] = 57.0
    group["timestamp_sec"], group["localization_eligible"] = None, False
    protocol["fingerprint"] = _fingerprint(protocol)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    out_dir = tmp_path / "action-labels"
    queue = prepare_queue(protocol_path, out_dir)
    return out_dir, protocol_path, queue, protocol


def _human(task, **overrides):
    return {"task_id": task["task_id"], "reviewer_id": "local-operator",
            "status": "verified", "interval": dict(task["interval"]), "coverage": "interval",
            "visibility": "clear", "actions": [{"action_id": "object_handoff", "presence": "present"}],
            "provenance": {"kind": "human_review"}, **overrides}


def _proposal_provenance():
    return {"model_id": "synthetic-local-model", "weights_sha256": "a" * 64,
            "recipe_sha256": "b" * 64, "frame_sha256s": ["c" * 64]}


def test_prepare_only_development_metadata_and_preserves_conflicting_anchors(action_queue):
    out_dir, protocol_path, queue, protocol = action_queue
    assert len(queue["tasks"]) == 33
    assert {task["group_id"] for task in queue["tasks"]} == set(protocol["development_group_ids"])
    assert not {task["group_id"] for task in queue["tasks"]} & set(protocol["holdout_group_ids"])
    conflict = [task for task in queue["tasks"] if task["ambiguous_anchor"]]
    assert {task["anchor_sec"] for task in conflict} == {57.0, 68.0}
    assert all(len(task["source_annotations"]) == 2 for task in conflict)
    assert all(task["initial_status"] == "unknown" and task["actions"] == [] for task in queue["tasks"])
    assert all(task["anchor_sec"] is None for task in queue["tasks"] if not task["official_target"])
    assert len(ACTION_VOCABULARY) == 8
    original = (out_dir / "queue.json").read_bytes()
    assert prepare_queue(protocol_path, out_dir) == queue
    assert original == (out_dir / "queue.json").read_bytes()


def test_preparation_never_opens_media(action_queue, monkeypatch):
    out_dir, protocol_path, _, _ = action_queue
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        assert self.suffix.lower() != ".mov", "metadata workflow must not open footage"
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    prepare_queue(protocol_path, out_dir)
    assert aggregate_summary(out_dir, protocol_path)["holdout_groups"] == 0


def test_holdout_task_or_modified_queue_cannot_enter_reviews(action_queue):
    out_dir, protocol_path, queue, protocol = action_queue
    with pytest.raises(ValueError, match="not in the frozen development"):
        append_review(_human({**queue["tasks"][0], "task_id": protocol["holdout_group_ids"][0]}),
                      out_dir, protocol_path)
    queue["tasks"][0]["group_id"] = protocol["holdout_group_ids"][0]
    (out_dir / "queue.json").write_text(json.dumps(queue), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from frozen"):
        load_queue(out_dir, protocol_path)


def test_negative_violation_clip_can_have_a_reviewed_action(action_queue):
    out_dir, protocol_path, queue, _ = action_queue
    negative = next(task for task in queue["tasks"] if task["official_target"] == 0)
    record = append_review(_human(negative), out_dir, protocol_path)
    labels = verified_training_labels(out_dir, protocol_path)
    assert labels[0]["target"] == 1
    assert record["source_sha256"] == negative["group_id"]
    assert record["provenance"]["kind"] == "human_review"
    assert aggregate_summary(out_dir, protocol_path)["verified_action_observations"] == 1


def test_models_never_create_verified_truth(action_queue):
    out_dir, protocol_path, queue, _ = action_queue
    task = queue["tasks"][0]
    provenance = _proposal_provenance()
    proposal = append_proposal(out_dir, task["task_id"],
                              [{"action_id": "object_handoff", "presence": "present"}],
                              "clear", provenance, protocol_path)
    assert proposal["status"] == "proposed" and proposal["reviewer_id"] is None
    assert verified_training_labels(out_dir, protocol_path) == []
    invalid = _human(task, provenance={**provenance, "kind": "local_model_proposal"})
    with pytest.raises(ValueError, match="cannot be verified"):
        append_review(invalid, out_dir, protocol_path)
    assert aggregate_summary(out_dir, protocol_path)["proposal_records"] == 1


def test_append_only_revision_and_ambiguous_review_revokes_training_label(action_queue):
    out_dir, protocol_path, queue, _ = action_queue
    first = append_review(_human(queue["tasks"][0]), out_dir, protocol_path)
    original = (out_dir / "reviews.jsonl").read_bytes()
    second = append_review(_human(queue["tasks"][0], status="ambiguous", actions=[]), out_dir, protocol_path)
    assert (out_dir / "reviews.jsonl").read_bytes().startswith(original)
    assert [record["revision"] for record in load_reviews(out_dir, protocol_path)] == [1, 2]
    assert second["previous_record_sha256"] == first["record_sha256"]
    assert verified_training_labels(out_dir, protocol_path) == []


def test_later_model_proposal_cannot_replace_human_truth(action_queue):
    out_dir, protocol_path, queue, _ = action_queue
    task = queue["tasks"][0]
    append_review(_human(task), out_dir, protocol_path)
    append_proposal(out_dir, task["task_id"], [{"action_id": "object_handoff", "presence": "absent"}],
                    "clear", _proposal_provenance(), protocol_path)
    assert verified_training_labels(out_dir, protocol_path)[0]["target"] == 1


@pytest.mark.parametrize("mutation", [
    {"reviewer_id": None}, {"reviewer_id": "\ninvalid"},
    {"source_sha256": "a" * 64}, {"partition": "holdout"},
    {"interval": {"start_sec": -1, "end_sec": 5}},
    {"interval": {"start_sec": 1, "end_sec": float("nan")}},
    {"interval": {"start_sec": 1, "end_sec": 1000}},
    {"interval": {"start_sec": 5, "end_sec": 5}},
    {"coverage": "entire_clip"}, {"visibility": "unobservable"},
    {"actions": [{"action_id": "invented_rule_or_action", "presence": "present"}]},
    {"actions": [{"action_id": "object_handoff", "presence": "unknown"}]},
    {"actions": [{"action_id": "unknown_observable_action", "presence": "present"}]},
    {"actions": [{"action_id": "object_handoff", "presence": "present"}] * 2},
    {"visibility": "partial", "actions": [{"action_id": "object_handoff", "presence": "absent"}]},
])
def test_invalid_reviews_are_rejected_without_journal_mutation(action_queue, mutation):
    out_dir, protocol_path, queue, _ = action_queue
    with pytest.raises(ValueError):
        append_review(_human(queue["tasks"][0], **mutation), out_dir, protocol_path)
    assert not (out_dir / "reviews.jsonl").exists()


def test_full_clip_action_absence_requires_explicit_review_and_does_not_expand_unknowns(action_queue):
    out_dir, protocol_path, queue, _ = action_queue
    task = queue["tasks"][0]
    append_review(_human(task, coverage="entire_clip",
                         interval={"start_sec": 0, "end_sec": task["duration_sec"]},
                         actions=[{"action_id": "object_handoff", "presence": "absent"}]), out_dir, protocol_path)
    labels = verified_training_labels(out_dir, protocol_path)
    assert len(labels) == 1 and labels[0]["target"] == 0
    assert labels[0]["coverage"] == "entire_clip"


def test_multilabel_review_and_unobservable_proposal_remain_separate(action_queue):
    out_dir, protocol_path, queue, _ = action_queue
    task = queue["tasks"][0]
    append_proposal(out_dir, task["task_id"],
                    [{"action_id": "object_handoff", "presence": "present", "visibility": "unobservable"}],
                    "partial", _proposal_provenance(), protocol_path)
    assert load_reviews(out_dir, protocol_path)[0]["actions"][0]["presence"] == "unknown"
    append_review(_human(task, actions=[{"action_id": "object_handoff", "presence": "present"},
                                       {"action_id": "container_open_close", "presence": "absent"}]), out_dir, protocol_path)
    assert len(verified_training_labels(out_dir, protocol_path)) == 2


def test_journal_tampering_is_detected(action_queue):
    out_dir, protocol_path, queue, _ = action_queue
    append_review(_human(queue["tasks"][0]), out_dir, protocol_path)
    path = out_dir / "reviews.jsonl"
    record = json.loads(path.read_text())
    record["actions"][0]["presence"] = "absent"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        load_reviews(out_dir, protocol_path)


def test_concurrent_local_appends_keep_every_revision(action_queue):
    out_dir, protocol_path, queue, _ = action_queue
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda _: append_review(_human(queue["tasks"][0]), out_dir, protocol_path), range(6)))
    assert {record["revision"] for record in records} == set(range(1, 7))
    assert len(load_reviews(out_dir, protocol_path)) == 6


def test_raw_destination_is_never_created(action_queue, monkeypatch, tmp_path):
    _, protocol_path, _, _ = action_queue
    raw = tmp_path / "raw"
    monkeypatch.setattr(config, "DATA_RAW", raw)
    with pytest.raises(ValueError, match="never write"):
        prepare_queue(protocol_path, raw / "forbidden")
    assert not raw.exists()
