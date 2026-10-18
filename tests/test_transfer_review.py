"""Focused transfer review uses synthetic metadata and bytes only."""
import asyncio
from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import prepare_transfer_review as transfer
from src import config, video_actions
from src.video_action_review import create_app
from src.video_eval import create_or_load_protocol
from test_video_action_review import Client, KEY


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(config, "DATA_PROC", processed)
    media = processed / "videos"
    media.mkdir()
    rows = []
    for index in range(40):
        path = media / f"synthetic-{index}.mov"
        payload = f"synthetic-video-{index}".encode()
        path.write_bytes(payload)
        rows.append({"video_id": f"synthetic-{index}", "path": str(path),
                     "sha256": hashlib.sha256(payload).hexdigest(), "target": int(index < 21),
                     "timestamp_sec": 10.0 if index < 21 else np.nan,
                     "duration_sec": 30.0, "localization_eligible": index < 21})
    inventory = processed / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    protocol_path = processed / "protocol.json"
    protocol = create_or_load_protocol(inventory, protocol_path)
    action_dir = processed / "actions"
    video_actions.prepare_queue(protocol_path, action_dir)
    directory = action_dir / "transfer-pilot-v1"
    queue = transfer.prepare_transfer_queue(directory, protocol_path)
    client = Client(create_app(api_key=KEY, out_dir=action_dir, protocol_path=protocol_path, media_root=media))
    return directory, protocol_path, protocol, queue, client, action_dir


def human(task, choice="transfer", **overrides):
    return {"task_id": task["task_id"], "reviewer_id": "synthetic-reviewer", "choice": choice,
            "interval": {"start_sec": 4., "end_sec": 12.}, "coverage": "interval",
            "visibility": "clear", "watched": True, "evidence_sec": 8.,
            "evidence_confirmed": choice == "transfer", "absence_confirmed": choice != "transfer",
            **overrides}


def test_queue_is_metadata_only_unknown_and_fixed_fold_without_reserved_groups(prepared, monkeypatch):
    directory, protocol_path, protocol, queue, _, _ = prepared
    original_open = type(directory).open
    def guarded(path, *args, **kwargs):
        assert path.suffix.lower() != ".mov"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(type(directory), "open", guarded)
    assert transfer.prepare_transfer_queue(directory, protocol_path) == queue
    assert len(queue["tasks"]) == 32 and len(queue["initial_task_ids"]) == 16
    initial = [task for task in queue["tasks"] if task["task_id"] in queue["initial_task_ids"]]
    assert len({task["group_id"] for task in initial}) == 16
    assert sum(task["pool"] == "train" for task in initial) == 8
    assert not {task["group_id"] for task in queue["tasks"]} & set(protocol["holdout_group_ids"])
    assert all(task["initial_status"] == "unknown" and "official_target" not in task for task in queue["tasks"])
    assert {task["group_id"] for task in queue["tasks"] if task["pool"] == "blind"} == set(protocol["folds"][0]["valid_group_ids"])


@pytest.mark.parametrize("mutation", [
    {"task_id": "f" * 64}, {"partition": "holdout"}, {"reviewer_id": ""}, {"watched": False},
    {"choice": "made_up"}, {"visibility": "partial"}, {"evidence_sec": None},
    {"evidence_sec": 20.}, {"evidence_confirmed": False},
    {"interval": {"start_sec": 0., "end_sec": 9.}}, {"coverage": "entire_clip"},
    {"provenance": {"kind": "local_model_proposal"}}, {"status": "proposed"},
])
def test_invalid_or_model_reviews_never_write(prepared, mutation):
    directory, protocol_path, _, queue, _, _ = prepared
    with pytest.raises((ValueError, TypeError)):
        transfer.append_transfer_review(human(queue["tasks"][0], **mutation), directory, protocol_path)
    assert not (directory / "reviews.jsonl").exists()


def test_negative_requires_complete_explicit_coverage_and_background_not_hard_negative(prepared):
    directory, protocol_path, _, queue, _, _ = prepared
    task = next(task for task in queue["tasks"] if task["pool"] == "train")
    for change in ({"absence_confirmed": False}, {"interval": {"start_sec": 4., "end_sec": 11.}}):
        with pytest.raises(ValueError):
            transfer.append_transfer_review(human(task, "contact_only", **change), directory, protocol_path)
    transfer.append_transfer_review(human(task, "background"), directory, protocol_path)
    record = transfer.load_transfer_labels(directory, protocol_path)["records"][0]
    assert record["target"] == 0 and record["hard_negative"] is False
    transfer.append_transfer_review(human(task, "same_person_relocation"), directory, protocol_path)
    assert transfer.load_transfer_labels(directory, protocol_path)["records"][0]["hard_negative"] is True


def test_append_only_ambiguous_revision_revokes_truth_and_preserves_general_proposals(prepared):
    directory, protocol_path, _, queue, _, action_dir = prepared
    task = next(task for task in queue["tasks"] if task["pool"] == "train")
    old = action_dir / "reviews.jsonl"
    old.write_bytes(b"preserved separate journal")
    first = transfer.append_transfer_review(human(task), directory, protocol_path)
    prefix = (directory / "reviews.jsonl").read_bytes()
    second = transfer.append_transfer_review(human(task, "uncertain", visibility="unobservable"), directory, protocol_path)
    assert (directory / "reviews.jsonl").read_bytes().startswith(prefix)
    assert second["revision"] == 2 and second["previous_record_sha256"] == first["record_sha256"]
    assert transfer.load_transfer_labels(directory, protocol_path)["records"] == []
    assert old.read_bytes() == b"preserved separate journal"


def test_hash_chain_tampering_rejected(prepared):
    directory, protocol_path, _, queue, _, _ = prepared
    transfer.append_transfer_review(human(queue["tasks"][0]), directory, protocol_path)
    path = directory / "reviews.jsonl"
    value = json.loads(path.read_text())
    value["reviewer_id"] = "changed"
    path.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="hash chain"):
        transfer.load_transfer_reviews(directory, protocol_path)


def test_transfer_journal_cannot_overwrite_existing_parent_journal_via_symlink(prepared):
    directory, protocol_path, _, queue, _, action_dir = prepared
    original = action_dir / "reviews.jsonl"
    original.write_bytes(b"preserved general proposals")
    (directory / "reviews.jsonl").symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        transfer.append_transfer_review(human(queue["tasks"][0]), directory, protocol_path)
    assert original.read_bytes() == b"preserved general proposals"


def test_blind_labels_hidden_until_fitted_artifact_commitment(prepared):
    directory, protocol_path, _, queue, _, _ = prepared
    blind = sorted((task for task in queue["tasks"] if task["pool"] == "blind"), key=lambda task: (task["group_id"], task["task_id"]))
    for index, task in enumerate(blind[:8]):
        transfer.append_transfer_review(human(task, "transfer" if index < 4 else "contact_only"), directory, protocol_path)
    private = transfer.load_transfer_labels(directory, protocol_path)
    assert not private["records"] and private["blind_selection_fingerprint"]
    assert private["quota_counts"]["blind"] == {"positive_recordings": 4, "hard_negative_recordings_after_positive_selection": 4}
    with pytest.raises(ValueError, match="frozen"):
        transfer.load_transfer_labels(directory, protocol_path, include_blind=True)
    released = transfer.load_transfer_labels(directory, protocol_path, include_blind=True, blind_release_fingerprint="a" * 64)
    assert len(released["records"]) == 8
    commitment = [{"task_id": record["task_id"], "review_sha256": record["review_sha256"]}
                  for record in sorted(released["records"], key=lambda record: record["task_id"])]
    assert transfer._digest(commitment) == private["blind_selection_fingerprint"]


def test_queue_mutation_and_raw_destination_rejected(prepared):
    directory, protocol_path, protocol, queue, _, _ = prepared
    wrong = deepcopy(queue)
    wrong["tasks"][0]["group_id"] = protocol["holdout_group_ids"][0]
    (directory / "queue.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="frozen"):
        transfer.load_transfer_queue(directory, protocol_path)
    with pytest.raises(ValueError, match="data/processed"):
        transfer.prepare_transfer_queue(config.DATA_RAW / "not-written", protocol_path)


def test_focused_api_is_private_sanitized_and_saves_reliably(prepared):
    directory, protocol_path, _, queue, client, _ = prepared
    task = queue["tasks"][0]
    async def scenario():
        assert (await client.request("GET", "/api/review/transfer/tasks", key="wrong"))[0] == 401
        status, result, _ = await client.request("GET", "/api/review/transfer/tasks")
        assert status == 200 and result["mode"] == "transfer"
        assert all(item["human_review"] is None and item["proposal"] is None for item in result["tasks"])
        assert not {"path", "official_target", "source_sha256", "group_id"} & result["tasks"][0].keys()
        assert (await client.request("GET", "/api/review/transfer/media/" + task["task_id"]))[1].startswith(b"synthetic-video-")
        status, result, _ = await client.request("POST", "/api/review/transfer/labels", json.dumps(human(task)).encode())
        assert status == 201 and result == {"saved": True, "task_id": task["task_id"], "revision": 1, "status": "verified"}
        status, summary, _ = await client.request("GET", "/api/review/transfer/summary")
        assert status == 200 and summary["verified_tasks"] == 1 and summary["holdout_groups"] == 0
    asyncio.run(scenario())


def test_focused_html_keeps_one_page_with_clear_controls(prepared):
    client = prepared[4]
    status, html, _ = asyncio.run(client.request("GET", "/", key=""))
    assert status == 200
    for control in (b"Object-transfer pilot", b' id="transferChoice"', b' id="evidenceConfirmed"', b' id="absenceConfirmed"'):
        assert control in html
    assert b"localStorage" not in html and b"https://" not in html
