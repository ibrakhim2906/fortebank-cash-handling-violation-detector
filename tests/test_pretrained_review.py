"""Read-only probe access uses synthetic artifacts and byte strings, never bank footage."""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from src import config
from src.video_action_review import PRETRAINED_CLASSES, ROOT, create_app
from test_transfer_review import prepared as transfer_prepared
from test_video_action_review import Client, KEY


def write_artifacts(directory, manifest, records):
    manifest_bytes = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(manifest_bytes)
    (directory / "records.json").write_text(json.dumps({"schema_version": 1,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "records": records}))
    (directory / "summary.json").write_text(json.dumps({"schema_version": 1, "status": "completed",
        "repeatability_passed": True, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "records_sha256": hashlib.sha256((directory / "records.json").read_bytes()).hexdigest()}))


@pytest.fixture
def prepared(transfer_prepared):
    _, protocol_path, protocol, _, _, action_dir = transfer_prepared
    directory = config.DATA_PROC / "synthetic-probe"
    directory.mkdir()
    selected = sorted(protocol["folds"][0]["train_group_ids"], key=lambda identifier:
        hashlib.sha256(f"{config.SEED}:{identifier}".encode()).hexdigest())[:8]
    groups = {group["group_id"]: group for group in protocol["groups"]}
    selections = [{"group_id": identifier, "path": groups[identifier]["path"],
                   "source_sha256": groups[identifier]["sha256"], "duration_sec": 30.0}
                  for identifier in selected]
    manifest = {"schema_version": 1, "protocol_fingerprint": protocol["fingerprint"], "selections": selections}
    predictions = [{"label": name, "class_index": index, "score": 0.01, "rank_1based": rank}
                   for rank, (name, index) in enumerate(PRETRAINED_CLASSES.items(), 1)]
    records = [{**item, "decoded_duration_sec": 30.0,
                "windows": [{"start_sec": 0.0, "end_sec": 8.0, "timestamp_sec": 4.0,
                "predictions": deepcopy(predictions), "logits": [0.0] * 600,
                "sampling": {"private_synthetic_metadata": True}}]} for item in selections]
    write_artifacts(directory, manifest, records)
    app = create_app(api_key=KEY, out_dir=action_dir, protocol_path=protocol_path,
                     media_root=config.DATA_PROC / "videos", pretrained_dir=directory)
    return Client(app), directory, manifest, records, protocol, action_dir


def request(client, path, method="GET", **kwargs):
    return asyncio.run(client.request(method, "/api/review/pretrained/" + path, **kwargs))


def test_probe_requires_auth_and_never_creates_action_labels(prepared):
    client, _, _, _, _, action_dir = prepared
    for path in ("tasks", "summary", "media/unknown"):
        assert request(client, path, key="wrong")[0] == 401
    assert request(client, "labels", method="POST", body=b'{}')[0] == 404
    assert request(client, "tasks", method="POST", body=b'{}')[0] == 405
    assert not (action_dir / "reviews.jsonl").exists()
    assert not (action_dir / "transfer-pilot-v1/reviews.jsonl").exists()


def test_probe_only_exposes_train_scores_and_bounds(prepared):
    client, _, _, records, _, _ = prepared
    status, result, headers = request(client, "tasks")
    assert status == 200 and result["read_only"] is True and len(result["tasks"]) == 8
    assert headers[b"cache-control"] == b"no-store"
    for task in result["tasks"]:
        assert task["review_status"] == "unverified" and task["anchor_sec"] is None
        assert not {"path", "source_sha256", "official_target", "group_id", "pool"} & task.keys()
        assert set(task["windows"][0]) == {"start_sec", "end_sec", "timestamp_sec", "predictions"}
        assert all(set(item) == {"label", "score", "rank_1based"} for item in task["windows"][0]["predictions"])
    assert records[0]["group_id"] not in json.dumps(result)
    assert request(client, "summary")[1] == {"train_recordings": 8, "windows": 8,
        "blind_recordings": 0, "holdout_recordings": 0, "verified_labels_created": 0,
        "read_only": True, "scores_are_uncalibrated": True}


def test_probe_media_hash_and_unknown_ids_are_checked(prepared):
    client, _, _, records, _, _ = prepared
    task = request(client, "tasks")[1]["tasks"][0]
    path = "media/" + task["task_id"]
    status, payload, _ = request(client, path, headers={"Range": "bytes=0-3"})
    assert status == 206 and payload == b"synt"
    assert request(client, "media/" + "f" * 64)[0] == 404
    Path(records[0]["path"]).write_bytes(b"changed synthetic source")
    assert request(client, path)[0] == 409


def test_probe_retains_protocol_duration_and_uses_validated_decoded_clock(prepared):
    client, directory, manifest, records, _, _ = prepared
    records[0]["decoded_duration_sec"] = 30.1
    records[0]["windows"][0].update(start_sec=22.1, end_sec=30.1, timestamp_sec=26.1)
    write_artifacts(directory, manifest, records)
    status, result, _ = request(client, "tasks")
    assert status == 200 and result["tasks"][0]["duration_sec"] == 30.1
    assert records[0]["duration_sec"] == manifest["selections"][0]["duration_sec"] == 30.0
    records[0]["decoded_duration_sec"] = 30.251
    write_artifacts(directory, manifest, records)
    assert request(client, "tasks")[0] == 409


@pytest.mark.parametrize("mutation", ["missing", "failed", "repeat_failed", "records_hash", "manifest_hash"])
def test_probe_requires_final_success_marker_and_matching_artifact_bytes(prepared, mutation):
    client, directory, _, _, _, _ = prepared
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    if mutation == "missing": path.unlink()
    else:
        if mutation == "failed": summary["status"] = "failed"
        elif mutation == "repeat_failed": summary["repeatability_passed"] = False
        elif mutation == "records_hash": summary["records_sha256"] = "0" * 64
        elif mutation == "manifest_hash": summary["manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(summary))
    assert request(client, "tasks")[0] == 409
    assert request(client, "summary")[0] == 409


@pytest.mark.parametrize("pool", ["holdout", "blind", "different_train"])
def test_probe_rejects_nonselected_or_reserved_groups(prepared, pool):
    client, directory, manifest, records, protocol, _ = prepared
    selected = {item["group_id"] for item in manifest["selections"]}
    options = {"holdout": protocol["holdout_group_ids"],
               "blind": protocol["folds"][0]["valid_group_ids"],
               "different_train": [identifier for identifier in protocol["folds"][0]["train_group_ids"]
                                   if identifier not in selected]}
    identifier = options[pool][0]
    group = next(item for item in protocol["groups"] if item["group_id"] == identifier)
    replacement = {"group_id": identifier, "path": group["path"],
                   "source_sha256": group["sha256"], "duration_sec": group["duration_sec"]}
    manifest["selections"][0] = replacement
    records[0].update(replacement)
    write_artifacts(directory, manifest, records)
    assert request(client, "tasks")[0] == 409
    assert request(client, "media/" + hashlib.sha256(identifier.encode()).hexdigest())[0] == 409


@pytest.mark.parametrize("mutation", ["manifest_hash", "protocol", "source", "duration", "duplicate", "window",
                                      "timestamp", "class_index", "score", "rank", "class_name", "partial"])
def test_probe_rejects_malformed_or_mismatched_artifacts(prepared, mutation):
    client, directory, manifest, records, _, _ = prepared
    window = records[0]["windows"][0]
    if mutation == "protocol": manifest["protocol_fingerprint"] = "0" * 64
    elif mutation == "source": records[0]["source_sha256"] = "0" * 64
    elif mutation == "duration": records[0]["duration_sec"] = 31.0
    elif mutation == "duplicate": records[1] = records[0]
    elif mutation == "window": window["end_sec"] = 31.0
    elif mutation == "timestamp": window["timestamp_sec"] = float("nan")
    elif mutation == "class_index": window["predictions"][0]["class_index"] = 0
    elif mutation == "score": window["predictions"][0]["score"] = float("inf")
    elif mutation == "rank": window["predictions"][0]["rank_1based"] = True
    elif mutation == "class_name": window["predictions"][0]["label"] = "synthetic invented class"
    elif mutation == "partial": records.pop()
    write_artifacts(directory, manifest, records)
    if mutation == "manifest_hash":
        (directory / "manifest.json").write_text(json.dumps(manifest) + "\n")
    assert request(client, "tasks")[0] == 409
    assert request(client, "summary")[0] == 409


def test_probe_task_ids_cannot_be_posted_into_verified_labels(prepared):
    client, _, _, _, _, action_dir = prepared
    task_id = request(client, "tasks")[1]["tasks"][0]["task_id"]
    for path in ("/api/review/labels", "/api/review/transfer/labels"):
        result = asyncio.run(client.request("POST", path, body=json.dumps({"task_id": task_id}).encode()))
        assert result[0] == 422
    assert not (action_dir / "reviews.jsonl").exists()
    assert not (action_dir / "transfer-pilot-v1/reviews.jsonl").exists()


def test_existing_page_has_read_only_probe_fragment_and_no_save_path():
    html = (ROOT / "app/action_review.html").read_text()
    assert "window.location.hash==='#pretrained'" in html
    assert 'id="pretrainedFields" hidden' in html and 'id="annotationFields"' in html
    assert "$('annotationFields').hidden=probe" in html
    assert "reviewMode==='pretrained'||saving" in html
    assert "Pretrained predictions are read only" in html
    assert "uncalibrated softmax" in html and "No annotation is created" in html
    assert 'id="strongestProbe"' in html and "Model-selected window, unverified" in html
    assert "innerHTML" not in html and "localStorage" not in html
