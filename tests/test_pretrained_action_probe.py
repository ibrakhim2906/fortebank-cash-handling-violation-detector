"""Synthetic metadata and fake extraction; no actual model or bank footage."""
from copy import deepcopy
import hashlib
import json
import os
import socket

import numpy as np
import pandas as pd
import pytest

from scripts import probe_pretrained_actions as probe
from src import video_eval


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.config, "ROOT", tmp_path)
    monkeypatch.setattr(probe.config, "DATA_PROC", tmp_path / "data/processed")
    video_dir = probe.config.DATA_PROC / "fortebank-20261017/open/videos"
    video_dir.mkdir(parents=True)
    rows = []
    for index in range(40):
        path = video_dir / f"PRIVATE-{index}.mov"
        path.write_bytes(f"synthetic placeholder {index}; not media".encode())
        target = int(index < 20)
        rows.append({"video_id": f"PRIVATE-{index}", "path": str(path.relative_to(tmp_path)),
                     "sha256": probe.digest(path), "target": target, "timestamp_sec": 3 if target else np.nan,
                     "duration_sec": 12.})
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    protocol_path = tmp_path / "protocol.json"
    protocol = video_eval.create_or_load_protocol(inventory, protocol_path)
    provenance = {"probe_source_sha256": "synthetic-source", "encoder": {"source_sha256": "synthetic-encoder"},
                  "feature_config": probe.engine.FEATURE_CONFIG}
    monkeypatch.setattr(probe, "_provenance", lambda *args: deepcopy(provenance))
    return tmp_path, protocol_path, protocol, provenance


def extracted(duration=12., drift=False):
    requested = [index / 5 for index in range(int(duration * 5))]
    sample = {"requested_times_sec": requested, "actual_times_sec": requested,
              "source_frame_indices": [index * 6 for index in range(len(requested))],
              "source_frame_sha256": [hashlib.sha256(f"synthetic-frame-{index}".encode()).hexdigest()
                                      for index in range(len(requested))],
              "source_shapes": [[848, 464, 3]] * len(requested), "timestamp_sources": ["decoder"] * len(requested),
              "feature_config": probe.engine.FEATURE_CONFIG}
    windows = []
    for interval in probe.engine.window_intervals(duration):
        indices = [index for index, timestamp in enumerate(requested)
                   if interval["start_sec"] <= timestamp < interval["end_sec"]]
        subset = {key: [sample[key][index] for index in indices] for key in probe._SAMPLE_FIELDS}
        subset["feature_config"] = probe.engine.FEATURE_CONFIG
        features = np.zeros(600, np.float32)
        features[112] = interval["start_sec"] + int(drift)
        windows.append({**interval, "timestamp_sec": (subset["actual_times_sec"][0] + subset["actual_times_sec"][-1]) / 2,
                        "features": features, "sampling": subset})
    return {"windows": windows, "duration_sec": duration, "sampling": sample,
            "extraction_sec": .01, "encoder_provenance": {"source_sha256": "synthetic-encoder"}}


def fake_engine(monkeypatch, *, drift=False, network=False):
    calls, encoders = [], []

    class Encoder:
        def __init__(self):
            encoders.append(self)

    def extract(path, deadline, encoder):
        calls.append((path, encoder))
        if network:
            socket.create_connection(("forbidden.invalid", 443))
        return extracted(drift=drift and len(calls) == 9)

    monkeypatch.setattr(probe.engine, "MoViNetEncoder", Encoder)
    monkeypatch.setattr(probe.engine, "extract_video", extract)
    return calls, encoders


def test_deterministic_seed_hash_selection_excludes_blind_and_holdout(case):
    _, _, protocol, _ = case
    selected = probe.select_groups(protocol)
    expected = sorted(protocol["folds"][0]["train_group_ids"],
                      key=lambda group: hashlib.sha256(f"42:{group}".encode()).hexdigest())[:8]
    assert [row["group_id"] for row in selected] == expected
    assert not set(expected) & set(protocol["holdout_group_ids"] + protocol["folds"][0]["valid_group_ids"])
    assert all(not set(row) & {"target", "timestamp_sec", "timestamp_annotations"} for row in selected)
    changed = deepcopy(protocol)
    for group in changed["groups"]:
        if group["target"]:
            group["timestamp_sec"] = 4.
            for annotation in group["timestamp_annotations"]:
                annotation["timestamp_sec"] = 4.
    changed["fingerprint"] = video_eval._fingerprint(changed)
    assert probe.select_groups(changed) == selected


def test_corrupt_fold_cannot_add_holdout(case):
    protocol = deepcopy(case[2])
    protocol["folds"][0]["train_group_ids"][0] = protocol["holdout_group_ids"][0]
    protocol["fingerprint"] = video_eval._fingerprint(protocol)
    with pytest.raises(ValueError):
        probe.select_groups(protocol)


def test_stable_600_way_softmax_and_one_based_tie_break():
    values = np.zeros(600)
    values[112] = 10000
    scores, ranks = probe.probabilities_and_ranks(values)
    assert np.isfinite(scores).all() and scores.sum() == 1 and scores[112] == 1 and ranks[112] == 1
    uniform, ties = probe.probabilities_and_ranks(np.full(600, 10000.))
    assert np.allclose(uniform, 1 / 600) and ties[112] == 113


@pytest.mark.parametrize("bad", [np.zeros(599), np.full(600, np.nan), np.full(600, np.inf)])
def test_invalid_logits_fail_closed(bad):
    with pytest.raises(probe.ProbeError, match="600 finite"):
        probe.probabilities_and_ranks(bad)


def test_prepare_freezes_manifest_without_decoder_or_model(case, monkeypatch):
    root, protocol_path, _, _ = case
    monkeypatch.setattr(probe.engine, "MoViNetEncoder", lambda: pytest.fail("Prepare must not load model"))
    monkeypatch.setattr(probe.engine, "extract_video", lambda *a, **k: pytest.fail("Prepare must not decode"))
    output = probe.config.DATA_PROC / "probe"
    _, _, manifest, manifest_sha = probe.prepare(output, protocol_path)
    original = (output / "manifest.json").read_bytes()
    assert manifest_sha == hashlib.sha256(original).hexdigest() and len(manifest["selections"]) == 8
    assert probe.prepare(output, protocol_path)[3] == manifest_sha
    assert (output / "manifest.json").read_bytes() == original
    assert os.stat(output).st_mode & 0o777 == 0o700
    assert all(os.stat(path).st_mode & 0o777 == 0o600 for path in output.iterdir())


def test_changed_source_rejected_before_decode(case, monkeypatch):
    root, protocol_path, protocol, _ = case
    selection = probe.select_groups(protocol)[0]
    (root / selection["path"]).write_bytes(b"changed synthetic source")
    monkeypatch.setattr(probe.engine, "extract_video", lambda *a, **k: pytest.fail("Changed source decoded"))
    with pytest.raises(probe.ProbeError, match="checksum changed"):
        probe.prepare(probe.config.DATA_PROC / "probe", protocol_path)


def test_source_path_outside_extracted_tree_is_rejected(case):
    root, _, protocol, _ = case
    selection = probe.select_groups(protocol)[0]
    changed = {**selection, "path": "../outside.mov"}
    with pytest.raises(probe.ProbeError, match="safe repository"):
        probe.checked_source(changed)


def test_full_fake_run_private_files_exact_fresh_repeat_and_no_stdout_predictions(case, monkeypatch, capsys):
    _, protocol_path, protocol, _ = case
    calls, encoders = fake_engine(monkeypatch)
    output = probe.config.DATA_PROC / "probe"
    assert probe.main(["--stage", "run", "--protocol", str(protocol_path), "--run-dir", str(output)]) == 0
    public = capsys.readouterr().out
    summary = json.loads((output / "summary.json").read_text())
    records = json.loads((output / "records.json").read_text())
    assert summary["status"] == "completed" and summary["repeatability_passed"]
    assert summary["completed_recordings"] == 8 and len(calls) == 9 and len(encoders) == 2
    assert all(encoder is encoders[0] for _, encoder in calls[:8]) and calls[-1][1] is encoders[1]
    assert summary["records_sha256"] == probe.digest(output / "records.json")
    assert records["manifest_sha256"] == probe.digest(output / "manifest.json")
    assert summary["action_accuracy"] is None and summary["verified_action_labels"] == 0
    assert not summary["automatic_promotion"] and summary["blind_pool_decoded"] == 0
    assert summary["reserved_recordings_untouched"] == 8 and summary["windows_completed"] == 24
    assert set(summary["class_statistics"]) == set(probe.CLASS_INDICES)
    assert not any(group in public or group in json.dumps(summary) for group in protocol["development_group_ids"])
    assert "PRIVATE-" not in public and "predictions" not in public and "counting money" not in public
    assert all(os.stat(path).st_mode & 0o777 == 0o600 for path in output.iterdir())
    assert probe.main(["--stage", "run", "--protocol", str(protocol_path), "--run-dir", str(output)]) == 1
    assert len(calls) == 9


def test_failed_repeat_retains_private_partial_records_without_complete_marker(case, monkeypatch):
    _, protocol_path, _, _ = case
    fake_engine(monkeypatch, drift=True)
    output = probe.config.DATA_PROC / "probe"
    summary = probe.run_probe(output, protocol_path)
    assert summary["status"] == "failed" and not summary["repeatability_passed"]
    assert summary["repeat_logits_equal"] is False and summary["completed_recordings"] == 8
    assert (output / "repeat_record.json").is_file()
    assert json.loads((output / "records.json").read_text())["status"] == "failed"


def test_python_connection_attempt_fails_and_hooks_restore(case, monkeypatch):
    _, protocol_path, _, _ = case
    fake_engine(monkeypatch, network=True)
    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    summary = probe.run_probe(probe.config.DATA_PROC / "probe", protocol_path)
    assert summary["status"] == "failed" and summary["python_connection_attempts"] == 1
    assert summary["completed_recordings"] == 0
    assert originals == (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)


def test_midrun_code_provenance_change_stops_next_extraction(case, monkeypatch):
    _, protocol_path, _, provenance = case
    calls, _ = fake_engine(monkeypatch)
    original_extract = probe.engine.extract_video

    def extract(*args, **kwargs):
        value = original_extract(*args, **kwargs)
        provenance["probe_source_sha256"] = "changed"
        return value

    monkeypatch.setattr(probe.engine, "extract_video", extract)
    summary = probe.run_probe(probe.config.DATA_PROC / "probe", protocol_path)
    assert summary["status"] == "failed" and len(calls) == 1 and summary["completed_recordings"] == 1
