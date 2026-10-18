"""Mocked replay and public synthetic observations only; no footage/model calls."""
import copy
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pytest

from scripts import check_spatial_replay as R
from src import config


class SyntheticClassifier:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, values):
        probability = .1 + .8 * values[:, 0]
        return np.column_stack([1 - probability, probability])


def observation_record(state="none"):
    observation = {field: states[0] for field, states in R.engine.FIELDS.items()}
    observation["people_count"] = state
    vector, _ = R.engine.observation_features(observation)
    audit, windows = [], []
    for index, timestamp in enumerate([1., 3., 5., 7.]):
        windows.append({"timestamp_sec": timestamp, "features": vector.copy()})
        audit.append({"requested_sec": timestamp, "actual_sec": timestamp,
                      "source_frame_index": int(timestamp * 4), "timestamp_source": "decoder",
                      "source_frame_sha256": hashlib.sha256(f"synthetic-{index}".encode()).hexdigest(),
                      "source_shape": [64, 96, 3], "views": [{"role": str(view), "shape": [64, 64, 3],
                          "crop_rect_xyxy": [0, 0, 64, 64], "sha256": hashlib.sha256(
                              f"synthetic-view-{view}-{index}".encode()).hexdigest()} for view in range(3)],
                      "observation": dict(observation)})
    matrix = np.stack([window["features"] for window in windows])
    summaries = {"max": matrix.max(axis=0), "mean": matrix.mean(axis=0), "std": matrix.std(axis=0)}
    return {"clip_features": np.concatenate([summaries[key] for key in R.engine.CLIP_STATISTICS]),
            "windows": windows, "observation_audit": audit, "duration_sec": 8.,
            "sampling": {"feature_config": R.engine.FEATURE_CONFIG, "source_fps": 4.}}


@pytest.fixture
def replay_context(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_PROC", tmp_path / "processed")
    run = config.DATA_PROC / "frozen-run"
    (run / "models").mkdir(parents=True)
    (run / "models/baseline.joblib").write_bytes(b"synthetic classifier bytes")
    failure = run / "inference_replay_check.json"
    failure.write_bytes(b"original failed report; must remain unchanged")
    cached = observation_record()
    provenance = {"script_sha256": R.digest(R.__file__), **R.engine.get_encoder_provenance(),
                  "classifier_sha256": R.digest(run / "models/baseline.joblib"),
                  "runtime_asset_fingerprint": "public-runtime-fingerprint",
                  "protocol_fingerprint": "public-protocol-fingerprint"}
    model = {"clip_model": SyntheticClassifier(), "window_model": SyntheticClassifier()}
    prepared = {"source": tmp_path / "PRIVATE-CLIP.mov", "chosen": {"group_id": "PRIVATE-ID"},
                "cached": cached, "model": model, "cache_sha256": "PRIVATE-CACHE-HASH",
                "original_failure": failure, "original_failure_sha256": R.digest(failure),
                "public_provenance": provenance}
    monkeypatch.setattr(R, "_prepare", lambda *args: prepared)
    probe = {"observation_audit": copy.deepcopy(cached["observation_audit"]),
             "request_audit": [{"body_sha256": "synthetic-body", "body_bytes": 1}] * 4}
    monkeypatch.setattr(R, "_decode_probe", lambda *args: copy.deepcopy(probe))
    captures = []

    @contextmanager
    def capture(**kwargs):
        rows = []
        captures.append(rows)
        yield rows

    monkeypatch.setattr(R, "_capture_requests", capture)
    calls = []
    variants = [observation_record(), observation_record()]

    def extract(source, deadline):
        calls.append(source)
        captures[-1].extend(copy.deepcopy(probe["request_audit"]))
        return copy.deepcopy(variants[len(calls) - 1])

    monkeypatch.setattr(R.engine, "extract_video", extract)
    return run, prepared, probe, calls, variants


def test_default_inspect_neither_decodes_nor_calls_inference_or_writes(replay_context, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Inspection must not decode or infer")

    monkeypatch.setattr(R, "_decode_probe", forbidden)
    monkeypatch.setattr(R.engine, "extract_video", forbidden)
    run = replay_context[0]
    result = R.diagnose(run)
    assert result["mode"] == "inspect"
    assert result["inference_calls_completed"] == 0
    assert not (run / "replay-diagnostics").exists()


def test_two_exact_sequential_replays_preserve_failure_and_private_permissions(replay_context):
    run, prepared, _, calls, _ = replay_context
    output = run / "diagnostic"
    original = prepared["original_failure"].read_bytes()
    result = R.diagnose(run, mode="run", output_dir=output)
    assert len(calls) == 2
    assert result["numeric_replay_reproducible"] is True
    assert result["inference_calls_completed"] == 2
    assert result["original_failed_report_preserved"] is True
    assert result["request_bodies_equal_between_replays"] is True
    assert result["request_bodies_equal_to_serialization_probe"] is True
    assert prepared["original_failure"].read_bytes() == original
    assert json.loads((output / "replay_diagnostic.json").read_text()) == result
    assert (os.stat(output).st_mode & 0o777) == 0o700
    assert all((os.stat(path).st_mode & 0o777) == 0o600 for path in output.iterdir())
    assert joblib.load(output / "replay_1.joblib")["clip_features"].shape == (84,)
    text = json.dumps(result)
    assert all(private not in text for private in ("PRIVATE-ID", "PRIVATE-CLIP", "PRIVATE-CACHE-HASH"))
    assert "probability" not in text


def test_sample_only_does_not_call_vlm(replay_context):
    run, _, _, calls, _ = replay_context
    result = R.diagnose(run, mode="sample-only", output_dir=run / "sample-check")
    assert calls == []
    assert result["sampling_identity_equal"] is True
    assert result["diagnostic_complete"] is True


def test_sampling_mismatch_stops_before_any_vlm_call(replay_context):
    run, _, probe, calls, _ = replay_context
    probe["observation_audit"][0]["source_frame_sha256"] = "changed synthetic frame"
    result = R.diagnose(run, mode="run", output_dir=run / "mismatch")
    assert calls == []
    assert result["sampling_identity_equal"] is False
    assert result["diagnostic_complete"] is False


def test_replay_sampling_mismatch_stops_before_second_call(replay_context):
    run, _, _, calls, variants = replay_context
    variants[0]["observation_audit"][0]["source_frame_sha256"] = "changed synthetic frame"
    result = R.diagnose(run, mode="run", output_dir=run / "mismatch")
    assert len(calls) == 1
    assert result["diagnostic_complete"] is False


def test_feature_changes_and_threshold_flips_are_counted_without_exposing_values(replay_context):
    run, _, _, calls, variants = replay_context
    variants[:] = [observation_record("one"), observation_record("one")]
    result = R.diagnose(run, mode="run", output_dir=run / "different")
    assert len(calls) == 2
    assert result["numeric_replay_reproducible"] is False
    assert result["cached_vs_replay_1"]["differing_clip_features"] > 0
    assert result["cached_vs_replay_1"]["classification_threshold_flips"] == 1
    assert result["replay_1_vs_replay_2"]["differing_clip_features"] == 0
    assert result["promotion_authorized"] is False
    assert result["semantic_accuracy_verified"] is False


def test_existing_diagnostic_directory_is_not_overwritten(replay_context):
    run = replay_context[0]
    output = run / "existing"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        R.diagnose(run, mode="run", output_dir=output)
    assert marker.read_text() == "original"
    assert replay_context[3] == []


def test_inference_failure_produces_sanitized_diagnostic_and_stops(replay_context, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE MODEL DESCRIPTION")

    monkeypatch.setattr(R.engine, "extract_video", fail)
    run = replay_context[0]
    output = run / "failed"
    result = R.diagnose(run, mode="run", output_dir=output)
    assert result["diagnostic_complete"] is False
    assert result["inference_calls_completed"] == 0
    assert "PRIVATE MODEL DESCRIPTION" not in json.dumps(result)
    assert (output / "replay_diagnostic.json").is_file()


def test_cli_exception_sanitizes_private_details(replay_context, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise ValueError("PRIVATE-ID and PRIVATE-PROBABILITIES")

    monkeypatch.setattr(R, "diagnose", fail)
    with pytest.raises(SystemExit) as exc:
        R.main([])
    assert "PRIVATE-ID" not in str(exc.value)
    assert capsys.readouterr().out == ""


def test_identity_checks_view_hash_and_source_position_independently():
    first, second = observation_record(), observation_record()
    second["observation_audit"][1]["source_frame_index"] += 1
    second["observation_audit"][2]["views"][0]["sha256"] = "changed synthetic JPEG"
    differences = R.identity_difference(first, second)
    assert differences["differing_sample_source_frame_index"] == 1
    assert differences["differing_sample_views"] == 1


def test_frozen_request_body_capture_never_changes_serialization_or_adds_network(monkeypatch):
    client = R.engine._observations
    observation = observation_record()["observation_audit"][0]["observation"]
    images = [b"\xff\xd8synthetic JPEG bytes"]

    def forbidden_builder(*args):
        pytest.fail("Serialization probe must not construct an actual network transport")

    monkeypatch.setattr(client, "build_opener", forbidden_builder)
    with R._capture_requests(observations=[observation]) as probe:
        assert client.request_observations(images) == observation
    forwarded = []

    class Delegate:
        def open(self, request, *args, **kwargs):
            forwarded.append(request.data)
            return io.BytesIO(json.dumps({"choices": [{"message": {
                "content": json.dumps(observation)}}]}).encode())

    monkeypatch.setattr(client, "build_opener", lambda *handlers: Delegate())
    with R._capture_requests() as actual:
        assert client.request_observations(images) == observation
    assert actual == probe
    assert len(forwarded) == 1
    assert actual[0]["body_sha256"] == hashlib.sha256(forwarded[0]).hexdigest()
