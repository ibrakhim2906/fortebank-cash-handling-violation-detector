"""Synthetic cached-feature tests for the single max-window alternative."""
import base64
import hashlib
from pathlib import Path
import time

import cv2
import joblib
import numpy as np
import pytest

from src import video_pretrained as P
from src import video_temporal as V


class WindowClassifier:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, values):
        # First synthetic descriptor stores a known desired probability.
        probabilities = values[:, 0]
        return np.column_stack([1 - probabilities, probabilities])


class ForbiddenClipClassifier:
    def predict_proba(self, values):
        raise AssertionError("Max-window detection must not invoke a clip classifier")


def model_file(tmp_path, **changes):
    artifact = {
        "schema_version": 1, "engine": V.ENGINE, "clip_model": None,
        "window_model": WindowClassifier(), "threshold": 0.5,
        "feature_names": {"clip": V.CLIP_FEATURE_NAMES, "window": V.WINDOW_FEATURE_NAMES},
        "feature_config": dict(V.FEATURE_CONFIG), "weights_sha256": V.WEIGHTS_SHA256,
        "source_sha256": hashlib.sha256(Path(V.__file__).read_bytes()).hexdigest(),
    }
    artifact.update(changes)
    path = tmp_path / "temporal.joblib"
    joblib.dump(artifact, path)
    return path


def extraction(probabilities):
    windows = []
    for index, probability in enumerate(probabilities):
        values = np.zeros(len(V.WINDOW_FEATURE_NAMES), dtype=np.float32)
        values[0] = probability
        windows.append({"features": values, "timestamp_sec": index * 2 + 0.75})
    return {"clip_features": np.zeros(len(V.CLIP_FEATURE_NAMES)), "windows": windows,
            "duration_sec": len(probabilities) * 2,
            "sampling": {"sampled_frames": 4 * len(probabilities), "feature_config": dict(P.FEATURE_CONFIG)}}


def synthetic_video(tmp_path):
    path = tmp_path / "synthetic.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    if not writer.isOpened():
        pytest.skip("No local synthetic video encoder")
    try:
        for index in range(60):
            frame = np.full((64, 96, 3), (index * 3, 30, 80), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_feature_adapter_only_replaces_provenance_and_preserves_frozen_source(monkeypatch):
    expected = extraction([0.2, 0.8])
    original_hash = hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest()
    calls = []

    def delegate(path, deadline):
        calls.append((path, deadline))
        return expected

    monkeypatch.setattr(P, "extract_video", delegate)
    actual = V.extract_video("synthetic.mov", 123)
    assert calls == [("synthetic.mov", 123)]
    assert actual["clip_features"] is expected["clip_features"]
    assert actual["windows"] is expected["windows"]
    assert actual["sampling"]["feature_config"] == V.FEATURE_CONFIG
    assert V.FEATURE_CONFIG["pretrained_source_sha256"] == original_hash
    assert hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest() == original_hash
    assert V.CLIP_FEATURE_NAMES is P.CLIP_FEATURE_NAMES
    assert V.WINDOW_FEATURE_NAMES is P.WINDOW_FEATURE_NAMES


def test_maximum_window_controls_verdict_and_actual_jpeg_timestamp(tmp_path, monkeypatch):
    video = synthetic_video(tmp_path)
    monkeypatch.setattr(V, "extract_video", lambda *args: extraction([0.1, 0.9, 0.2]))
    result = V.analyze_video(video, model_file(tmp_path, clip_model=ForbiddenClipClassifier()))
    assert result["verdict"] == "violation"
    event = result["violations"][0]
    assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert event["timecode_sec"] == pytest.approx(2.7, abs=0.11)
    assert "MobileNetV2" in event["description"]
    assert "не установлены" in event["description"]
    assert "юридическое" in event["description"]
    jpeg = base64.b64decode(event["evidence_frames"][0]["image_base64"], validate=True)
    assert jpeg.startswith(b"\xff\xd8")
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (64, 96, 3)


def test_all_window_scores_below_threshold_produce_no_events(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "extract_video", lambda *args: extraction([0.49, 0.01, 0.3]))
    assert V.analyze_video("synthetic.mov", model_file(tmp_path)) == {
        "verdict": "no_violation", "violations": []}


def test_short_event_is_not_diluted_by_many_negative_windows(tmp_path, monkeypatch):
    probabilities = [0.1] * 50
    probabilities[23] = 0.8
    monkeypatch.setattr(V, "extract_video", lambda *args: extraction(probabilities))
    seen = []

    def evidence(path, timestamps, deadline):
        seen.extend(timestamps)
        return [{"timestamp_sec": timestamps[0], "image_base64": "synthetic-placeholder"}]

    monkeypatch.setattr(V, "evidence_frames", evidence)
    result = V.analyze_video("synthetic.mov", model_file(tmp_path))
    assert result["verdict"] == "violation"
    assert seen == [46.75]


def test_threshold_is_inclusive_and_ties_choose_first_window(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "extract_video", lambda *args: extraction([0.5, 0.5, 0.1]))
    monkeypatch.setattr(V, "evidence_frames", lambda path, times, deadline: [
        {"timestamp_sec": times[0], "image_base64": "synthetic-placeholder"}])
    assert V.analyze_video("synthetic.mov", model_file(tmp_path))["violations"][0]["timecode_sec"] == 0.75


@pytest.mark.parametrize("field,value", [
    ("schema_version", 99), ("engine", "mobilenet"), ("feature_names", {}),
    ("feature_config", {}), ("weights_sha256", "wrong"), ("source_sha256", "stale"),
    ("threshold", 0.7), ("threshold", True), ("window_model", None),
])
def test_schema_and_provenance_mismatches_stop_before_extraction(tmp_path, monkeypatch, field, value):
    def forbidden(*args):
        pytest.fail("Invalid artifact must not extract video")

    monkeypatch.setattr(V, "extract_video", forbidden)
    with pytest.raises(V.VideoAnalysisError):
        V.analyze_video("synthetic.mov", model_file(tmp_path, **{field: value}))


def test_expired_deadline_stops_before_artifact_read():
    with pytest.raises(TimeoutError, match="deadline"):
        V.analyze_video("synthetic.mov", deadline=time.monotonic() - 1)


def test_empty_windows_and_invalid_probabilities_fail(tmp_path, monkeypatch):
    path = model_file(tmp_path)
    monkeypatch.setattr(V, "extract_video", lambda *args: extraction([]))
    with pytest.raises(V.VideoAnalysisError, match="no windows"):
        V.analyze_video("synthetic.mov", path)
    for bad_value in (np.nan, 1.1, -0.1):
        monkeypatch.setattr(V, "extract_video", lambda *args: extraction([bad_value]))
        with pytest.raises(V.VideoAnalysisError, match="invalid probabilities"):
            V.analyze_video("synthetic.mov", path)
