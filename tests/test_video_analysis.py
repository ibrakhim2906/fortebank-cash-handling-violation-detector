"""All video pixels in these tests are synthetic, created locally."""
import base64
import hashlib
from pathlib import Path
import shutil
import time

import cv2
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src import video_analysis as V


def make_video(path, brightness=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    if not writer.isOpened():
        pytest.skip("The local OpenCV build does not provide a synthetic MOV encoder")
    try:
        for index in range(40):
            frame = np.full((64, 96, 3), brightness, dtype=np.uint8)
            x = (index * 2) % 72
            cv2.rectangle(frame, (x, 18), (x + 16, 38), (30, 180, 90), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "synthetic.mov")


def make_model(tmp_path, verdict=1, window_model=True):
    clip = DummyClassifier(strategy="constant", constant=verdict).fit(
        np.zeros((2, len(V.CLIP_FEATURE_NAMES))), [0, 1])
    window = DummyClassifier(strategy="constant", constant=1).fit(
        np.zeros((2, len(V.WINDOW_FEATURE_NAMES))), [0, 1]) if window_model else None
    artifact = {
        "schema_version": 1, "clip_model": clip, "window_model": window,
        "threshold": 0.5,
        "feature_names": {"clip": V.CLIP_FEATURE_NAMES, "window": V.WINDOW_FEATURE_NAMES},
        "feature_config": dict(V.FEATURE_CONFIG),
        "source_sha256": hashlib.sha256(Path(V.__file__).read_bytes()).hexdigest(),
    }
    path = tmp_path / "baseline.joblib"
    joblib.dump(artifact, path)
    return path


def test_numeric_extraction_has_fixed_schema_and_relative_windows(video):
    result = V.extract_video(video, deadline=time.monotonic() + 10)
    assert result["clip_features"].shape == (len(V.CLIP_FEATURE_NAMES),)
    assert len(V.CLIP_FEATURE_NAMES) == 222
    assert len(V.WINDOW_FEATURE_NAMES) == 74
    assert len(set(V.CLIP_FEATURE_NAMES)) == len(V.CLIP_FEATURE_NAMES)
    assert np.isfinite(result["clip_features"]).all()
    assert 3.9 <= result["duration_sec"] <= 4.1
    assert result["sampling"]["decoded_frames"] == 40
    assert 6 <= result["sampling"]["sampled_frames"] <= 8
    assert result["sampling"]["feature_config"] == V.FEATURE_CONFIG
    assert len(result["windows"]) == 2
    for window in result["windows"]:
        assert window["features"].shape == (len(V.WINDOW_FEATURE_NAMES),)
        assert np.isfinite(window["features"]).all()
        assert 0 <= window["timestamp_sec"] < result["duration_sec"]


def test_features_ignore_filename_and_are_repeatable(video, tmp_path):
    alias = tmp_path / "positive_label_123.mov"
    shutil.copyfile(video, alias)
    first = V.extract_video(video)
    second = V.extract_video(alias)
    np.testing.assert_array_equal(first["clip_features"], second["clip_features"])
    for one, two in zip(first["windows"], second["windows"], strict=True):
        assert one["timestamp_sec"] == two["timestamp_sec"]
        np.testing.assert_array_equal(one["features"], two["features"])


def test_positive_result_has_actual_jpeg_and_honest_description(video, tmp_path):
    result = V.analyze_video(video, make_model(tmp_path))
    assert result["verdict"] == "violation"
    assert len(result["violations"]) == 1
    event = result["violations"][0]
    assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert "не установлены" in event["description"]
    assert "юридическое" in event["description"]
    frames = event["evidence_frames"]
    assert len(frames) == 1
    assert 0 <= event["timecode_sec"] < 4
    assert frames[0]["timestamp_sec"] == event["timecode_sec"]
    jpeg = base64.b64decode(frames[0]["image_base64"], validate=True)
    assert jpeg.startswith(b"\xff\xd8")
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (64, 96, 3)


def test_evidence_changes_with_uploaded_pixels(video, tmp_path):
    other = make_video(tmp_path / "other.mov", brightness=220)
    model = make_model(tmp_path)
    first = V.analyze_video(video, model)["violations"][0]["evidence_frames"][0]
    second = V.analyze_video(other, model)["violations"][0]["evidence_frames"][0]
    assert first["image_base64"] != second["image_base64"]


def test_negative_result_has_empty_events(video, tmp_path):
    assert V.analyze_video(video, make_model(tmp_path, verdict=0)) == {
        "verdict": "no_violation", "violations": []}


def test_missing_window_model_uses_bounded_motion_candidate(video, tmp_path):
    result = V.analyze_video(video, make_model(tmp_path, window_model=False))
    assert 0 <= result["violations"][0]["timecode_sec"] < 4


def test_corrupt_or_empty_video_is_failure(tmp_path):
    path = tmp_path / "invalid.mov"
    path.write_bytes(b"not a movie")
    with pytest.raises(V.VideoAnalysisError, match="open"):
        V.extract_video(path)
    path.write_bytes(b"")
    with pytest.raises(V.VideoAnalysisError, match="empty"):
        V.extract_video(path)


def test_expired_deadline_stops_before_decode_or_model(video):
    with pytest.raises(TimeoutError, match="deadline"):
        V.extract_video(video, deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError, match="deadline"):
        V.analyze_video(video, deadline=time.monotonic() - 1)


@pytest.mark.parametrize("field,bad_value", [
    ("schema_version", 99), ("feature_names", {}),
    ("feature_config", {}), ("source_sha256", "stale"), ("threshold", 2),
])
def test_invalid_or_stale_model_is_failure(video, tmp_path, field, bad_value):
    path = make_model(tmp_path)
    artifact = joblib.load(path)
    artifact[field] = bad_value
    joblib.dump(artifact, path)
    with pytest.raises(V.VideoAnalysisError):
        V.analyze_video(video, path)


def test_frame_bound_fails_instead_of_returning_partial_features(video, monkeypatch):
    monkeypatch.setattr(V, "MAX_DECODED_FRAMES", 5)
    with pytest.raises(V.VideoAnalysisError, match="frame safety"):
        V.extract_video(video)
