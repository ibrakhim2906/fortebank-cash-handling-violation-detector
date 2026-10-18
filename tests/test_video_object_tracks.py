"""Synthetic-only anonymous-track features; no supplied footage or real model."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import time

import cv2
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src import video_object_tracks as V


def detection(class_id=0, box=(.1, .1, .5, .9), score=.8):
    return {"class_id": class_id, "score": score, "box_xyxy": list(box)}


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "synthetic.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    if not writer.isOpened():
        pytest.skip("No local synthetic video encoder")
    try:
        for index in range(40):
            frame = np.full((64, 96, 3), (20 + index, 70, 100), np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def detector(monkeypatch):
    ready = {"weights_sha256": V.WEIGHTS_SHA256, "model_revision": V._assets.SOURCE_COMMIT}
    monkeypatch.setattr(V._assets, "verify_ready", lambda: dict(ready))
    calls = []

    class FakeDetector:
        def detect(self, frame, deadline=None):
            assert deadline is not None and deadline > time.monotonic()
            calls.append(hashlib.sha256(frame.tobytes()).hexdigest())
            offset = (len(calls) - 1) * .005
            return [detection(), detection(67, (.2 + offset, .3, .3 + offset, .5))]

    monkeypatch.setattr(V._objects, "ObjectDetector", FakeDetector)
    return calls


def artifact(tmp_path, verdict=1, **changes):
    clip = DummyClassifier(strategy="constant", constant=verdict).fit(
        np.zeros((2, len(V.CLIP_FEATURE_NAMES))), [0, 1])
    window = DummyClassifier(strategy="constant", constant=1).fit(
        np.zeros((2, len(V.WINDOW_FEATURE_NAMES))), [0, 1])
    value = {"schema_version": 1, **V.get_encoder_provenance(), "clip_model": clip,
             "window_model": window, "threshold": .5,
             "feature_names": {"clip": V.CLIP_FEATURE_NAMES, "window": V.WINDOW_FEATURE_NAMES},
             "feature_config": deepcopy(V.FEATURE_CONFIG), **changes}
    path = tmp_path / "classifier.joblib"
    joblib.dump(value, path)
    return path


def test_frozen_schema_and_sampling():
    assert V.ENGINE == "object-tracks"
    assert V.CLASS_IDS == (0, 24, 26, 28, 63, 67)
    assert len(V.WINDOW_FEATURE_NAMES) == 57 and len(V.CLIP_FEATURE_NAMES) == 171
    assert len(set(V.WINDOW_FEATURE_NAMES)) == 57 and len(set(V.CLIP_FEATURE_NAMES)) == 171
    assert V.FEATURE_CONFIG["metadata_features"] is False
    np.testing.assert_array_equal(V.requested_times(2), [0, .5, 1., 1.5])
    assert len(V.requested_times(150)) == 300
    with pytest.raises(V.VideoAnalysisError, match="300-sample"):
        V.requested_times(150.1)
    assert set(V.INHERITED_SOURCES) == {"src/video_analysis.py", "src/video_objects.py", "scripts/fetch_video_objects.py"}
    for relative, expected in V.INHERITED_SOURCES.items():
        assert hashlib.sha256((V.config.ROOT / relative).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("duration", [0, -1, 301, True, float("nan"), float("inf"), "5"])
def test_invalid_sampling_duration_fails(duration):
    with pytest.raises(V.VideoAnalysisError):
        V.requested_times(duration)


@pytest.mark.parametrize("value", [
    None, [detection()] * 101, [detection(class_id=True)], [detection(class_id=77)],
    [detection(score=float("nan"))], [detection(score=True)],
    [detection(box=(.2, .2, .1, .4))], [detection(box=(-.1, 0, .3, .4))],
    [{**detection(), "identity": "unsupported"}],
])
def test_detector_schema_is_strict(value):
    with pytest.raises(V.VideoAnalysisError):
        V.validate_detections(value)
    assert V.validate_detections([]) == []  # Operationally valid absence of model detections.


def test_tracks_are_same_class_anonymous_and_expire_after_fixed_gap():
    tracker = V.AnonymousTracker()
    first = tracker.update(V.validate_detections([detection(), detection(67)]), 0.)
    second = tracker.update(V.validate_detections([detection(), detection(67)]), .5)
    assert [item["track_id"] for item in first] == [item["track_id"] for item in second]
    third = tracker.update(V.validate_detections([detection(), detection(67)]), 2.)
    assert not {item["track_id"] for item in first} & {item["track_id"] for item in third}
    with pytest.raises(V.VideoAnalysisError, match="increasing"):
        tracker.update([], 2.)
    restarted = V.AnonymousTracker().update(V.validate_detections([detection()]), 0.)
    assert restarted[0]["track_id"] == 0


def test_matching_is_one_to_one_and_class_separated():
    tracker = V.AnonymousTracker()
    tracker.update(V.validate_detections([detection()]), 0.)
    matched = tracker.update(V.validate_detections([detection(), detection(score=.7), detection(67)]), .5)
    assert len({item["track_id"] for item in matched}) == 3
    assert matched[0]["track_id"] == 0
    assert matched[-1]["track_id"] != 0


def test_window_statistics_and_relations_have_explicit_missing_indicators():
    tracker = V.AnonymousTracker()
    rows = []
    for index, raw in enumerate(([detection(), detection(67, (.3, .3, .4, .4))], [detection()])):
        rows.append({"timestamp_sec": index * .5, "detections": tracker.update(V.validate_detections(raw), index * .5)})
    values = dict(zip(V.WINDOW_FEATURE_NAMES, V.window_features(rows)))
    assert values["class_0_mean_count"] == 1 and values["class_67_mean_count"] == .5
    assert values["class_67_max_count"] == 1 and values["class_67_max_score"] == pytest.approx(.8)
    assert values["class_67_mean_area"] == pytest.approx(.01)
    assert values["class_67_person_missing_frame_fraction"] == .5
    assert values["class_24_person_missing_frame_fraction"] == 1
    expected_distance = np.linalg.norm(np.array([.35, .35]) - np.array([.3, .5])) / np.sqrt(2)
    assert values["class_67_person_mean_distance"] == pytest.approx(expected_distance)
    assert values["class_0_track_count"] == 1 and values["class_0_track_mean_visible_span_seconds"] == 1
    assert values["class_0_track_mean_center_speed"] == 0


def test_motion_uses_source_time_and_earlier_window_is_unchanged_by_future_rows():
    tracker, rows = V.AnonymousTracker(), []
    for index in range(4):
        raw = [detection(67, (.2 + .01 * index, .2, .4 + .01 * index, .4))]
        rows.append({"timestamp_sec": index * .5, "detections": tracker.update(V.validate_detections(raw), index * .5)})
    before, _ = V.summarize_windows(rows)
    initial = before[0]["features"].copy()
    for index in range(4, 8):
        raw = [detection(67, (.2 + .01 * index, .2, .4 + .01 * index, .4))]
        rows.append({"timestamp_sec": index * .5, "detections": tracker.update(V.validate_detections(raw), index * .5)})
    after, clip = V.summarize_windows(rows)
    np.testing.assert_array_equal(initial, after[0]["features"])
    values = dict(zip(V.WINDOW_FEATURE_NAMES, initial))
    assert values["class_67_track_mean_center_speed"] == pytest.approx(.01 / np.sqrt(2) / .5)
    assert clip.shape == (171,) and len(after) == 2


def test_sequential_sampling_uses_every_fixed_request_and_keeps_private_source_audit(video, detector):
    result = V.extract_video(video, time.monotonic() + 15)
    assert len(detector) == 8 and result["clip_features"].shape == (171,)
    assert result["sampling"]["decoded_frames"] == 40 and result["sampling"]["sampled_frames"] == 8
    assert len(result["windows"]) == 2
    for index, entry in enumerate(result["object_audit"]):
        assert entry["requested_sec"] == index * .5
        assert entry["timestamp_sec"] == pytest.approx(index * .5, abs=.11)
        assert entry["source_frame_index"] == index * 5
        assert entry["source_frame_sha256"] == detector[index]
        assert entry["source_shape"] == [64, 96, 3]
        assert {item["class_id"] for item in entry["detections"]} == {0, 67}
        assert len({item["track_id"] for item in entry["detections"]}) == 2
    matrix = np.stack([window["features"] for window in result["windows"]])
    np.testing.assert_array_equal(result["clip_features"], np.concatenate([
        matrix.max(axis=0), matrix.mean(axis=0), matrix.std(axis=0)]))


def test_failed_inference_is_not_an_empty_valid_detection(video, detector, monkeypatch):
    class BrokenDetector:
        def detect(self, *args, **kwargs):
            raise V.VideoAnalysisError("synthetic inference failure")

    monkeypatch.setattr(V._objects, "ObjectDetector", BrokenDetector)
    with pytest.raises(V.VideoAnalysisError, match="inference failure"):
        V.extract_video(video)


def test_assets_and_deadline_checked_before_media(monkeypatch):
    monkeypatch.setattr(V, "_open_video", lambda *args: pytest.fail("Must verify assets before media"))
    monkeypatch.setattr(V._assets, "verify_ready", lambda: (_ for _ in ()).throw(ValueError("missing")))
    with pytest.raises(V.VideoAnalysisError, match="not ready"):
        V.extract_video("synthetic.mov")
    monkeypatch.setattr(V._assets, "verify_ready", lambda: pytest.fail("Expired jobs must stop before asset checks"))
    with pytest.raises(TimeoutError):
        V.extract_video("synthetic.mov", time.monotonic() - 1)


def test_filename_has_no_effect_on_input_pixels_or_features(video, detector, monkeypatch, tmp_path):
    alias = tmp_path / "positive_target_event_75.mov"
    shutil.copyfile(video, alias)

    class ConstantDetector:
        def detect(self, frame, **kwargs):
            return [detection()]

    monkeypatch.setattr(V._objects, "ObjectDetector", ConstantDetector)
    first, second = V.extract_video(video), V.extract_video(alias)
    np.testing.assert_array_equal(first["clip_features"], second["clip_features"])
    assert [item["source_frame_sha256"] for item in first["object_audit"]] == [
        item["source_frame_sha256"] for item in second["object_audit"]]


@pytest.mark.parametrize("changes", [
    {"engine": "numeric"}, {"threshold": True}, {"threshold": .4}, {"feature_config": {}},
    {"inherited_sources": {}}, {"source_sha256": "stale"}, {"weights_sha256": "stale"},
    {"clip_model": None}, {"window_model": None}, {"feature_names": {}},
])
def test_bad_artifact_cannot_decode_media(tmp_path, monkeypatch, changes):
    monkeypatch.setattr(V, "extract_video", lambda *args: pytest.fail("Bad model cannot decode media"))
    with pytest.raises(V.VideoAnalysisError):
        V.analyze_video("synthetic.mov", artifact(tmp_path, **changes))


def test_public_api_result_has_no_raw_detections_or_identity_claims(video, detector, tmp_path):
    result = V.analyze_video(video, artifact(tmp_path))
    assert set(result) == {"verdict", "violations"} and result["verdict"] == "violation"
    event = result["violations"][0]
    assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert "не установлены" in event["description"] and "анонимных" in event["description"]
    assert V.analyze_video(video, artifact(tmp_path, verdict=0)) == {"verdict": "no_violation", "violations": []}
