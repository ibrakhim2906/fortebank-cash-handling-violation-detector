"""Synthetic-only spatial engine tests; every local VLM request is mocked."""
import base64
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

from src import video_qwen_spatial_v3 as V


def observation(number=0, visibility="clear"):
    result = {field: states[number % len(states)] for field, states in V.FIELDS.items()}
    result["visibility"] = visibility
    return result


@pytest.fixture
def local_model(monkeypatch):
    value = {"weights_sha256": V.WEIGHTS_SHA256,
             "projector_sha256": V.FEATURE_CONFIG["projector_sha256"],
             "model_revision": V.FEATURE_CONFIG["model_revision"],
             "runtime_release": V.FEATURE_CONFIG["runtime_release"]}
    monkeypatch.setattr(V._provision, "verify_ready", lambda: dict(value))
    return value


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "synthetic.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 320))
    if not writer.isOpened():
        pytest.skip("No local synthetic video encoder")
    try:
        for index in range(40):
            frame = np.full((320, 480, 3), (index * 3, 70, 120), dtype=np.uint8)
            cv2.rectangle(frame, (index * 4, 30), (index * 4 + 40, 70), (120, 20, 180), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


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


def test_fixed_recipe_and_provenance_do_not_depend_on_targets():
    assert V.ENGINE == "qwen-spatial-v3"
    assert len(V.WINDOW_FEATURE_NAMES) == 28 and len(V.CLIP_FEATURE_NAMES) == 84
    assert V.FEATURE_CONFIG["window_centers_fraction"] == [.125, .375, .625, .875]
    assert V.uniform_windows(80) == [{"start_sec": 8, "end_sec": 12}, {"start_sec": 28, "end_sec": 32},
                                    {"start_sec": 48, "end_sec": 52}, {"start_sec": 68, "end_sec": 72}]
    assert V.FEATURE_CONFIG["metadata_features"] is False
    assert set(V.INHERITED_SOURCES) == {
        "src/video_analysis.py", "src/video_qwen_actions.py", "src/video_action_proposals.py",
        "src/video_actions.py", "scripts/provision_action_model.py", "src/video_local_observations.py"}
    for relative, expected in V.INHERITED_SOURCES.items():
        assert hashlib.sha256((V.config.ROOT / relative).read_bytes()).hexdigest() == expected
    assert V.get_encoder_provenance()["source_sha256"] == hashlib.sha256(Path(V.__file__).read_bytes()).hexdigest()


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), float("inf"), 301, "5"])
def test_invalid_duration_fails(duration):
    with pytest.raises(V.VideoAnalysisError):
        V.uniform_windows(duration)


def test_one_hot_representation_is_strict_and_has_no_missing_fills():
    values, visibility = V.observation_features(observation())
    assert values.shape == (28,) and visibility == "clear"
    assert values.sum() == len(V.FIELDS)
    offset = 0
    for field, states in V.FIELDS.items():
        assert values[offset:offset + len(states)].tolist() == [1] + [0] * (len(states) - 1)
        offset += len(states)
    bad = observation()
    del bad["work_surface"]
    for value in (bad, {**observation(), "rule": "unsupported"}, {**observation(), "people_count": "four"},
                  {**observation(), "work_surface": None}, "untrusted model text"):
        with pytest.raises(V.VideoAnalysisError, match="invalid structured"):
            V.observation_features(value)


@pytest.mark.parametrize("shape,rectangles", [
    ((848, 464, 3), [[0, 0, 464, 848], [0, 0, 464, 464], [0, 384, 464, 848]]),
    ((320, 480, 3), [[0, 0, 480, 320], [0, 0, 320, 320], [160, 0, 480, 320]]),
    ((100, 100, 3), [[0, 0, 100, 100]] * 3),
])
def test_same_instant_views_cover_fixed_geometry_without_upsampling(shape, rectangles):
    frame = np.full(shape, 90, np.uint8)
    images, metadata = V.same_instant_views(frame)
    assert len(images) == len(metadata) == 3
    assert [item["crop_rect_xyxy"] for item in metadata] == rectangles
    for payload, item, (left, top, right, bottom) in zip(images, metadata, rectangles):
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        assert list(decoded.shape) == item["shape"]
        assert decoded.shape[0] <= bottom - top and decoded.shape[1] <= right - left
        assert max(decoded.shape[:2]) <= 640
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()


def test_real_synthetic_decode_records_aligned_samples_and_private_provenance(video, local_model, monkeypatch):
    calls = []

    def request(images, timeout):
        assert len(images) == 3 and 0 < timeout <= 45
        calls.append(images)
        return observation(len(calls))

    monkeypatch.setattr(V._observations, "request_observations", request)
    result = V.extract_video(video, time.monotonic() + 15)
    assert len(calls) == 4 and result["clip_features"].shape == (84,)
    assert result["sampling"]["sampled_frames"] == 4
    assert result["sampling"]["sampled_views"] == 12
    matrix = np.stack([window["features"] for window in result["windows"]])
    np.testing.assert_array_equal(result["clip_features"], np.concatenate([
        matrix.max(axis=0), matrix.mean(axis=0), matrix.std(axis=0)]))
    previous_time, previous_index = -1, -1
    for index, entry in enumerate(result["observation_audit"]):
        assert entry["requested_sec"] == [.5, 1.5, 2.5, 3.5][index]
        assert abs(entry["actual_sec"] - entry["requested_sec"]) <= .25
        assert entry["actual_sec"] > previous_time and entry["source_frame_index"] > previous_index
        assert result["windows"][index]["timestamp_sec"] == entry["actual_sec"]
        assert entry["observation"] == observation(index + 1)
        assert entry["source_shape"] == [320, 480, 3]
        assert len(entry["source_frame_sha256"]) == 64
        assert [item["sha256"] for item in entry["views"]] == [hashlib.sha256(image).hexdigest() for image in calls[index]]
        previous_time, previous_index = entry["actual_sec"], entry["source_frame_index"]


class FakeCapture:
    def __init__(self, wrong_time=False):
        self.index = 0
        self.wrong_time = wrong_time
        self.restarts = 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.index + 1
        if prop == cv2.CAP_PROP_POS_MSEC:
            return 9000 if self.wrong_time else self.index * 100
        return 0

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES and value == 0
        self.index = -1
        self.restarts += 1
        return True

    def grab(self):
        self.index += 1
        return self.index < 40

    def retrieve(self):
        return True, np.full((100, 80, 3), self.index, np.uint8)


def test_nominally_successful_misaligned_seek_recovers_sequentially(monkeypatch):
    cap = FakeCapture()
    monkeypatch.setattr(V, "_read_frame_at", lambda *args: np.zeros((100, 80, 3), np.uint8))
    _, entry = V._sample_center(cap, 2., 10., 40, 4., time.monotonic() + 5)
    assert cap.restarts == 1 and entry["source_frame_index"] == 20
    assert entry["actual_sec"] == 2. and entry["decoding_mode"] == "sequential_alignment_recovery"


def test_permanent_time_misalignment_is_an_explicit_failure(monkeypatch):
    cap = FakeCapture(wrong_time=True)
    monkeypatch.setattr(V, "_read_frame_at", lambda *args: np.zeros((100, 80, 3), np.uint8))
    with pytest.raises(V.VideoAnalysisError, match="not aligned"):
        V._sample_center(cap, 2., 10., 40, 4., time.monotonic() + 5)
    assert cap.restarts == 1


def test_missing_decoder_time_uses_the_recorded_frame_index():
    cap = FakeCapture()
    cap.index = 20
    original_get = cap.get
    cap.get = lambda prop: float("nan") if prop == cv2.CAP_PROP_POS_MSEC else original_get(prop)
    assert V._decoded_position(cap, 2., 10., 40, 4.) == (2., 20, "frame_index_over_fps")


def test_four_distinct_positions_required_before_second_request(video, local_model, monkeypatch):
    calls = []
    monkeypatch.setattr(V, "_sample_center", lambda *args: ([b"fake"] * 3,
                        {"actual_sec": .5, "source_frame_index": 5}))
    monkeypatch.setattr(V._observations, "request_observations", lambda *args, **kwargs: calls.append(1) or observation())
    with pytest.raises(V.VideoAnalysisError, match="strictly increasing"):
        V.extract_video(video)
    assert len(calls) == 1


def test_failures_cannot_become_zero_filled_or_negative_results(video, local_model, monkeypatch):
    def request(*args, **kwargs):
        raise RuntimeError("private synthetic details")

    monkeypatch.setattr(V._observations, "request_observations", request)
    with pytest.raises(V.VideoAnalysisError, match="encoder request failed") as failure:
        V.extract_video(video)
    assert "private synthetic details" not in str(failure.value)
    monkeypatch.setattr(V._observations, "request_observations", lambda *args, **kwargs: observation(visibility="unobservable"))
    with pytest.raises(V.VideoAnalysisError, match="visually unobservable"):
        V.extract_video(video)


def test_model_verification_and_deadline_happen_before_decode(monkeypatch):
    monkeypatch.setattr(V, "_open_video", lambda *args: pytest.fail("Must verify model before decode"))
    monkeypatch.setattr(V._provision, "verify_ready", lambda: (_ for _ in ()).throw(ValueError("missing")))
    with pytest.raises(V.VideoAnalysisError, match="not ready"):
        V.extract_video("synthetic.mov")
    monkeypatch.setattr(V._provision, "verify_ready", lambda: pytest.fail("Expired job must not verify assets"))
    with pytest.raises(TimeoutError):
        V.extract_video("synthetic.mov", time.monotonic() - 1)


def test_file_alias_never_changes_requests_or_features(video, local_model, monkeypatch, tmp_path):
    alias = tmp_path / "positive_target_timestamp_99.mov"
    shutil.copyfile(video, alias)
    calls = []

    def request(images, **kwargs):
        calls.append([hashlib.sha256(image).hexdigest() for image in images])
        return observation()

    monkeypatch.setattr(V._observations, "request_observations", request)
    first, second = V.extract_video(video), V.extract_video(alias)
    assert calls[:4] == calls[4:]
    np.testing.assert_array_equal(first["clip_features"], second["clip_features"])


@pytest.mark.parametrize("changes", [
    {"engine": "qwen-actions"}, {"schema_version": 9}, {"threshold": True}, {"threshold": .4},
    {"clip_model": None}, {"window_model": None}, {"feature_names": {}}, {"feature_config": {}},
    {"source_sha256": "wrong"}, {"weights_sha256": "wrong"}, {"inherited_sources": {}},
])
def test_bad_artifact_is_rejected_before_decoding(tmp_path, monkeypatch, changes):
    monkeypatch.setattr(V, "extract_video", lambda *args: pytest.fail("Bad artifacts cannot decode media"))
    with pytest.raises(V.VideoAnalysisError):
        V.analyze_video("synthetic.mov", artifact(tmp_path, **changes))


def test_public_results_have_source_jpeg_and_no_private_observation_records(video, local_model, monkeypatch, tmp_path):
    monkeypatch.setattr(V._observations, "request_observations", lambda *args, **kwargs: observation())
    positive = V.analyze_video(video, artifact(tmp_path))
    assert set(positive) == {"verdict", "violations"}
    event = positive["violations"][0]
    assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert "не установлены" in event["description"] and "неподтверждённые" in event["description"]
    jpeg = base64.b64decode(event["evidence_frames"][0]["image_base64"], validate=True)
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (320, 480, 3)
    assert V.analyze_video(video, artifact(tmp_path, verdict=0)) == {"verdict": "no_violation", "violations": []}
