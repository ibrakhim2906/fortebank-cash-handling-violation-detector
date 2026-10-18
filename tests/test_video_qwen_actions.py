"""Synthetic action-encoder checks; no supplied footage or real VLM calls."""
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

from src import video_qwen_actions as V


@pytest.fixture
def local_model(monkeypatch):
    manifest = {"weights_sha256": V.WEIGHTS_SHA256,
                "projector_sha256": V.FEATURE_CONFIG["projector_sha256"],
                "model_revision": V.FEATURE_CONFIG["model_revision"],
                "runtime_release": V.FEATURE_CONFIG["runtime_release"]}
    monkeypatch.setattr(V._provision, "verify_ready", lambda: dict(manifest))
    return manifest


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


def _observation(present=True, visibility="clear"):
    return {"actions": [{"action_id": V.ACTION_IDS[0],
                         "presence": "present" if present else "unknown"}],
            "visibility": visibility}


def _artifact(tmp_path, verdict=1, **changes):
    clip = DummyClassifier(strategy="constant", constant=verdict).fit(
        np.zeros((2, len(V.CLIP_FEATURE_NAMES))), [0, 1])
    window = DummyClassifier(strategy="constant", constant=1).fit(
        np.zeros((2, len(V.WINDOW_FEATURE_NAMES))), [0, 1])
    value = {"schema_version": 1, "engine": V.ENGINE, "clip_model": clip,
             "window_model": window, "threshold": 0.5,
             "feature_names": {"clip": V.CLIP_FEATURE_NAMES, "window": V.WINDOW_FEATURE_NAMES},
             "feature_config": dict(V.FEATURE_CONFIG), "weights_sha256": V.WEIGHTS_SHA256,
             "source_sha256": hashlib.sha256(Path(V.__file__).read_bytes()).hexdigest(),
             "inherited_sources": dict(V.INHERITED_SOURCES), **changes}
    path = tmp_path / "model.joblib"
    joblib.dump(value, path)
    return path


def test_uniform_sampling_is_fixed_without_targets_or_annotation_anchors():
    assert V.uniform_windows(80) == [{"start_sec": 8, "end_sec": 12},
                                     {"start_sec": 28, "end_sec": 32},
                                     {"start_sec": 48, "end_sec": 52},
                                     {"start_sec": 68, "end_sec": 72}]
    assert V.uniform_windows(8) == [{"start_sec": 0, "end_sec": 2},
                                    {"start_sec": 2, "end_sec": 4},
                                    {"start_sec": 4, "end_sec": 6},
                                    {"start_sec": 6, "end_sec": 8}]
    assert V.FEATURE_CONFIG["metadata_features"] is False
    assert not any(name in V.CLIP_FEATURE_NAMES for name in ("filename", "duration", "timestamp", "target"))
    assert len(V.WINDOW_FEATURE_NAMES) == 19 and len(V.CLIP_FEATURE_NAMES) == 57
    assert V.CLIP_STATISTICS == ("max", "mean", "std")


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), 301, True, "10"])
def test_invalid_uniform_sampling_duration_fails(duration):
    with pytest.raises(V.VideoAnalysisError):
        V.uniform_windows(duration)


def test_unknown_or_unlisted_action_is_never_an_absence_label():
    values, visibility = V.observation_features(_observation())
    assert visibility == "clear" and values.shape == (19,)
    assert values[0] == 1 and values[8] == 0
    assert np.all(values[1:8] == 0) and np.all(values[9:16] == 1)
    assert values[16:].tolist() == [1, 0, 0]
    raw = {"actions": [{"action_id": V.ACTION_IDS[0], "presence": "absent"}], "visibility": "partial"}
    values, _ = V.observation_features(raw)
    assert np.all(values[:8] == 0) and np.all(values[8:16] == 1)
    assert values[16:].tolist() == [0, 1, 0]


def test_unobservable_observation_downgrades_every_action_to_unknown():
    values, visibility = V.observation_features(_observation(visibility="unobservable"))
    assert visibility == "unobservable"
    assert np.all(values[:8] == 0) and np.all(values[8:16] == 1)
    assert values[16:].tolist() == [0, 0, 1]


def test_bad_model_text_cannot_be_parsed_into_features():
    for value in ("untrusted free text", {"actions": [], "visibility": "clear", "rule": "invented"},
                  {"actions": [{"action_id": "legal_conclusion", "presence": "present"}], "visibility": "clear"}):
        with pytest.raises(V.VideoAnalysisError, match="invalid structured"):
            V.observation_features(value)


def test_extraction_requires_four_bounded_local_requests_and_fixed_numeric_summaries(video, local_model, monkeypatch):
    calls = []

    def request(images, *, port, timeout):
        assert port == 8003 and 0 < timeout <= 45
        assert len(images) == 3
        for payload in images:
            decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            assert max(decoded.shape[:2]) == 384
        calls.append(images)
        return _observation(present=len(calls) % 2 == 0)

    monkeypatch.setattr(V._proposals, "request_local_vision", request)
    result = V.extract_video(video, time.monotonic() + 20)
    assert len(calls) == V.WINDOW_COUNT == 4
    assert result["clip_features"].shape == (57,)
    assert result["sampling"]["sampled_frames"] == 12
    assert result["sampling"]["window_count"] == 4
    assert result["duration_sec"] == pytest.approx(4)
    matrix = np.stack([window["features"] for window in result["windows"]])
    expected = np.concatenate([matrix.max(axis=0), matrix.mean(axis=0), matrix.std(axis=0)])
    np.testing.assert_array_equal(result["clip_features"], expected)
    assert result["clip_features"][0] == 1
    assert result["clip_features"][19] == 0.5
    assert result["clip_features"][38] == 0.5
    assert [window["timestamp_sec"] for window in result["windows"]] == pytest.approx([0.5, 1.5, 2.5, 3.5], abs=0.11)


def test_model_verified_before_any_media_is_opened(video, monkeypatch):
    def unavailable():
        raise ValueError("missing local asset")

    def forbidden(*args):
        pytest.fail("Unverified model must not open footage")

    monkeypatch.setattr(V._provision, "verify_ready", unavailable)
    monkeypatch.setattr(V, "_open_video", forbidden)
    with pytest.raises(V.VideoAnalysisError, match="not ready"):
        V.extract_video(video)


def test_changed_model_identity_is_rejected_before_media(video, local_model, monkeypatch):
    monkeypatch.setattr(V._provision, "verify_ready", lambda: {**local_model, "projector_sha256": "a" * 64})
    monkeypatch.setattr(V, "_open_video", lambda *args: pytest.fail("Mismatched model must not open footage"))
    with pytest.raises(V.VideoAnalysisError, match="identity differs"):
        V.extract_video(video)


def test_request_failure_and_all_unobservable_are_failures_not_negatives(video, local_model, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic server failure; private contents must not escape")

    monkeypatch.setattr(V._proposals, "request_local_vision", unavailable)
    with pytest.raises(V.VideoAnalysisError, match="encoder request failed") as failure:
        V.extract_video(video)
    assert "private contents" not in str(failure.value)
    monkeypatch.setattr(V._proposals, "request_local_vision", lambda *args, **kwargs: _observation(visibility="unobservable"))
    with pytest.raises(V.VideoAnalysisError, match="visually unobservable"):
        V.extract_video(video)


def test_successful_unknown_is_an_abstention_descriptor_not_operational_failure(video, local_model, monkeypatch):
    monkeypatch.setattr(V._proposals, "request_local_vision", lambda *args, **kwargs: _observation(False, "partial"))
    result = V.extract_video(video)
    assert np.all(result["clip_features"][:8] == 0)
    assert result["sampling"]["unobservable_windows"] == 0


def test_four_valid_calls_required_no_partial_feature_bag(video, local_model, monkeypatch):
    called = []

    def request(*args, **kwargs):
        called.append(1)
        if len(called) == 4:
            raise RuntimeError("last uniform request failed")
        return _observation()

    monkeypatch.setattr(V._proposals, "request_local_vision", request)
    with pytest.raises(V.VideoAnalysisError):
        V.extract_video(video)
    assert len(called) == 4


def test_filename_changes_do_not_enter_the_prompt_or_features(video, local_model, monkeypatch, tmp_path):
    alias = tmp_path / "official_positive_target_event_anchor_123.mov"
    shutil.copyfile(video, alias)
    requests = []

    def request(images, **kwargs):
        requests.append([hashlib.sha256(value).hexdigest() for value in images])
        return _observation()

    monkeypatch.setattr(V._proposals, "request_local_vision", request)
    first = V.extract_video(video)
    second = V.extract_video(alias)
    assert requests[:4] == requests[4:]
    np.testing.assert_array_equal(first["clip_features"], second["clip_features"])


def test_expired_deadline_stops_before_model_verification(monkeypatch):
    monkeypatch.setattr(V._provision, "verify_ready", lambda: pytest.fail("Expired jobs must stop before model reads"))
    with pytest.raises(TimeoutError, match="deadline"):
        V.extract_video("synthetic.mov", time.monotonic() - 1)


def test_remaining_deadline_caps_local_request_timeout(video, local_model, monkeypatch):
    timeouts = []

    def request(images, *, port, timeout):
        timeouts.append(timeout)
        return _observation()

    monkeypatch.setattr(V._proposals, "request_local_vision", request)
    V.extract_video(video, time.monotonic() + 5)
    assert all(0 < timeout < 5 for timeout in timeouts)


def test_declared_inherited_sources_are_exact_and_match_current_snapshots():
    assert set(V.INHERITED_SOURCES) == {"src/video_analysis.py", "src/video_action_proposals.py",
                                        "src/video_actions.py", "scripts/provision_action_model.py"}
    assert V.FEATURE_CONFIG["inherited_sources"] == V.INHERITED_SOURCES
    for relative, expected in V.INHERITED_SOURCES.items():
        assert hashlib.sha256((V.config.ROOT / relative).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("changes", [
    {"schema_version": 99}, {"engine": "mobilenet"}, {"feature_names": {}},
    {"feature_config": {}}, {"weights_sha256": "wrong"},
    {"source_sha256": "wrong"}, {"inherited_sources": {}},
    {"threshold": 0.7}, {"threshold": True}, {"clip_model": None},
    {"window_model": "invalid"},
])
def test_invalid_classifier_schema_fails_before_extraction(tmp_path, monkeypatch, changes):
    monkeypatch.setattr(V, "extract_video", lambda *args: pytest.fail("Invalid artifact must not extract footage"))
    with pytest.raises(V.VideoAnalysisError):
        V.analyze_video("synthetic.mov", _artifact(tmp_path, **changes))


def test_positive_output_uses_actual_source_jpeg_and_honest_generic_description(video, local_model, monkeypatch, tmp_path):
    monkeypatch.setattr(V._proposals, "request_local_vision", lambda *args, **kwargs: _observation())
    result = V.analyze_video(video, _artifact(tmp_path))
    assert result["verdict"] == "violation"
    event = result["violations"][0]
    assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert "Qwen3-VL" in event["description"] and "не установлены" in event["description"]
    assert "подтверждёнными" in event["description"]
    assert 0 <= event["timecode_sec"] <= 4
    jpeg = base64.b64decode(event["evidence_frames"][0]["image_base64"], validate=True)
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (320, 480, 3)


def test_negative_output_and_expired_analysis_remain_exact(video, local_model, monkeypatch, tmp_path):
    monkeypatch.setattr(V._proposals, "request_local_vision", lambda *args, **kwargs: _observation())
    assert V.analyze_video(video, _artifact(tmp_path, verdict=0)) == {"verdict": "no_violation", "violations": []}
    with pytest.raises(TimeoutError, match="deadline"):
        V.analyze_video(video, deadline=time.monotonic() - 1)


def test_decoder_successful_timestamp_read_returns_original_frame_unchanged():
    original = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    calls = []

    class Cap:
        def set(self, prop, value):
            calls.append((prop, value))
            return True

        def read(self):
            return True, original

        def get(self, prop):
            pytest.fail("Successful timestamp reads must not enter fallback")

        def grab(self):
            pytest.fail("Successful timestamp reads must not sequentially decode")

    actual = V._read_frame_at(Cap(), 1.5, 30, time.monotonic() + 5)
    assert actual is original
    assert calls == [(cv2.CAP_PROP_POS_MSEC, 1500.0)]
    np.testing.assert_array_equal(actual, np.arange(36, dtype=np.uint8).reshape(3, 4, 3))


def test_decoder_failed_timestamp_and_frame_index_seek_rewind_and_retrieve_last_frame():
    original = np.full((3, 4, 3), 73, dtype=np.uint8)

    class Cap:
        def __init__(self):
            self.sets, self.reads, self.grabs, self.retrievals = [], 0, 0, 0

        def set(self, prop, value):
            self.sets.append((prop, value))
            return True

        def read(self):
            self.reads += 1
            return False, None

        def get(self, prop):
            assert prop == cv2.CAP_PROP_FRAME_COUNT
            return 5

        def grab(self):
            self.grabs += 1
            return self.grabs <= 5

        def retrieve(self):
            self.retrievals += 1
            assert self.grabs == 5
            return True, original

    cap = Cap()
    assert V._read_frame_at(cap, 2.4, 2, time.monotonic() + 5) is original
    assert cap.sets == [(cv2.CAP_PROP_POS_MSEC, 2400.0),
                        (cv2.CAP_PROP_POS_FRAMES, 4), (cv2.CAP_PROP_POS_FRAMES, 0)]
    assert (cap.reads, cap.grabs, cap.retrievals) == (2, 5, 1)


def test_decoder_sequential_early_eof_raises_instead_of_fabricating_a_sample():
    class Cap:
        def __init__(self):
            self.grabs = 0

        def set(self, prop, value):
            return True

        def read(self):
            return False, None

        def get(self, prop):
            assert prop == cv2.CAP_PROP_FRAME_COUNT
            return 5

        def grab(self):
            self.grabs += 1
            return self.grabs <= 2

        def retrieve(self):
            pytest.fail("Early EOF cannot produce an evidence frame")

    cap = Cap()
    with pytest.raises(V.VideoAnalysisError, match="ended before the requested frame"):
        V._read_frame_at(cap, 2.4, 2, time.monotonic() + 5)
    assert cap.grabs == 3
