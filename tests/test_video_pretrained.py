"""Synthetic local-only checks; no downloaded weights or supplied footage used."""
import base64
import hashlib
from pathlib import Path
import shutil
import socket
import time

import cv2
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src import config
from src import video_pretrained as V


class SyntheticNet:
    def __init__(self):
        self.input = None
        self.backend, self.target = None, None

    def setPreferableBackend(self, value):
        self.backend = value

    def setPreferableTarget(self, value):
        self.target = value

    def getLayerNames(self):
        return [V.OUTPUT_LAYER]

    def setInput(self, value):
        self.input = value.copy()

    def forward(self, layer):
        assert layer == V.OUTPUT_TENSOR
        channels = self.input.mean(axis=(0, 2, 3))
        vector = np.linspace(0, 1, V.POOLED_DIM, dtype=np.float32)
        return (vector * (1 + channels[0]) + channels[1] - channels[2]).reshape(1, -1, 1, 1)


@pytest.fixture
def encoder(tmp_path, monkeypatch):
    payload = b"Synthetic ONNX placeholder: unit tests never parse real weights"
    path = tmp_path / "synthetic.onnx"
    path.write_bytes(payload)
    monkeypatch.setenv("DATATHON_VIDEO_WEIGHTS", str(path))
    monkeypatch.setattr(V, "WEIGHTS_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(V, "_ENCODERS", {})
    nets = []

    def load(buffer):
        assert buffer.tobytes() == payload
        net = SyntheticNet()
        nets.append(net)
        return net

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", load)
    return path, nets


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "synthetic.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    if not writer.isOpened():
        pytest.skip("No local synthetic video encoder")
    try:
        for index in range(40):
            frame = np.full((64, 96, 3), (20, 70, 120), dtype=np.uint8)
            cv2.rectangle(frame, (index % 50, 15), (index % 50 + 15, 40), (170, 30, 80), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def model_file(tmp_path, verdict=1, window_model=True):
    clip = DummyClassifier(strategy="constant", constant=verdict).fit(
        np.zeros((2, len(V.CLIP_FEATURE_NAMES))), [0, 1])
    window = (DummyClassifier(strategy="constant", constant=1).fit(
        np.zeros((2, len(V.WINDOW_FEATURE_NAMES))), [0, 1]) if window_model else None)
    artifact = {
        "schema_version": 1, "engine": "mobilenet", "clip_model": clip,
        "window_model": window, "threshold": 0.5,
        "feature_names": {"clip": V.CLIP_FEATURE_NAMES, "window": V.WINDOW_FEATURE_NAMES},
        "feature_config": dict(V.FEATURE_CONFIG), "weights_sha256": V.WEIGHTS_SHA256,
        "source_sha256": hashlib.sha256(Path(V.__file__).read_bytes()).hexdigest(),
    }
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    return path


def test_preprocess_matches_official_rgb_crop_and_normalization():
    frame = np.full((256, 256, 3), (10, 20, 30), dtype=np.uint8)
    frame[:16] = 255
    frame[240:] = 255
    frame[:, :16] = 255
    frame[:, 240:] = 255
    blob = V._preprocess(frame)
    expected = ((np.asarray([30, 20, 10]) / 255
                 - np.asarray([0.485, 0.456, 0.406])) / np.asarray([0.229, 0.224, 0.225]))
    assert blob.shape == (1, 3, 224, 224)
    assert blob.dtype == np.float32 and blob.flags.c_contiguous
    np.testing.assert_allclose(blob[0, :, 0, 0], expected, atol=1e-6)
    np.testing.assert_allclose(blob[0, :, -1, -1], expected, atol=1e-6)


def test_preprocess_rejects_nonvisual_or_wrong_shape():
    with pytest.raises(V.VideoAnalysisError, match="three-channel"):
        V._preprocess(np.zeros((10, 10), dtype=np.uint8))


def test_projection_is_fixed_label_free_and_named_in_configuration(encoder):
    np.testing.assert_array_equal(V._PROJECTION, (
        np.random.default_rng(config.SEED).standard_normal((1280, 128)) / np.sqrt(128)
    ).astype(np.float32))
    vector = V._encode(np.zeros((30, 40, 3), dtype=np.uint8), V._load_encoder())
    assert vector.shape == (128,)
    assert V.FEATURE_CONFIG["projection"]["seed"] == config.SEED
    assert V.FEATURE_CONFIG["projection"]["dimensions"] == [1280, 128]
    assert np.isfinite(vector).all()


def test_encoder_cache_hashes_every_use_and_forces_cpu(encoder):
    path, nets = encoder
    first = V._load_encoder(path)
    assert first is V._load_encoder(path)
    assert len(nets) == 1
    assert nets[0].backend == cv2.dnn.DNN_BACKEND_OPENCV
    assert nets[0].target == cv2.dnn.DNN_TARGET_CPU
    path.write_bytes(b"tampered but previously cached encoder")
    with pytest.raises(V.VideoAnalysisError, match="SHA-256"):
        V._load_encoder(path)
    assert len(nets) == 1


def test_missing_weights_fail_without_downloading(tmp_path, monkeypatch):
    def forbidden_network(*args, **kwargs):
        pytest.fail("Inference must never create a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    with pytest.raises(V.VideoAnalysisError, match="weights are missing"):
        V._load_encoder(tmp_path / "absent.onnx")


def test_wrong_pooled_shape_or_nonfinite_output_fails(encoder, monkeypatch):
    net, _ = V._load_encoder()
    for output in (np.zeros(1000), np.full(1280, np.nan)):
        monkeypatch.setattr(net, "forward", lambda layer: output)
        with pytest.raises(V.VideoAnalysisError, match="invalid pooled"):
            V._encode(np.zeros((16, 16, 3), dtype=np.uint8), V._load_encoder())


def test_extraction_has_fixed_numeric_schema_and_relative_windows(video, encoder):
    result = V.extract_video(video, deadline=time.monotonic() + 15)
    assert result["clip_features"].shape == (1152,)
    assert len(V.CLIP_FEATURE_NAMES) == 1152
    assert len(V.WINDOW_FEATURE_NAMES) == 384
    assert len(set(V.CLIP_FEATURE_NAMES)) == 1152
    assert np.isfinite(result["clip_features"]).all()
    assert 3.9 <= result["duration_sec"] <= 4.1
    assert result["sampling"]["decoded_frames"] == 40
    assert 6 <= result["sampling"]["sampled_frames"] <= 8
    assert result["sampling"]["feature_config"] == V.FEATURE_CONFIG
    assert len(result["windows"]) == 2
    for window in result["windows"]:
        assert window["features"].shape == (384,)
        assert 0 <= window["timestamp_sec"] < 4


def test_numeric_features_ignore_filename_and_repeat_without_network(video, encoder, tmp_path, monkeypatch):
    alias = tmp_path / "positive_label_123.mov"
    shutil.copyfile(video, alias)

    def forbidden_network(*args, **kwargs):
        pytest.fail("Local video extraction attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    first = V.extract_video(video)
    second = V.extract_video(alias)
    np.testing.assert_array_equal(first["clip_features"], second["clip_features"])
    for one, two in zip(first["windows"], second["windows"], strict=True):
        np.testing.assert_array_equal(one["features"], two["features"])
    assert not any(word in name for name in V.CLIP_FEATURE_NAMES
                   for word in ("filename", "duration", "timestamp", "audio", "identity", "target"))


def test_temporal_summary_stays_inside_each_window(video, encoder, monkeypatch):
    values = iter(np.full(128, index, dtype=np.float32) for index in range(8))
    monkeypatch.setattr(V, "_encode", lambda frame, encoder: next(values))
    result = V.extract_video(video)
    for window in result["windows"]:
        np.testing.assert_allclose(window["features"][128:256], np.std([0, 1, 2, 3]))
        np.testing.assert_allclose(window["features"][256:], 0.75)


def test_actual_jpeg_evidence_and_honest_description(video, encoder, tmp_path):
    result = V.analyze_video(video, model_file(tmp_path))
    assert result["verdict"] == "violation"
    event = result["violations"][0]
    assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert "MobileNetV2" in event["description"]
    assert "не установлены" in event["description"]
    assert "юридическое" in event["description"]
    frame = event["evidence_frames"][0]
    jpeg = base64.b64decode(frame["image_base64"], validate=True)
    assert jpeg.startswith(b"\xff\xd8")
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (64, 96, 3)
    assert event["timecode_sec"] == frame["timestamp_sec"]
    assert 0 <= event["timecode_sec"] < 4


def test_negative_output_and_temporal_fallback(video, encoder, tmp_path):
    assert V.analyze_video(video, model_file(tmp_path, verdict=0)) == {
        "verdict": "no_violation", "violations": []}
    result = V.analyze_video(video, model_file(tmp_path, window_model=False))
    assert 0 <= result["violations"][0]["timecode_sec"] < 4


@pytest.mark.parametrize("field,value", [
    ("schema_version", 99), ("engine", "baseline"), ("feature_names", {}),
    ("feature_config", {}), ("weights_sha256", "wrong"), ("source_sha256", "stale"),
    ("threshold", 2), ("clip_model", None), ("window_model", "unsupported"),
])
def test_wrong_model_or_source_schema_fails_before_decode(video, encoder, tmp_path, field, value):
    path = model_file(tmp_path)
    artifact = joblib.load(path)
    artifact[field] = value
    joblib.dump(artifact, path)
    with pytest.raises(V.VideoAnalysisError):
        V.analyze_video(video, path)
    assert not encoder[1]


def test_deadline_prevents_model_weights_and_decode(video):
    with pytest.raises(TimeoutError, match="deadline"):
        V.extract_video(video, deadline=time.monotonic() - 1)
    with pytest.raises(TimeoutError, match="deadline"):
        V.analyze_video(video, deadline=time.monotonic() - 1)


def test_deadline_is_checked_after_each_inference(video, encoder, monkeypatch):
    times = iter([0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="deadline"):
        V.extract_video(video, deadline=1.0)


def test_frame_safety_limit_fails_instead_of_returning_partial_features(video, encoder, monkeypatch):
    monkeypatch.setattr(V, "MAX_DECODED_FRAMES", 5)
    with pytest.raises(V.VideoAnalysisError, match="frame safety"):
        V.extract_video(video)
