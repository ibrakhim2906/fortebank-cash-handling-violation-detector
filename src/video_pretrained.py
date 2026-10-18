"""Offline MobileNetV2 visual descriptors and locally trained event candidates.

This ImageNet encoder describes appearance; it does not recognize bank actions,
identify people, process audio, or establish legal violations. No media leaves
the machine, and this module never downloads weights or writes decoded frames.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import threading
import time

import cv2
import joblib
import numpy as np

from src import config as C
from src import video_analysis as _baseline
from src.video_analysis import (
    VideoAnalysisError, _check_deadline, _open_video, _positive_probability,
    evidence_frames,
)

SCHEMA_VERSION = 1
ENGINE = "mobilenet"
WEIGHTS_SHA256 = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
DEFAULT_WEIGHTS_PATH = C.ROOT / "models/encoders/mobilenetv2.onnx"
SAMPLE_FPS = 2.0
WINDOW_SECONDS = 2.0
MAX_DURATION_SECONDS = 300.0
MAX_DECODED_FRAMES = 18_000
POOLED_DIM = 1280
EMBEDDING_DIM = 128
OUTPUT_LAYER = "GlobalAveragePool_97"
OUTPUT_TENSOR = "464"
FEATURE_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "engine": ENGINE,
    "weights_sha256": WEIGHTS_SHA256,
    "output_layer": OUTPUT_LAYER,
    "output_tensor": OUTPUT_TENSOR,
    "pooled_dim": POOLED_DIM,
    "embedding_dim": EMBEDDING_DIM,
    "projection": {"kind": "fixed_gaussian", "version": 1, "seed": C.SEED,
                   "dimensions": [POOLED_DIM, EMBEDDING_DIM], "scale": "1/sqrt(output_dim)"},
    "sample_fps": SAMPLE_FPS,
    "window_seconds": WINDOW_SECONDS,
    "color": "RGB",
    "resize": [256, 256],
    "center_crop": [16, 16, 224, 224],
    "interpolation": "INTER_LINEAR",
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "window_statistics": ["mean", "std", "delta_abs_mean"],
    "clip_statistics": ["mean", "std", "max"],
    "delta_boundary": "first sampled frame in each window has zero delta",
    "opencv_threads": 2,
    "helper_source_sha256": hashlib.sha256(Path(_baseline.__file__).read_bytes()).hexdigest(),
}
WINDOW_FEATURE_NAMES = tuple(
    f"mobilenet_{index:04d}_{stat}"
    for stat in FEATURE_CONFIG["window_statistics"] for index in range(EMBEDDING_DIM)
)
CLIP_FEATURE_NAMES = tuple(
    f"{name}_{stat}" for stat in FEATURE_CONFIG["clip_statistics"]
    for name in WINDOW_FEATURE_NAMES
)
_ENCODERS: dict[Path, tuple[object, threading.Lock]] = {}
_CACHE_LOCK = threading.Lock()
_PROJECTION = (np.random.default_rng(C.SEED).standard_normal((POOLED_DIM, EMBEDDING_DIM))
               / math.sqrt(EMBEDDING_DIM)).astype(np.float32)


def _weights_path(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("DATATHON_VIDEO_WEIGHTS", str(DEFAULT_WEIGHTS_PATH))).resolve()


def _load_encoder(path: str | Path | None = None):
    """Hash on every extraction, including cache hits; never trust file metadata."""
    path = _weights_path(path)
    if not path.is_file():
        raise VideoAnalysisError("Local MobileNetV2 weights are missing; provision the verified ONNX file")
    # Read one verified snapshot so replacement between hashing and loading cannot
    # cause OpenCV to load bytes different from the bytes whose digest we checked.
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != WEIGHTS_SHA256:
        raise VideoAnalysisError("Local MobileNetV2 weights SHA-256 does not match the approved encoder")
    with _CACHE_LOCK:
        if path not in _ENCODERS:
            try:
                net = cv2.dnn.readNetFromONNX(np.frombuffer(payload, dtype=np.uint8))
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                if OUTPUT_LAYER not in net.getLayerNames():
                    raise VideoAnalysisError("Approved MobileNetV2 pooled output layer is missing")
            except cv2.error as exc:
                raise VideoAnalysisError("OpenCV could not load the approved MobileNetV2 encoder") from exc
            _ENCODERS[path] = net, threading.Lock()
        return _ENCODERS[path]


def _preprocess(frame: np.ndarray) -> np.ndarray:
    """Match OpenCV Zoo's RGB resize/crop and ImageNet normalization."""
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise VideoAnalysisError("Encoder requires a nonempty three-channel BGR frame")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    cropped = resized[16:240, 16:240].astype(np.float32) / 255.0
    normalized = ((cropped - np.asarray(FEATURE_CONFIG["normalization_mean"], dtype=np.float32))
                  / np.asarray(FEATURE_CONFIG["normalization_std"], dtype=np.float32))
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, :, :, :])


def _encode(frame: np.ndarray, encoder) -> np.ndarray:
    net, lock = encoder
    with lock:
        try:
            net.setInput(_preprocess(frame))
            output = np.asarray(net.forward(OUTPUT_TENSOR), dtype=np.float32).reshape(-1)
        except cv2.error as exc:
            raise VideoAnalysisError("Local MobileNetV2 inference failed") from exc
    if output.shape != (POOLED_DIM,) or not np.isfinite(output).all():
        raise VideoAnalysisError("MobileNetV2 encoder returned invalid pooled descriptors")
    # Fixed label-free projection, with no corpus-level fitting or PCA.
    projected = np.asarray(output @ _PROJECTION, dtype=np.float32)
    if not np.isfinite(projected).all():
        raise VideoAnalysisError("MobileNetV2 projection returned invalid descriptors")
    return projected


def extract_video(path: str | Path, deadline: float | None = None,
                  *, encoder_path: str | Path | None = None) -> dict:
    """Return fixed-length numeric descriptors; all preprocessing is label-free.

    Relative timestamps only select/localize windows. They, source FPS, duration,
    filenames, labels and audio never become classifier inputs.
    """
    started = time.monotonic()
    _check_deadline(deadline)
    encoder = _load_encoder(encoder_path)
    _check_deadline(deadline)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(C.SEED)
    cap, fps, expected_frames = _open_video(path, deadline)
    rows, timestamps = [], []
    next_sample, last_timestamp = 0.0, -1.0
    count, timestamp_fallbacks = 0, 0
    try:
        while True:
            _check_deadline(deadline)
            if not cap.grab():
                break
            if count >= MAX_DECODED_FRAMES:
                raise VideoAnalysisError("Video exceeds the decoded-frame safety limit")
            source_timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if (not math.isfinite(source_timestamp) or source_timestamp < 0
                    or (count > 0 and source_timestamp <= last_timestamp)):
                source_timestamp = count / fps
                timestamp_fallbacks += 1
            if source_timestamp > MAX_DURATION_SECONDS:
                raise VideoAnalysisError("Video exceeds the supported 300 second duration")
            last_timestamp = source_timestamp
            count += 1
            if source_timestamp + 1e-6 < next_sample:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None or frame.size == 0:
                raise VideoAnalysisError("A sampled video frame could not be decoded")
            rows.append(_encode(frame, encoder))
            _check_deadline(deadline)
            timestamps.append(source_timestamp)
            next_sample = source_timestamp + 1.0 / SAMPLE_FPS
    finally:
        cap.release()
    _check_deadline(deadline)
    if not rows:
        raise VideoAnalysisError("Video contains no decodable frames")
    if expected_frames > count + max(2, int(expected_frames * 0.01)):
        raise VideoAnalysisError("Video ended before its declared frame count")
    frame_values = np.vstack(rows)
    times = np.asarray(timestamps, dtype=float)
    window_ids = np.floor(times / WINDOW_SECONDS).astype(int)
    windows = []
    for index in np.unique(window_ids):
        mask = window_ids == index
        values = frame_values[mask]
        # No delta crosses a localization-window boundary. An isolated sample
        # contributes zero temporal change, rather than an undefined NaN.
        changes = np.abs(np.diff(values, axis=0, prepend=values[:1]))
        windows.append({
            "timestamp_sec": float(np.median(times[mask])),
            "features": np.concatenate((values.mean(axis=0), values.std(axis=0),
                                        changes.mean(axis=0))).astype(np.float32),
        })
    window_values = np.vstack([row["features"] for row in windows])
    clip_features = np.concatenate((window_values.mean(axis=0), window_values.std(axis=0),
                                    window_values.max(axis=0))).astype(np.float32)
    _check_deadline(deadline)
    return {
        "clip_features": clip_features,
        "windows": windows,
        "duration_sec": float(last_timestamp + 1.0 / fps),
        "sampling": {
            "decoded_frames": count, "sampled_frames": len(rows), "source_fps": fps,
            "timestamp_fallbacks": timestamp_fallbacks,
            "elapsed_sec": time.monotonic() - started,
            "feature_config": dict(FEATURE_CONFIG),
        },
    }


def _load_model(model_path: str | Path | None) -> dict:
    path = Path(model_path or os.environ.get(
        "DATATHON_VIDEO_MODEL", str(C.DATA_PROC / "video-mobilenet/models/baseline.joblib")))
    if not path.is_file():
        raise VideoAnalysisError("Local pretrained-feature model is missing; train it before analysis")
    # Only trusted local training artifacts are accepted, never uploaded pickle files.
    artifact = joblib.load(path)
    if (not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION
            or artifact.get("engine") != ENGINE):
        raise VideoAnalysisError("Unsupported local pretrained model schema or engine")
    names = artifact.get("feature_names", {})
    if not isinstance(names, dict):
        raise VideoAnalysisError("Local pretrained model feature names are invalid")
    for key, expected in (("clip", CLIP_FEATURE_NAMES), ("window", WINDOW_FEATURE_NAMES)):
        if tuple(names.get(key, ())) != expected:
            raise VideoAnalysisError(f"Local pretrained model {key} feature names do not match")
    if artifact.get("feature_config") != FEATURE_CONFIG:
        raise VideoAnalysisError("Local pretrained model feature configuration does not match")
    if artifact.get("weights_sha256") != WEIGHTS_SHA256:
        raise VideoAnalysisError("Local pretrained model encoder weights do not match")
    if artifact.get("source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise VideoAnalysisError("Local pretrained model was built with different analysis source code")
    if not hasattr(artifact.get("clip_model"), "predict_proba"):
        raise VideoAnalysisError("Local pretrained clip model does not support probabilities")
    if (artifact.get("window_model") is not None
            and not hasattr(artifact["window_model"], "predict_proba")):
        raise VideoAnalysisError("Local pretrained window model does not support probabilities")
    threshold = artifact.get("threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise VideoAnalysisError("Local pretrained model threshold is invalid")
    return artifact


def analyze_video(path: str | Path, model_path: str | Path | None = None,
                  deadline: float | None = None) -> dict:
    """Apply the local supervised classifier without inventing event semantics."""
    _check_deadline(deadline)
    artifact = _load_model(model_path)
    extracted = extract_video(path, deadline)
    probability = float(_positive_probability(
        artifact["clip_model"], extracted["clip_features"].reshape(1, -1))[0])
    _check_deadline(deadline)
    if probability < artifact.get("threshold", 0.5):
        return {"verdict": "no_violation", "violations": []}
    windows = extracted["windows"]
    values = np.vstack([row["features"] for row in windows])
    if artifact.get("window_model") is not None:
        scores = _positive_probability(artifact["window_model"], values)
    else:
        # Temporal embedding change only nominates a review candidate. This is
        # a deterministic fallback, not an event-localization accuracy claim.
        scores = values[:, 2 * EMBEDDING_DIM:3 * EMBEDDING_DIM].mean(axis=1)
    timestamp = float(windows[int(np.argmax(scores))]["timestamp_sec"])
    _check_deadline(deadline)
    frames = evidence_frames(path, [timestamp], deadline)
    return {
        "verdict": "violation",
        "violations": [{
            "type": "Визуальный признак для проверки",
            "timecode_sec": frames[0]["timestamp_sec"],
            "description": (
                "Локальная модель на визуальных признаках MobileNetV2 выявила "
                "паттерн, похожий на примеры с положительной меткой. Кадр выбран "
                "как кандидат для проверки человеком. Конкретное действие, тип "
                "нарушения и нормативный источник этой моделью не установлены. "
                "Это не юридическое или дисциплинарное заключение."
            ),
            "evidence_frames": frames,
        }],
    }
