"""Small, offline visual baseline; numeric similarity is not a legal conclusion.

No filenames, labels, evidence pixels, or source timestamps enter the features.
Videos and evidence are decoded locally with OpenCV only. This baseline has no
event taxonomy or rule-grounding capability; its output must say so explicitly.
"""
from __future__ import annotations

import base64
import hashlib
import math
import os
from pathlib import Path
import time

import cv2
import joblib
import numpy as np

from src import config as C

SCHEMA_VERSION = 1
SAMPLE_FPS = 2.0
WINDOW_SECONDS = 2.0
MAX_DURATION_SECONDS = 300.0
MAX_DECODED_FRAMES = 18_000
FEATURE_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "sample_fps": SAMPLE_FPS,
    "window_seconds": WINDOW_SECONDS,
    "resize": [96, 96],
    "grid": [3, 3],
}
_GRID = tuple(f"r{r}_c{c}" for r in range(3) for c in range(3))
_FRAME_FEATURE_NAMES = (
    *(f"gray_{cell}" for cell in _GRID),
    *(f"hue_bin_{i}" for i in range(8)),
    "saturation_mean", "value_std",
    *(f"edge_{cell}" for cell in _GRID),
    *(f"motion_{cell}" for cell in _GRID),
)
WINDOW_FEATURE_NAMES = tuple(
    f"{name}_{stat}" for stat in ("mean", "std") for name in _FRAME_FEATURE_NAMES
)
CLIP_FEATURE_NAMES = tuple(
    f"{name}_{stat}" for stat in ("mean", "std", "max") for name in WINDOW_FEATURE_NAMES
)


class VideoAnalysisError(ValueError):
    """Invalid video/model input, distinct from a negative verdict."""


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Local video processing deadline exceeded")


def _open_video(path: str | Path, deadline: float | None):
    _check_deadline(deadline)
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoAnalysisError("Video file is absent or empty")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoAnalysisError("OpenCV could not open the video")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        cap.release()
        raise VideoAnalysisError("Video has no supported frame-rate metadata")
    if math.isfinite(count) and count > 0 and count / fps > MAX_DURATION_SECONDS:
        cap.release()
        raise VideoAnalysisError("Video exceeds the supported 300 second duration")
    return cap, fps, max(0, int(count)) if math.isfinite(count) else 0


def _grid_means(array: np.ndarray) -> list[float]:
    return [float(part.mean()) for row in np.array_split(array, 3, axis=0)
            for part in np.array_split(row, 3, axis=1)]


def _frame_features(frame: np.ndarray, previous_gray: np.ndarray | None):
    small = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue = np.histogram(hsv[:, :, 0], bins=8, range=(0, 180))[0] / gray.size
    edges = cv2.Canny(gray, 50, 100).astype(np.float32) / 255.0
    motion = (np.zeros_like(gray, dtype=np.float32) if previous_gray is None
              else cv2.absdiff(gray, previous_gray).astype(np.float32) / 255.0)
    values = [*_grid_means(gray.astype(np.float32) / 255.0), *hue,
              float(hsv[:, :, 1].mean() / 255.0), float(hsv[:, :, 2].std() / 255.0),
              *_grid_means(edges), *_grid_means(motion)]
    return np.asarray(values, dtype=np.float32), gray


def extract_video(path: str | Path, deadline: float | None = None) -> dict:
    """Decode in order and return bounded numeric clip/window descriptors.

    ``deadline`` is an absolute ``time.monotonic()`` value. Source-relative
    decoder timestamps are preferred; frame index / FPS is the fallback.
    Window timestamps localize observations but are never model features.
    """
    started = time.monotonic()
    cap, fps, expected_frames = _open_video(path, deadline)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(C.SEED)
    rows: list[np.ndarray] = []
    timestamps: list[float] = []
    previous_gray = None
    next_sample = 0.0
    last_timestamp = -1.0
    count = 0
    timestamp_fallbacks = 0
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
            vector, previous_gray = _frame_features(frame, previous_gray)
            rows.append(vector)
            timestamps.append(source_timestamp)
            # Avoid burst sampling after a discontinuity in a variable-FPS clip.
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
        windows.append({
            "timestamp_sec": float(np.median(times[mask])),
            "features": np.concatenate((values.mean(axis=0), values.std(axis=0))).astype(np.float32),
        })
    window_values = np.vstack([row["features"] for row in windows])
    clip_features = np.concatenate((window_values.mean(axis=0), window_values.std(axis=0),
                                    window_values.max(axis=0))).astype(np.float32)
    return {
        "clip_features": clip_features,
        "windows": windows,
        "duration_sec": float(last_timestamp + 1.0 / fps),
        "sampling": {
            "decoded_frames": count, "sampled_frames": len(rows),
            "source_fps": fps, "timestamp_fallbacks": timestamp_fallbacks,
            "elapsed_sec": time.monotonic() - started,
            "feature_config": dict(FEATURE_CONFIG),
        },
    }


def _load_model(model_path: str | Path | None) -> dict:
    path = Path(model_path or os.environ.get(
        "DATATHON_VIDEO_MODEL", str(C.ROOT / "data/processed/video-baseline/models/baseline.joblib")))
    if not path.is_file():
        raise VideoAnalysisError("Local baseline model is missing; train it before starting analysis")
    # Artifacts must be trusted, locally trained files; never unpickle uploads.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise VideoAnalysisError("Unsupported local model schema")
    feature_names = artifact.get("feature_names", {})
    if not isinstance(feature_names, dict):
        raise VideoAnalysisError("Local model feature schema is invalid")
    for key, expected in (("clip", CLIP_FEATURE_NAMES), ("window", WINDOW_FEATURE_NAMES)):
        if tuple(feature_names.get(key, ())) != tuple(expected):
            raise VideoAnalysisError(f"Local model {key} feature names do not match the extractor")
    if artifact.get("feature_config") != FEATURE_CONFIG:
        raise VideoAnalysisError("Local model sampling configuration does not match the extractor")
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if artifact.get("source_sha256") != source_sha256:
        raise VideoAnalysisError("Local model was built with different analysis source code")
    if not hasattr(artifact.get("clip_model"), "predict_proba"):
        raise VideoAnalysisError("Local clip model does not support probabilities")
    threshold = artifact.get("threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise VideoAnalysisError("Local model threshold is invalid")
    return artifact


def _positive_probability(model, values: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        raise VideoAnalysisError("Local model has no positive class")
    probabilities = np.asarray(model.predict_proba(values), dtype=float)[:, classes.index(1)]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise VideoAnalysisError("Local model returned invalid probabilities")
    return probabilities


def evidence_frames(path: str | Path, timestamps: list[float],
                    deadline: float | None = None) -> list[dict]:
    """Return JPEGs from the actual submitted video, never spreadsheet images."""
    if not 1 <= len(timestamps) <= 3:
        raise VideoAnalysisError("Evidence requires one to three source timestamps")
    cap, fps, _ = _open_video(path, deadline)
    frames = []
    try:
        for timestamp in timestamps:
            _check_deadline(deadline)
            if not math.isfinite(timestamp) or timestamp < 0:
                raise VideoAnalysisError("Evidence timestamp is invalid")
            if not cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0):
                raise VideoAnalysisError("Video does not support evidence seeking")
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                raise VideoAnalysisError("Evidence frame could not be decoded")
            actual_timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not math.isfinite(actual_timestamp) or (timestamp > 0 and actual_timestamp <= 0):
                actual_timestamp = max(0.0, (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1.0) / fps)
            scale = min(1.0, 960.0 / max(frame.shape[:2]))
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise VideoAnalysisError("Evidence frame could not be JPEG encoded")
            frames.append({"timestamp_sec": float(actual_timestamp),
                           "image_base64": base64.b64encode(encoded).decode("ascii")})
    finally:
        cap.release()
    _check_deadline(deadline)
    return frames


def analyze_video(path: str | Path, model_path: str | Path | None = None,
                  deadline: float | None = None) -> dict:
    """Run the saved local baseline; decode/model errors propagate as failures."""
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
        indices = [i for i, name in enumerate(WINDOW_FEATURE_NAMES)
                   if name.startswith("motion_") and name.endswith("_mean")]
        scores = values[:, indices].mean(axis=1)
    timestamp = float(windows[int(np.argmax(scores))]["timestamp_sec"])
    _check_deadline(deadline)
    frames = evidence_frames(path, [timestamp], deadline)
    return {
        "verdict": "violation",
        "violations": [{
            "type": "Визуальный признак для проверки",
            "timecode_sec": frames[0]["timestamp_sec"],
            "description": (
                "Локальная базовая модель выявила визуальный паттерн, похожий на "
                "примеры с положительной меткой. Кадр выбран как кандидат для "
                "проверки человеком. Конкретное действие, тип нарушения и "
                "нормативный источник этой моделью не установлены. "
                "Это не юридическое или дисциплинарное заключение."
            ),
            "evidence_frames": frames,
        }],
    }
