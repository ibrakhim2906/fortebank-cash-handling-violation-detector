"""Fixed, local Qwen action observations as label-independent video features.

These descriptors are pretrained-model observations, never verified action
labels or legal conclusions. Every video uses identical uniform sampling. No
organizer target, event anchor, filename or operator review enters a prompt or
classifier input. Media and model text remain local and in memory.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time

import cv2
import joblib
import numpy as np

from scripts import provision_action_model as _provision
from src import config
from src import video_action_proposals as _proposals
from src.video_actions import ACTION_IDS
from src.video_analysis import (
    VideoAnalysisError, _check_deadline, _open_video, _positive_probability,
    evidence_frames,
)


SCHEMA_VERSION = 1
ENGINE = "qwen-actions"
WEIGHTS_SHA256 = _provision.MODEL_SHA256
WINDOW_COUNT = 4
WINDOW_MAX_SECONDS = 4.0
FRAME_COUNT = 3
FRAME_MAX_SIDE = 384
JPEG_QUALITY = 85
LOCAL_PORT = 8003
REQUEST_TIMEOUT_SECONDS = 45.0
PROCESSING_TIMEOUT_SECONDS = 240.0
CLIP_STATISTICS = ("max", "mean", "std")
VISIBILITY_IDS = ("clear", "partial", "unobservable")
INHERITED_SOURCES = {
    relative: hashlib.sha256((config.ROOT / relative).read_bytes()).hexdigest()
    for relative in (
        "src/video_analysis.py", "src/video_action_proposals.py",
        "src/video_actions.py", "scripts/provision_action_model.py",
    )
}
FEATURE_CONFIG = {
    "schema_version": SCHEMA_VERSION, "engine": ENGINE,
    "model_id": _provision.MODEL_ID, "model_revision": _provision.MODEL_REVISION,
    "weights_sha256": WEIGHTS_SHA256,
    "projector_sha256": _provision.PROJECTOR_SHA256,
    "runtime_release": _provision.RELEASE,
    "runtime_archive_sha256": _provision.ASSETS[2].sha256,
    "prompt_recipe_sha256": _proposals.RECIPE_SHA256,
    "inherited_sources": dict(INHERITED_SOURCES),
    "action_ids": list(ACTION_IDS), "visibility_ids": list(VISIBILITY_IDS),
    "window_count": WINDOW_COUNT,
    "window_centers_fraction": [(2 * index + 1) / (2 * WINDOW_COUNT)
                                for index in range(WINDOW_COUNT)],
    "window_width": "min(4 seconds, duration/window_count)",
    "frame_selection": "start, midpoint, end of each uniform window; cap at last frame",
    "frame_count_per_window": FRAME_COUNT, "frame_max_side": FRAME_MAX_SIDE,
    "jpeg_quality": JPEG_QUALITY, "feature_states": ["present", "unknown"],
    "unlisted_or_absent_action": "unknown", "all_unobservable": "processing failure",
    "clip_statistics": list(CLIP_STATISTICS),
    "temperature": 0, "seed": config.SEED, "max_tokens": 256,
    "runtime_context": 4096, "runtime_image_max_tokens": 256,
    "cpu_only": True, "local_port": LOCAL_PORT,
    "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    "processing_timeout_seconds": PROCESSING_TIMEOUT_SECONDS,
    "metadata_features": False,
    "failed_seek_fallback": "frame_index_then_sequential_from_start",
}
WINDOW_FEATURE_NAMES = tuple(
    f"qwen_{action_id}_{state}" for state in ("present", "unknown") for action_id in ACTION_IDS
) + tuple(f"qwen_visibility_{visibility}" for visibility in VISIBILITY_IDS)
CLIP_FEATURE_NAMES = tuple(
    f"{name}_{stat}" for stat in CLIP_STATISTICS for name in WINDOW_FEATURE_NAMES
)


def uniform_windows(duration_sec):
    """Select intervals only from duration; never accepts labels or anchors."""
    if (isinstance(duration_sec, bool) or not isinstance(duration_sec, (int, float))
            or not math.isfinite(duration_sec) or not 0 < duration_sec <= 300):
        raise VideoAnalysisError("Uniform action sampling requires supported duration metadata")
    width = min(WINDOW_MAX_SECONDS, float(duration_sec) / WINDOW_COUNT)
    return [{"start_sec": float(duration_sec) * fraction - width / 2,
             "end_sec": float(duration_sec) * fraction + width / 2}
            for fraction in FEATURE_CONFIG["window_centers_fraction"]]


def observation_features(observation):
    """Represent observations and abstentions without treating absence as fact."""
    try:
        value = _proposals.validate_observations(observation)
    except (TypeError, ValueError, KeyError):
        raise VideoAnalysisError("Local action encoder returned invalid structured observations") from None
    states = {item["action_id"]: item["presence"] for item in value["actions"]}
    present = [float(states.get(identifier, "unknown") == "present") for identifier in ACTION_IDS]
    unknown = [1.0 - item for item in present]
    visibility = [float(value["visibility"] == identifier) for identifier in VISIBILITY_IDS]
    return np.asarray(present + unknown + visibility, dtype=np.float32), value["visibility"]


def _read_frame_at(cap, timestamp, fps, deadline):
    """Keep successful seeks unchanged; recover MOV end-of-stream seek quirks."""
    _check_deadline(deadline)
    if cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return frame
    _check_deadline(deadline)
    count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(count) or count < 1:
        raise VideoAnalysisError("Failed action seek has no usable frame count")
    target = min(max(0, int(round(float(timestamp) * fps))), int(count) - 1)
    if cap.set(cv2.CAP_PROP_POS_FRAMES, target):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return frame
    _check_deadline(deadline)
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
        raise VideoAnalysisError("Failed action seek could not restart sequential decoding")
    for _ in range(target + 1):
        _check_deadline(deadline)
        if not cap.grab():
            raise VideoAnalysisError("Sequential action sampling ended before the requested frame")
    ok, frame = cap.retrieve()
    _check_deadline(deadline)
    if not ok or frame is None or not frame.size:
        raise VideoAnalysisError("A sequential action sample could not be decoded")
    return frame


def _sample_interval(cap, fps, duration, interval, deadline):
    start, end = interval["start_sec"], interval["end_sec"]
    times = (start, (start + end) / 2, min(end, max(0.0, duration - 1 / fps)))
    images, actual_times = [], []
    for timestamp in times:
        _check_deadline(deadline)
        frame = _read_frame_at(cap, timestamp, fps, deadline)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise VideoAnalysisError("Uniform action sampling requires three-channel visual frames")
        actual = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        if not math.isfinite(actual) or actual < 0 or (timestamp > 0 and actual <= 0):
            actual = max(0.0, (float(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1) / fps)
        if not math.isfinite(actual) or not 0 <= actual <= duration:
            raise VideoAnalysisError("Uniform action sampling returned invalid frame timing")
        height, width = frame.shape[:2]
        scale = min(1.0, FRAME_MAX_SIDE / max(height, width))
        if scale < 1:
            frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise VideoAnalysisError("A uniformly sampled action frame could not be encoded")
        images.append(encoded.tobytes())
        actual_times.append(float(actual))
        _check_deadline(deadline)
    return images, actual_times


def _check_inherited_sources():
    for relative, expected in INHERITED_SOURCES.items():
        if hashlib.sha256((config.ROOT / relative).read_bytes()).hexdigest() != expected:
            raise VideoAnalysisError("Local action feature helper source changed; use a fresh process and run")


def extract_video(path: str | Path, deadline: float | None = None) -> dict:
    """Return fixed uniform observations, with failures distinct from negatives."""
    started = time.monotonic()
    deadline = min(started + PROCESSING_TIMEOUT_SECONDS, deadline) if deadline is not None else started + PROCESSING_TIMEOUT_SECONDS
    _check_deadline(deadline)
    _check_inherited_sources()
    # Validate model/projector/runtime bytes before reading any video frame.
    try:
        model = _provision.verify_ready()
    except (OSError, ValueError, RuntimeError):
        raise VideoAnalysisError("Pinned local action model is not ready") from None
    if (model.get("weights_sha256") != WEIGHTS_SHA256
            or model.get("projector_sha256") != FEATURE_CONFIG["projector_sha256"]
            or model.get("model_revision") != FEATURE_CONFIG["model_revision"]
            or model.get("runtime_release") != FEATURE_CONFIG["runtime_release"]):
        raise VideoAnalysisError("Local action model identity differs from the frozen extractor")
    _check_deadline(deadline)
    cap, fps, frame_count = _open_video(path, deadline)
    windows, visibilities, sampled_times = [], [], []
    try:
        if frame_count <= 0:
            raise VideoAnalysisError("Uniform action sampling requires complete frame-count metadata")
        duration = float(frame_count / fps)
        intervals = uniform_windows(duration)
        cv2.setNumThreads(2)
        cv2.setRNGSeed(config.SEED)
        for interval in intervals:
            images, timestamps = _sample_interval(cap, fps, duration, interval, deadline)
            _check_deadline(deadline)
            timeout = min(REQUEST_TIMEOUT_SECONDS, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Local video processing deadline exceeded")
            try:
                observation = _proposals.request_local_vision(images, port=LOCAL_PORT, timeout=timeout)
            except (OSError, ValueError, RuntimeError):
                # Never stringify server text or any media-derived details.
                raise VideoAnalysisError("Local action encoder request failed") from None
            _check_deadline(deadline)
            values, visibility = observation_features(observation)
            windows.append({"timestamp_sec": float(np.median(timestamps)), "features": values})
            visibilities.append(visibility)
            sampled_times.extend(timestamps)
    finally:
        cap.release()
    if len(windows) != WINDOW_COUNT:
        raise VideoAnalysisError("Local action encoder did not complete every uniform window")
    if all(value == "unobservable" for value in visibilities):
        raise VideoAnalysisError("All uniform action samples are visually unobservable")
    matrix = np.stack([window["features"] for window in windows])
    summaries = {"max": matrix.max(axis=0), "mean": matrix.mean(axis=0), "std": matrix.std(axis=0)}
    clip = np.concatenate([summaries[stat] for stat in CLIP_STATISTICS]).astype(np.float32)
    if clip.shape != (len(CLIP_FEATURE_NAMES),) or not np.isfinite(clip).all():
        raise VideoAnalysisError("Local action encoder produced invalid numeric features")
    _check_deadline(deadline)
    return {
        "clip_features": clip, "windows": windows, "duration_sec": duration,
        "sampling": {
            "retrieved_frames": len(sampled_times), "sampled_frames": len(sampled_times),
            "window_count": len(windows), "sample_times_sec": sampled_times,
            "sampled_interval_sec": sum(item["end_sec"] - item["start_sec"] for item in intervals),
            "coverage_note": "Sparse still-frame samples do not establish continuous interval coverage",
            "unobservable_windows": visibilities.count("unobservable"),
            "elapsed_sec": time.monotonic() - started, "feature_config": dict(FEATURE_CONFIG),
        },
    }


def _load_model(model_path):
    path = Path(model_path or os.environ.get(
        "DATATHON_VIDEO_MODEL", str(config.DATA_PROC / "video-qwen-actions/models/baseline.joblib")))
    if not path.is_file():
        raise VideoAnalysisError("Local action-feature classifier is missing; train before analysis")
    # Only trusted local training artifacts are accepted, never uploaded pickle.
    artifact = joblib.load(path)
    if (not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION
            or artifact.get("engine") != ENGINE):
        raise VideoAnalysisError("Unsupported local action-feature model schema or engine")
    names = artifact.get("feature_names", {})
    if (not isinstance(names, dict)
            or tuple(names.get("clip", ())) != CLIP_FEATURE_NAMES
            or tuple(names.get("window", ())) != WINDOW_FEATURE_NAMES):
        raise VideoAnalysisError("Local action-feature model feature names differ from the extractor")
    if (artifact.get("feature_config") != FEATURE_CONFIG
            or artifact.get("weights_sha256") != WEIGHTS_SHA256
            or artifact.get("inherited_sources") != INHERITED_SOURCES
            or artifact.get("source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise VideoAnalysisError("Local action-feature model provenance differs from the extractor")
    _check_inherited_sources()
    if not hasattr(artifact.get("clip_model"), "predict_proba"):
        raise VideoAnalysisError("Local action-feature clip model does not support probabilities")
    if (artifact.get("window_model") is not None
            and not hasattr(artifact["window_model"], "predict_proba")):
        raise VideoAnalysisError("Local action-feature window model does not support probabilities")
    threshold = artifact.get("threshold")
    if type(threshold) not in (int, float) or threshold != 0.5:
        raise VideoAnalysisError("Local action-feature model requires the fixed 0.5 threshold")
    return artifact


def analyze_video(path: str | Path, model_path: str | Path | None = None,
                  deadline: float | None = None) -> dict:
    """Classify original video targets without asserting verified action types."""
    started = time.monotonic()
    deadline = min(started + PROCESSING_TIMEOUT_SECONDS, deadline) if deadline is not None else started + PROCESSING_TIMEOUT_SECONDS
    _check_deadline(deadline)
    artifact = _load_model(model_path)
    extracted = extract_video(path, deadline)
    probability = float(_positive_probability(artifact["clip_model"], extracted["clip_features"].reshape(1, -1))[0])
    _check_deadline(deadline)
    if probability < 0.5:
        return {"verdict": "no_violation", "violations": []}
    windows = extracted["windows"]
    if not windows:
        raise VideoAnalysisError("Local action encoder returned no candidate windows")
    values = np.stack([window["features"] for window in windows])
    if artifact.get("window_model") is not None:
        scores = _positive_probability(artifact["window_model"], values)
    else:
        scores = values[:, :len(ACTION_IDS)].sum(axis=1)
    timestamp = float(windows[int(np.argmax(scores))]["timestamp_sec"])
    frames = evidence_frames(path, [timestamp], deadline)
    _check_deadline(deadline)
    return {"verdict": "violation", "violations": [{
        "type": "Визуальный признак для проверки",
        "timecode_sec": frames[0]["timestamp_sec"],
        "description": (
            "Локальная модель на автоматических наблюдениях Qwen3-VL выделила "
            "паттерн, похожий на примеры с положительной меткой. Фрагмент и кадр "
            "предлагаются для проверки человеком. Наблюдения модели не являются "
            "подтверждёнными метками действий. Конкретный тип нарушения и "
            "нормативный источник этой моделью не установлены. Это не юридическое "
            "или дисциплинарное заключение."
        ), "evidence_frames": frames,
    }]}
