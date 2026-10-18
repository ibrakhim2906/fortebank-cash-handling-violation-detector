"""Local, unverified static observations from fixed same-instant image views.

No organizer label, timestamp annotation, filename, review or regulatory claim
enters extraction. Private observations and sample provenance stay on disk;
they are pretrained numeric inputs, never independently verified action truth.
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
from src import video_local_observations as _observations
from src.video_analysis import (VideoAnalysisError, _check_deadline, _open_video,
                                _positive_probability, evidence_frames)
from src.video_qwen_actions import _read_frame_at


SCHEMA_VERSION = 1
ENGINE = "qwen-spatial-v3"
WEIGHTS_SHA256 = _provision.MODEL_SHA256
WINDOW_COUNT = 4
WINDOW_MAX_SECONDS = 4.0
FRAME_MAX_SIDE = 640
FRAME_COUNT = 3  # Three views of one instant, not three temporal samples.
JPEG_QUALITY = 90
REQUEST_TIMEOUT_SECONDS = 45.0
PROCESSING_TIMEOUT_SECONDS = 240.0
CLIP_STATISTICS = ("max", "mean", "std")
FIELDS = _observations.FIELDS
INHERITED_SOURCES = {
    relative: hashlib.sha256((config.ROOT / relative).read_bytes()).hexdigest()
    for relative in (
        "src/video_analysis.py", "src/video_qwen_actions.py",
        "src/video_action_proposals.py", "src/video_actions.py",
        "scripts/provision_action_model.py", "src/video_local_observations.py",
    )
}
FEATURE_CONFIG = {
    "schema_version": SCHEMA_VERSION, "engine": ENGINE,
    "model_id": _provision.MODEL_ID, "model_revision": _provision.MODEL_REVISION,
    "weights_sha256": WEIGHTS_SHA256, "projector_sha256": _provision.PROJECTOR_SHA256,
    "runtime_release": _provision.RELEASE,
    "runtime_archive_sha256": _provision.ASSETS[2].sha256,
    "prompt_recipe_sha256": _observations.RECIPE_SHA256,
    "inherited_sources": dict(INHERITED_SOURCES),
    "fields": {field: list(states) for field, states in FIELDS.items()},
    "window_count": WINDOW_COUNT,
    "window_centers_fraction": [(2 * index + 1) / (2 * WINDOW_COUNT) for index in range(WINDOW_COUNT)],
    "window_width": "min(4 seconds, duration/window_count); supervision only",
    "frame_selection": "one frame at each uniform center",
    "views": "overview, first square and last square along source long axis; same instant",
    "frame_max_side": FRAME_MAX_SIDE, "upsampling": False, "jpeg_quality": JPEG_QUALITY,
    "alignment_tolerance": "max(0.25 seconds, 2/source_fps)",
    "alignment_failure": "retry sequential decoding then explicit processing failure",
    "strict_increasing_samples": True, "all_unobservable": "processing failure",
    "feature_encoding": "strict per-field one-hot; no default or missing-value filling",
    "clip_statistics": list(CLIP_STATISTICS), "metadata_features": False,
    "temperature": 0, "seed": config.SEED, "max_tokens": 256,
    "runtime_context": 4096, "runtime_image_max_tokens": 256,
    "cpu_only": True, "local_port": 8003,
    "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
    "processing_timeout_seconds": PROCESSING_TIMEOUT_SECONDS,
}
WINDOW_FEATURE_NAMES = tuple(f"spatial_{field}_{state}" for field, states in FIELDS.items() for state in states)
CLIP_FEATURE_NAMES = tuple(f"{name}_{stat}" for stat in CLIP_STATISTICS for name in WINDOW_FEATURE_NAMES)


def get_encoder_provenance():
    return {"engine": ENGINE, "engine_module": "src.video_qwen_spatial_v3",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "weights_sha256": WEIGHTS_SHA256, "inherited_sources": dict(INHERITED_SOURCES)}


def _check_inherited_sources():
    for relative, expected in INHERITED_SOURCES.items():
        if hashlib.sha256((config.ROOT / relative).read_bytes()).hexdigest() != expected:
            raise VideoAnalysisError("Spatial feature helper source changed; use a fresh process and run")


def uniform_windows(duration_sec):
    if (isinstance(duration_sec, bool) or not isinstance(duration_sec, (int, float))
            or not math.isfinite(duration_sec) or not 0 < duration_sec <= 300):
        raise VideoAnalysisError("Uniform spatial sampling requires supported duration metadata")
    width = min(WINDOW_MAX_SECONDS, float(duration_sec) / WINDOW_COUNT)
    return [{"start_sec": duration_sec * fraction - width / 2,
             "end_sec": duration_sec * fraction + width / 2}
            for fraction in FEATURE_CONFIG["window_centers_fraction"]]


def observation_features(observation):
    if (not isinstance(observation, dict) or set(observation) != set(FIELDS)
            or any(not isinstance(observation[field], str) or observation[field] not in states
                   for field, states in FIELDS.items())):
        raise VideoAnalysisError("Spatial encoder returned invalid structured observations")
    values = np.asarray([float(observation[field] == state) for field, states in FIELDS.items()
                         for state in states], dtype=np.float32)
    return values, observation["visibility"]


def same_instant_views(frame):
    """Return JPEG views and source-coordinate rectangles without upsampling."""
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
            or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1):
        raise VideoAnalysisError("Spatial sampling requires a nonempty three-channel visual frame")
    height, width = frame.shape[:2]
    side = min(height, width)
    rectangles = [("overview", (0, 0, width, height)),
                  ("first_square", (0, 0, side, side)),
                  ("last_square", (0, height - side, side, height) if height >= width
                   else (width - side, 0, width, side))]
    images, metadata = [], []
    for role, rectangle in rectangles:
        left, top, right, bottom = rectangle
        view = frame[top:bottom, left:right]
        scale = min(1.0, FRAME_MAX_SIDE / max(view.shape[:2]))
        if scale < 1:
            view = cv2.resize(view, (max(1, round(view.shape[1] * scale)),
                                    max(1, round(view.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise VideoAnalysisError("A spatial image view could not be encoded")
        payload = encoded.tobytes()
        images.append(payload)
        metadata.append({"role": role, "crop_rect_xyxy": list(rectangle),
                         "shape": list(view.shape), "sha256": hashlib.sha256(payload).hexdigest()})
    return images, metadata


def _decoded_position(cap, requested, fps, frame_count, duration):
    raw_index = float(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if (not math.isfinite(raw_index) or abs(raw_index - round(raw_index)) > 1e-5
            or not 0 <= raw_index < frame_count):
        raise VideoAnalysisError("Spatial sampling returned an invalid source frame index")
    index = int(round(raw_index))
    actual = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
    timestamp_source = "decoder"
    if not math.isfinite(actual) or actual < 0 or (actual == 0 and index > 0):
        actual, timestamp_source = index / fps, "frame_index_over_fps"
    if (not math.isfinite(actual) or not 0 <= actual < duration
            or abs(actual - requested) > max(.25, 2 / fps)):
        raise VideoAnalysisError("Spatial sample is not aligned with the requested instant")
    return actual, index, timestamp_source


def _sample_center(cap, requested, fps, frame_count, duration, deadline):
    frame = _read_frame_at(cap, requested, fps, deadline)
    try:
        actual, index, timestamp_source = _decoded_position(cap, requested, fps, frame_count, duration)
        decoding_mode = "seek_or_frozen_fallback"
    except VideoAnalysisError:
        # A nominally successful seek can return the wrong frame. Re-decode in
        # order rather than silently attaching the requested time to it.
        _check_deadline(deadline)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise VideoAnalysisError("Misaligned spatial sample could not restart decoding") from None
        target = min(frame_count - 1, max(0, int(round(requested * fps))))
        for _ in range(target + 1):
            _check_deadline(deadline)
            if not cap.grab():
                raise VideoAnalysisError("Spatial sequential sampling ended before its requested instant") from None
        ok, frame = cap.retrieve()
        if not ok or frame is None or not frame.size:
            raise VideoAnalysisError("Spatial sequential sampling could not decode a frame") from None
        actual, index, timestamp_source = _decoded_position(cap, requested, fps, frame_count, duration)
        decoding_mode = "sequential_alignment_recovery"
    _check_deadline(deadline)
    images, views = same_instant_views(frame)
    return images, {"requested_sec": float(requested), "actual_sec": float(actual),
                    "source_frame_index": index, "timestamp_source": timestamp_source,
                    "decoding_mode": decoding_mode,
                    "source_frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    "source_shape": list(frame.shape), "views": views}


def extract_video(path: str | Path, deadline: float | None = None) -> dict:
    started = time.monotonic()
    deadline = min(started + PROCESSING_TIMEOUT_SECONDS, deadline) if deadline is not None else started + PROCESSING_TIMEOUT_SECONDS
    _check_deadline(deadline)
    _check_inherited_sources()
    try:
        model = _provision.verify_ready()
    except (OSError, ValueError, RuntimeError):
        raise VideoAnalysisError("Pinned local spatial model is not ready") from None
    if (model.get("weights_sha256") != WEIGHTS_SHA256
            or model.get("projector_sha256") != FEATURE_CONFIG["projector_sha256"]
            or model.get("model_revision") != FEATURE_CONFIG["model_revision"]
            or model.get("runtime_release") != FEATURE_CONFIG["runtime_release"]):
        raise VideoAnalysisError("Local spatial model identity differs from the frozen extractor")
    cap, fps, frame_count = _open_video(path, deadline)
    windows, audit = [], []
    try:
        if frame_count < WINDOW_COUNT:
            raise VideoAnalysisError("Spatial sampling requires at least four source frames")
        duration = float(frame_count / fps)
        intervals = uniform_windows(duration)
        cv2.setNumThreads(2)
        cv2.setRNGSeed(config.SEED)
        for interval in intervals:
            requested = (interval["start_sec"] + interval["end_sec"]) / 2
            images, entry = _sample_center(cap, requested, fps, frame_count, duration, deadline)
            if audit and (entry["actual_sec"] <= audit[-1]["actual_sec"]
                          or entry["source_frame_index"] <= audit[-1]["source_frame_index"]):
                raise VideoAnalysisError("Spatial samples must have strictly increasing source positions")
            timeout = min(REQUEST_TIMEOUT_SECONDS, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Local video processing deadline exceeded")
            try:
                observation = _observations.request_observations(images, timeout=timeout)
            except (OSError, ValueError, RuntimeError):
                raise VideoAnalysisError("Local spatial encoder request failed") from None
            _check_deadline(deadline)
            values, _ = observation_features(observation)
            windows.append({"timestamp_sec": entry["actual_sec"], "features": values})
            audit.append({**entry, "observation": dict(observation), "interval": dict(interval)})
    finally:
        cap.release()
    if len(windows) != WINDOW_COUNT:
        raise VideoAnalysisError("Spatial encoder did not complete every uniform sample")
    if all(entry["observation"]["visibility"] == "unobservable" for entry in audit):
        raise VideoAnalysisError("All spatial samples are visually unobservable")
    matrix = np.stack([window["features"] for window in windows])
    summaries = {"max": matrix.max(axis=0), "mean": matrix.mean(axis=0), "std": matrix.std(axis=0)}
    clip = np.concatenate([summaries[stat] for stat in CLIP_STATISTICS]).astype(np.float32)
    if clip.shape != (len(CLIP_FEATURE_NAMES),) or not np.isfinite(clip).all():
        raise VideoAnalysisError("Spatial encoder produced invalid numeric features")
    _check_deadline(deadline)
    return {"clip_features": clip, "windows": windows, "duration_sec": duration,
            "observation_audit": audit,
            "sampling": {"sampled_frames": WINDOW_COUNT, "sampled_views": WINDOW_COUNT * FRAME_COUNT,
                         "window_count": WINDOW_COUNT, "source_fps": fps,
                         "sample_times_sec": [entry["actual_sec"] for entry in audit],
                         "coverage_note": "Four still instants; no continuous action coverage",
                         "unobservable_windows": sum(entry["observation"]["visibility"] == "unobservable" for entry in audit),
                         "elapsed_sec": time.monotonic() - started, "feature_config": dict(FEATURE_CONFIG)}}


def _load_model(model_path):
    path = Path(model_path or os.environ.get("DATATHON_VIDEO_MODEL", str(
        config.DATA_PROC / "video-qwen-spatial-v3/models/baseline.joblib")))
    if not path.is_file():
        raise VideoAnalysisError("Local spatial classifier is missing; train before analysis")
    artifact = joblib.load(path)  # Trusted operator artifact only, never uploads.
    if (not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION
            or artifact.get("engine") != ENGINE):
        raise VideoAnalysisError("Unsupported local spatial model schema or engine")
    names = artifact.get("feature_names", {})
    if (not isinstance(names, dict) or tuple(names.get("clip", ())) != CLIP_FEATURE_NAMES
            or tuple(names.get("window", ())) != WINDOW_FEATURE_NAMES):
        raise VideoAnalysisError("Spatial model feature names differ from the extractor")
    provenance = get_encoder_provenance()
    if (artifact.get("feature_config") != FEATURE_CONFIG
            or any(artifact.get(key) != provenance[key] for key in ("source_sha256", "weights_sha256", "inherited_sources"))):
        raise VideoAnalysisError("Spatial model provenance differs from the extractor")
    _check_inherited_sources()
    if not hasattr(artifact.get("clip_model"), "predict_proba"):
        raise VideoAnalysisError("Spatial clip model does not support probabilities")
    if not hasattr(artifact.get("window_model"), "predict_proba"):
        raise VideoAnalysisError("Spatial window model does not support probabilities")
    if type(artifact.get("threshold")) not in (int, float) or artifact["threshold"] != .5:
        raise VideoAnalysisError("Spatial model requires the fixed 0.5 threshold")
    return artifact


def analyze_video(path: str | Path, model_path: str | Path | None = None,
                  deadline: float | None = None) -> dict:
    started = time.monotonic()
    deadline = min(started + PROCESSING_TIMEOUT_SECONDS, deadline) if deadline is not None else started + PROCESSING_TIMEOUT_SECONDS
    _check_deadline(deadline)
    artifact = _load_model(model_path)
    extracted = extract_video(path, deadline)
    probability = float(_positive_probability(artifact["clip_model"], extracted["clip_features"].reshape(1, -1))[0])
    _check_deadline(deadline)
    if probability < .5:
        return {"verdict": "no_violation", "violations": []}
    windows = extracted["windows"]
    scores = _positive_probability(artifact["window_model"], np.stack([window["features"] for window in windows]))
    timestamp = float(windows[int(np.argmax(scores))]["timestamp_sec"])
    frames = evidence_frames(path, [timestamp], deadline)
    _check_deadline(deadline)
    return {"verdict": "violation", "violations": [{
        "type": "Визуальный признак для проверки", "timecode_sec": frames[0]["timestamp_sec"],
        "description": (
            "Локальная модель Qwen3-VL выделила сочетание автоматических пространственных "
            "наблюдений, похожее на положительные примеры. Это неподтверждённые наблюдения "
            "отдельных кадров, а не проверенные метки действий. Фрагмент предлагается "
            "для проверки человеком. Тип нарушения и нормативный источник этой моделью "
            "не установлены. Это не юридическое или дисциплинарное заключение."
        ), "evidence_frames": frames}]}
