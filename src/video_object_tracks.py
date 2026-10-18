"""Offline object detections and anonymous geometry tracks, not action truth.

Frames are sampled uniformly without labels or organizer event timestamps.
Private model detections remain numeric local audit material. Anonymous track
IDs exist only within one decoded clip and never identify a person.
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

from scripts import fetch_video_objects as _assets
from src import config
from src import video_objects as _objects
from src.video_analysis import (VideoAnalysisError, _check_deadline, _open_video,
                                _positive_probability, evidence_frames)


SCHEMA_VERSION = 1
ENGINE = "object-tracks"
WEIGHTS_SHA256 = _assets.WEIGHTS_SHA256
CLASS_IDS = tuple(_objects.CLASS_IDS)
CLASS_NAMES = dict(_objects.CLASS_NAMES)
OBJECT_CLASS_IDS = tuple(value for value in CLASS_IDS if value != 0)
SAMPLE_FPS = 2.0
MAX_SAMPLES = 300
WINDOW_SECONDS = 2.0
PROCESSING_TIMEOUT_SECONDS = 240.0
TRACK_IOU = .3
TRACK_MAX_GAP_SECONDS = 1.0
CLIP_STATISTICS = ("max", "mean", "std")
INHERITED_SOURCES = {
    relative: hashlib.sha256((config.ROOT / relative).read_bytes()).hexdigest()
    for relative in ("src/video_analysis.py", "src/video_objects.py", "scripts/fetch_video_objects.py")
}
FEATURE_CONFIG = {
    "schema_version": SCHEMA_VERSION, "engine": ENGINE,
    "weights_sha256": WEIGHTS_SHA256, "model_id": _assets.MODEL_ID,
    "model_revision": _assets.SOURCE_COMMIT, "license_sha256": _assets.LICENSE_SHA256,
    "detector": dict(_objects.FEATURE_CONFIG), "inherited_sources": dict(INHERITED_SOURCES),
    "class_ids": list(CLASS_IDS), "sample_fps": SAMPLE_FPS, "max_samples": MAX_SAMPLES,
    "sample_selection": "requested arange(0,duration,0.5); first sequential source frame at/after request",
    "sample_cap_policy": "fail if more than 300 requested samples; never vary sampling rate",
    "window_seconds": WINDOW_SECONDS, "window_assignment": "floor(actual_timestamp/2)",
    "count_statistics": ["mean_count", "max_count", "max_score", "mean_area"],
    "relations": "each nonperson detection to nearest person center; distance / sqrt(2)",
    "relation_statistics": ["mean_distance", "min_distance", "missing_frame_fraction"],
    "missing_geometry": "distance zero with explicit missing fraction; count/statistics zero when no detection",
    "track_association": "same-class greedy descending IoU; stable track-id/detection-index tie break",
    "track_iou": TRACK_IOU, "track_max_gap_seconds": TRACK_MAX_GAP_SECONDS,
    "track_statistics": ["unique_tracks", "mean_visible_span_seconds", "mean_center_speed"],
    "visible_span": "min(2 seconds,last-first+0.5) within each window only",
    "center_speed": "center displacement / sqrt(2) / seconds; within-window consecutive observations only",
    "clip_statistics": list(CLIP_STATISTICS), "metadata_features": False,
    "alignment_tolerance": "max(0.25 seconds,2/source_fps)",
    "timestamp_fallback": "source_frame_index/source_fps when decoder timestamp invalid or nonincreasing",
    "strict_increasing_samples": True, "processing_timeout_seconds": PROCESSING_TIMEOUT_SECONDS,
    "seed": config.SEED, "inference_network_required": False,
}
WINDOW_FEATURE_NAMES = (
    *(f"class_{identifier}_{stat}" for identifier in CLASS_IDS
      for stat in ("mean_count", "max_count", "max_score", "mean_area")),
    *(f"class_{identifier}_person_{stat}" for identifier in OBJECT_CLASS_IDS
      for stat in ("mean_distance", "min_distance", "missing_frame_fraction")),
    *(f"class_{identifier}_track_{stat}" for identifier in CLASS_IDS
      for stat in ("count", "mean_visible_span_seconds", "mean_center_speed")),
)
CLIP_FEATURE_NAMES = tuple(f"{name}_{stat}" for stat in CLIP_STATISTICS for name in WINDOW_FEATURE_NAMES)


def get_encoder_provenance():
    return {"engine": ENGINE, "engine_module": "src.video_object_tracks",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "weights_sha256": WEIGHTS_SHA256, "inherited_sources": dict(INHERITED_SOURCES)}


def _check_inherited_sources():
    for relative, expected in INHERITED_SOURCES.items():
        if hashlib.sha256((config.ROOT / relative).read_bytes()).hexdigest() != expected:
            raise VideoAnalysisError("Object feature helper source changed; use a fresh process and run")


def requested_times(duration):
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or not 0 < duration <= 300):
        raise VideoAnalysisError("Object sampling requires a supported video duration")
    times = np.arange(0., float(duration), 1. / SAMPLE_FPS)
    if len(times) > MAX_SAMPLES:
        raise VideoAnalysisError("Video exceeds the fixed 300-sample object analysis budget")
    return times


def validate_detections(detections):
    if not isinstance(detections, list) or len(detections) > 100:
        raise VideoAnalysisError("Object detector returned invalid bounded detections")
    result = []
    for detection in detections:
        if (not isinstance(detection, dict)
                or set(detection) != {"class_id", "score", "box_xyxy"}
                or type(detection["class_id"]) is not int or detection["class_id"] not in CLASS_IDS):
            raise VideoAnalysisError("Object detector returned an unsupported detection schema")
        score, box = detection["score"], detection["box_xyxy"]
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1
                or not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or not 0 <= value <= 1 for value in box)
                or not (box[0] < box[2] and box[1] < box[3])):
            raise VideoAnalysisError("Object detector returned invalid normalized geometry")
        result.append({"class_id": detection["class_id"], "score": float(score),
                       "box_xyxy": list(map(float, box))})
    return sorted(result, key=lambda item: (item["class_id"], -item["score"], item["box_xyxy"]))


def _center(box):
    return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _iou(first, second):
    width = max(0., min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0., min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return intersection / (area(first) + area(second) - intersection)


class AnonymousTracker:
    """Short same-class IoU association; no appearance, identity or future data."""

    def __init__(self):
        self.active = {}
        self.next_id = 0
        self.last_timestamp = None

    def update(self, detections, timestamp):
        if (not math.isfinite(timestamp) or timestamp < 0
                or self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise VideoAnalysisError("Anonymous tracks require increasing source timestamps")
        self.last_timestamp = timestamp
        self.active = {key: item for key, item in self.active.items()
                       if timestamp - item["timestamp_sec"] <= TRACK_MAX_GAP_SECONDS + 1e-8}
        candidates = []
        for index, detection in enumerate(detections):
            for track_id, previous in self.active.items():
                if detection["class_id"] == previous["class_id"]:
                    overlap = _iou(detection["box_xyxy"], previous["box_xyxy"])
                    if overlap >= TRACK_IOU:
                        candidates.append((-overlap, track_id, index))
        assignments, used_tracks = {}, set()
        for _, track_id, index in sorted(candidates):
            if index not in assignments and track_id not in used_tracks:
                assignments[index] = track_id
                used_tracks.add(track_id)
        output = []
        for index, detection in enumerate(detections):
            track_id = assignments.get(index)
            if track_id is None:
                track_id, self.next_id = self.next_id, self.next_id + 1
            self.active[track_id] = {**detection, "timestamp_sec": timestamp}
            output.append({**detection, "track_id": track_id})
        return output


def window_features(rows):
    """Summaries use only observations supplied for this two-second window."""
    if not rows:
        raise VideoAnalysisError("Object window requires sampled source frames")
    values = []
    for identifier in CLASS_IDS:
        per_frame = [[detection for detection in row["detections"] if detection["class_id"] == identifier]
                     for row in rows]
        counts = [len(items) for items in per_frame]
        detections = [item for items in per_frame for item in items]
        areas = [(item["box_xyxy"][2] - item["box_xyxy"][0]) *
                 (item["box_xyxy"][3] - item["box_xyxy"][1]) for item in detections]
        values.extend([float(np.mean(counts)), float(max(counts)),
                       max((item["score"] for item in detections), default=0.), float(np.mean(areas)) if areas else 0.])
    for identifier in OBJECT_CLASS_IDS:
        distances, missing = [], 0
        for row in rows:
            people = [_center(item["box_xyxy"]) for item in row["detections"] if item["class_id"] == 0]
            objects = [_center(item["box_xyxy"]) for item in row["detections"] if item["class_id"] == identifier]
            if not people or not objects:
                missing += 1
                continue
            distances.extend(float(min(np.linalg.norm(obj - person) for person in people) / math.sqrt(2))
                             for obj in objects)
        values.extend([float(np.mean(distances)) if distances else 0., min(distances, default=0.), missing / len(rows)])
    for identifier in CLASS_IDS:
        tracks = {}
        for row in rows:
            for item in row["detections"]:
                if item["class_id"] == identifier:
                    tracks.setdefault(item["track_id"], []).append((row["timestamp_sec"], _center(item["box_xyxy"])))
        spans, speeds = [], []
        for observations in tracks.values():
            spans.append(min(WINDOW_SECONDS, observations[-1][0] - observations[0][0] + 1 / SAMPLE_FPS))
            speeds.extend(float(np.linalg.norm(after[1] - before[1]) / math.sqrt(2) / (after[0] - before[0]))
                          for before, after in zip(observations, observations[1:]))
        values.extend([float(len(tracks)), float(np.mean(spans)) if spans else 0., float(np.mean(speeds)) if speeds else 0.])
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(WINDOW_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise VideoAnalysisError("Object window features are invalid")
    return result


def summarize_windows(audit):
    if not audit:
        raise VideoAnalysisError("Object extraction produced no sampled frames")
    times = np.asarray([item["timestamp_sec"] for item in audit])
    windows = []
    for identifier in np.unique(np.floor(times / WINDOW_SECONDS).astype(int)):
        rows = [item for item in audit if int(item["timestamp_sec"] // WINDOW_SECONDS) == identifier]
        windows.append({"timestamp_sec": float(np.median([item["timestamp_sec"] for item in rows])),
                        "features": window_features(rows)})
    matrix = np.stack([item["features"] for item in windows])
    statistics = {"max": matrix.max(axis=0), "mean": matrix.mean(axis=0), "std": matrix.std(axis=0)}
    return windows, np.concatenate([statistics[stat] for stat in CLIP_STATISTICS]).astype(np.float32)


def extract_video(path: str | Path, deadline: float | None = None) -> dict:
    started = time.monotonic()
    deadline = min(started + PROCESSING_TIMEOUT_SECONDS, deadline) if deadline is not None else started + PROCESSING_TIMEOUT_SECONDS
    _check_deadline(deadline)
    _check_inherited_sources()
    try:
        ready = _assets.verify_ready()
    except (OSError, ValueError, RuntimeError):
        raise VideoAnalysisError("Pinned local object detector is not ready") from None
    if ready.get("weights_sha256") != WEIGHTS_SHA256 or ready.get("model_revision") != _assets.SOURCE_COMMIT:
        raise VideoAnalysisError("Object model identity differs from the frozen extractor")
    _check_deadline(deadline)
    cap, fps, frame_count = _open_video(path, deadline)
    audit, tracker, timestamp_fallbacks = [], AnonymousTracker(), 0
    decoded = 0
    last_source_time = -1.
    try:
        if frame_count <= 0:
            raise VideoAnalysisError("Object sampling requires complete source frame metadata")
        duration = float(frame_count / fps)
        requested = requested_times(duration)
        detector = _objects.ObjectDetector()
        cv2.setRNGSeed(config.SEED)
        for frame_index in range(frame_count):
            _check_deadline(deadline)
            if not cap.grab():
                raise VideoAnalysisError("Object video ended before its declared source frame count")
            decoded += 1
            source_time = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            timestamp_source = "decoder"
            if (not math.isfinite(source_time) or source_time < 0
                    or frame_index > 0 and source_time <= last_source_time):
                source_time, timestamp_source = frame_index / fps, "source_frame_index_over_fps"
                timestamp_fallbacks += 1
            if not math.isfinite(source_time) or source_time <= last_source_time or source_time >= duration:
                raise VideoAnalysisError("Object video timestamps are inconsistent with its duration")
            last_source_time = source_time
            if len(audit) >= len(requested) or source_time + 1e-8 < requested[len(audit)]:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None or not frame.size:
                raise VideoAnalysisError("A sampled object frame could not be decoded")
            raw_index = float(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if (not math.isfinite(raw_index) or abs(raw_index - frame_index) > 1e-5
                    or abs(source_time - requested[len(audit)]) > max(.25, 2 / fps)
                    or audit and (source_time <= audit[-1]["timestamp_sec"] or frame_index <= audit[-1]["source_frame_index"])):
                raise VideoAnalysisError("Object sample is not aligned with its requested source position")
            _check_deadline(deadline)
            detections = validate_detections(detector.detect(frame, deadline=deadline))
            _check_deadline(deadline)
            audit.append({"timestamp_sec": float(source_time), "requested_sec": float(requested[len(audit)]),
                          "source_frame_index": frame_index, "timestamp_source": timestamp_source,
                          "source_shape": list(frame.shape), "source_frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                          "detections": tracker.update(detections, float(source_time))})
    finally:
        cap.release()
    if len(audit) != len(requested):
        raise VideoAnalysisError("Object video did not provide every requested sample")
    windows, clip = summarize_windows(audit)
    _check_deadline(deadline)
    return {"clip_features": clip, "windows": windows, "duration_sec": duration, "object_audit": audit,
            "sampling": {"sampled_frames": len(audit), "decoded_frames": decoded, "source_fps": fps,
                         "timestamp_fallbacks": timestamp_fallbacks,
                         "sample_times_sec": [item["timestamp_sec"] for item in audit],
                         "coverage_note": "Uniform still detections and short anonymous box tracks; not verified actions",
                         "elapsed_sec": time.monotonic() - started, "feature_config": dict(FEATURE_CONFIG)}}


def _load_model(model_path):
    path = Path(model_path or os.environ.get("DATATHON_VIDEO_MODEL", str(
        config.DATA_PROC / "video-object-tracks/models/baseline.joblib")))
    if not path.is_file():
        raise VideoAnalysisError("Local object-track classifier is missing; train before analysis")
    artifact = joblib.load(path)  # Trusted local artifacts only; never uploads.
    if (not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION
            or artifact.get("engine") != ENGINE):
        raise VideoAnalysisError("Unsupported local object-track model schema or engine")
    names = artifact.get("feature_names", {})
    if (not isinstance(names, dict) or tuple(names.get("clip", ())) != CLIP_FEATURE_NAMES
            or tuple(names.get("window", ())) != WINDOW_FEATURE_NAMES):
        raise VideoAnalysisError("Object-track model feature names differ from the extractor")
    provenance = get_encoder_provenance()
    if (artifact.get("feature_config") != FEATURE_CONFIG
            or any(artifact.get(key) != provenance[key] for key in ("source_sha256", "weights_sha256", "inherited_sources"))):
        raise VideoAnalysisError("Object-track model provenance differs from the extractor")
    _check_inherited_sources()
    if not hasattr(artifact.get("clip_model"), "predict_proba") or not hasattr(artifact.get("window_model"), "predict_proba"):
        raise VideoAnalysisError("Object-track models must support probabilities")
    if type(artifact.get("threshold")) not in (int, float) or artifact["threshold"] != .5:
        raise VideoAnalysisError("Object-track model requires the fixed 0.5 threshold")
    return artifact


def analyze_video(path: str | Path, model_path: str | Path | None = None, deadline: float | None = None) -> dict:
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
    scores = _positive_probability(artifact["window_model"], np.stack([item["features"] for item in windows]))
    frames = evidence_frames(path, [float(windows[int(np.argmax(scores))]["timestamp_sec"])], deadline)
    _check_deadline(deadline)
    return {"verdict": "violation", "violations": [{
        "type": "Визуальный признак для проверки", "timecode_sec": frames[0]["timestamp_sec"],
        "description": (
            "Локальная модель YOLOX-S выделила сочетание автоматических наблюдений "
            "объектов и коротких анонимных траекторий, похожее на положительные примеры. "
            "Это предположение модели для проверки человеком, а не подтверждённая метка "
            "действия. Тип нарушения, принадлежность объектов, намерение и нормативное "
            "основание этой моделью не установлены. Это не юридическое или дисциплинарное заключение."
        ), "evidence_frames": frames}]}
