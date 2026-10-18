"""Check private object caches without decoding/inference; print aggregates only.

Only trusted operator-produced joblib caches are accepted. This audit checks
recorded provenance and numeric consistency, not object/action correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re

import joblib
import numpy as np

from src import config
from src import video_object_tracks as engine
from src.video_eval import load_protocol
from src.video_train import verify_feature_record

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROW_KEYS = {"timestamp_sec", "requested_sec", "source_frame_index", "timestamp_source",
             "source_shape", "source_frame_sha256", "detections"}
_DETECTION_KEYS = {"class_id", "score", "box_xyxy", "track_id"}
_LIMITATIONS = [
    "Recorded source hashes checked against current files; no separate source snapshots exist in this run.",
    "Frame hash format and timing metadata checked; frame contents were not decoded or independently rehashed.",
    "Object, action, identity, ownership and regulatory correctness are not verified.",
    "NMS cannot be independently reconstructed from cached clipped boxes without the raw network output.",
    "No models were called, no videos were opened, and no holdout cache was loaded.",
]


class AuditError(ValueError):
    """A static, aggregate-safe audit failure without private record details."""


def _require(condition, message):
    if not condition:
        raise AuditError(message)


def _number(value, lower=0., upper=math.inf):
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
            and math.isfinite(value) and lower <= value <= upper)


def _integer(value, lower=0, upper=math.inf):
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, Integral)
            and lower <= value <= upper)


def _cache_paths(run, expected):
    cache = run / "cache"
    _require(not cache.is_symlink(), "Object cache directory must not be a symlink")
    if not cache.exists():
        return []
    _require(cache.is_dir(), "Object cache location must be a directory")
    paths = list(cache.iterdir())
    _require(all(path.is_file() and not path.is_symlink() and path.suffix == ".joblib"
                 for path in paths), "Object cache contains unsupported entries")
    _require({path.stem for path in paths} <= expected,
             "Object cache contains a reserved or unknown group; nothing outside development may be loaded")
    return sorted(paths)


def _audit_record(record, provenance, expected_duration):
    _require(isinstance(record, dict), "Object cache record must be a mapping")
    try:
        verify_feature_record(record, provenance, engine.FEATURE_CONFIG)
    except (KeyError, TypeError, ValueError):
        raise AuditError("Object cache source/configuration provenance differs from the frozen extractor") from None
    duration, extraction = record.get("duration_sec"), record.get("extraction_sec")
    _require(_number(duration, 0., 300.) and duration > 0
             and _number(extraction, 0., engine.PROCESSING_TIMEOUT_SECONDS + 1.),
             "Object cache duration or extraction runtime is invalid")
    sampling = record.get("sampling")
    _require(isinstance(sampling, dict) and sampling.get("feature_config") == engine.FEATURE_CONFIG,
             "Object sampling configuration differs from the frozen extractor")
    fps = sampling.get("source_fps")
    decoded = sampling.get("decoded_frames")
    fallbacks = sampling.get("timestamp_fallbacks")
    elapsed = sampling.get("elapsed_sec")
    _require(_number(fps, 0., 240.) and fps > 0 and _integer(decoded, 1)
             and _integer(fallbacks, 0, decoded) and _number(elapsed, 0., extraction + .001),
             "Object source frame metadata or extraction timing is invalid")
    _require(decoded == round(duration * fps)
             and abs(duration - expected_duration) <= max(.25, 2 / fps),
             "Object duration differs from source count or frozen inventory")
    requested = engine.requested_times(duration)
    audit = record.get("object_audit")
    _require(isinstance(audit, list) and len(audit) == len(requested)
             and _integer(sampling.get("sampled_frames"), 1)
             and sampling.get("sampled_frames") == len(audit),
             "Object cache does not contain every requested two-FPS sample")
    tracker = engine.AnonymousTracker()
    times, indices, reconstructed = [], [], []
    total_detections = sampled_fallbacks = 0
    for wanted, row in zip(requested, audit, strict=True):
        _require(isinstance(row, dict) and set(row) == _ROW_KEYS,
                 "Object sample schema is invalid")
        actual, request, index = row["timestamp_sec"], row["requested_sec"], row["source_frame_index"]
        _require(_number(actual, 0., duration) and actual < duration
                 and _number(request, 0., duration) and request == wanted
                 and actual + 1e-8 >= request and actual - request <= max(.25, 2 / fps)
                 and _integer(index, 0, decoded - 1), "Object sample timing or source index is invalid")
        _require(not times or actual > times[-1] and index > indices[-1],
                 "Object samples are not strictly chronological")
        source = row["timestamp_source"]
        _require(source in {"decoder", "source_frame_index_over_fps"},
                 "Object sample timestamp provenance is invalid")
        if source == "source_frame_index_over_fps":
            _require(abs(actual - index / fps) < 1e-7,
                     "Object timestamp fallback does not equal its source frame position")
            sampled_fallbacks += 1
        shape = row["source_shape"]
        _require(isinstance(shape, (list, tuple)) and len(shape) == 3 and shape[2] == 3
                 and all(_integer(value, 1) for value in shape), "Object source dimensions are invalid")
        _require(isinstance(row["source_frame_sha256"], str)
                 and _SHA256.fullmatch(row["source_frame_sha256"]) is not None,
                 "Object sample frame-hash provenance is incomplete")
        detections = row["detections"]
        _require(isinstance(detections, list)
                 and all(isinstance(item, dict) and set(item) == _DETECTION_KEYS
                         and _integer(item["track_id"]) for item in detections),
                 "Object anonymous-track schema is invalid")
        try:
            validated = engine.validate_detections([
                {key: item[key] for key in ("class_id", "score", "box_xyxy")}
                for item in detections])
        except (KeyError, TypeError, ValueError):
            raise AuditError("Object cache contains invalid classes, scores or normalized boxes") from None
        _require(all(item["score"] >= engine._objects.CONFIDENCE_THRESHOLD - 1e-7 for item in validated),
                 "Object cache contains a score below the frozen acceptance threshold")
        associated = tracker.update(validated, actual)
        _require(associated == detections,
                 "Object tracks differ from deterministic same-class IoU association")
        reconstructed.append({**row, "detections": associated})
        total_detections += len(detections)
        times.append(actual)
        indices.append(index)
    _require(sampled_fallbacks <= fallbacks
             and sampling.get("sample_times_sec") == times,
             "Object sample timing summary differs from its private records")
    expected_windows, expected_clip = engine.summarize_windows(reconstructed)
    windows = record.get("windows")
    _require(isinstance(windows, list) and len(windows) == len(expected_windows),
             "Object two-second window coverage is inconsistent")
    for actual, expected in zip(windows, expected_windows, strict=True):
        _require(isinstance(actual, dict) and set(actual) == {"timestamp_sec", "features"},
                 "Object window schema is invalid")
        vector = np.asarray(actual["features"])
        _require(vector.shape == (57,) and np.isfinite(vector).all()
                 and actual["timestamp_sec"] == expected["timestamp_sec"]
                 and np.array_equal(vector, expected["features"]),
                 "Object window features differ from recomputed private observations")
    clip = np.asarray(record.get("clip_features"))
    _require(clip.shape == (171,) and np.isfinite(clip).all()
             and np.array_equal(clip, expected_clip), "Object clip pooling differs from its private windows")
    return {"clip": clip, "samples": len(audit), "windows": len(windows),
            "detections": total_detections, "tracks": tracker.next_id, "fallbacks": sampled_fallbacks}


def _completed_reports(run, protocol, expected, provenance, progress):
    required = (run / "development.json", run / "model_manifest.json")
    present = sum(path.is_file() for path in required)
    if present != len(required):
        _require(progress, "Complete object audit requires development and model manifests")
    for path in required:
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        try:
            verify_feature_record(report, provenance, engine.FEATURE_CONFIG)
        except (KeyError, TypeError, ValueError):
            raise AuditError("Completed object report provenance differs from the extractor") from None
        _require(report.get("protocol_fingerprint") == protocol["fingerprint"]
                 and Path(report.get("cache_dir", "")).resolve() == (run / "cache").resolve(),
                 "Completed object report protocol/cache location differs from this run")
        if path.name == "development.json":
            groups = report.get("group_ids", [])
            _require(report.get("holdout_evaluated") is False and report.get("seed") == config.SEED
                     and report.get("threshold") == .5, "Object development policy differs from the fixed trial")
        else:
            groups = report.get("training_groups", [])
            _require(report.get("stage") == "development-only", "Object model stage is not development-only")
            model_path = run / "models/baseline.joblib"
            _require(model_path.is_file() and not model_path.is_symlink()
                     and report.get("model_sha256") == hashlib.sha256(model_path.read_bytes()).hexdigest(),
                     "Object model bytes differ from the completed manifest")
        _require(len(groups) == len(expected) and set(groups) == expected,
                 "Completed object report membership differs from the development partition")
    return present


def audit(run_dir, protocol_path=None, *, progress=False, write_report=True):
    """Full mode requires completion; progress mode never claims a full audit."""
    run = Path(run_dir).resolve()
    _require(not run.is_relative_to(config.DATA_RAW.resolve()), "Object audit requires derived artifacts")
    protocol = load_protocol(protocol_path or config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    expected = set(protocol["development_group_ids"])
    heldout = set(protocol["holdout_group_ids"])
    _require(len(expected) == 32 and len(heldout) == 8 and not expected & heldout,
             "Object audit requires the frozen 32-development/eight-reserved protocol")
    paths = _cache_paths(run, expected)
    _require(progress or len(paths) == len(expected), "Complete object audit requires all development caches")
    engine._check_inherited_sources()
    ready = engine._assets.verify_ready()  # Hashes local assets; never loads or calls a model.
    _require(ready["weights_sha256"] == engine.WEIGHTS_SHA256
             and ready["model_revision"] == engine.FEATURE_CONFIG["model_revision"],
             "Object asset provenance differs from the frozen extractor")
    _require(len(engine.CLIP_FEATURE_NAMES) == 171 and len(engine.WINDOW_FEATURE_NAMES) == 57,
             "Object extractor dimensions differ from the fixed trial")
    _require(engine.SAMPLE_FPS == 2. and engine.WINDOW_SECONDS == 2. and engine.MAX_SAMPLES == 300,
             "Object extractor sampling policy differs from the fixed trial")
    provenance = engine.get_encoder_provenance()
    reports = _completed_reports(run, protocol, expected, provenance, progress)
    durations = {group["group_id"]: group["duration_sec"] for group in protocol["groups"]
                 if group["group_id"] in expected}
    records = []
    for path in paths:
        # Membership/symlinks were checked before unpickling. Operator caches only.
        record = joblib.load(path)
        records.append(_audit_record(record, provenance, durations[path.stem]))
    if not progress:
        _require({path.stem for path in _cache_paths(run, expected)} == expected,
                 "Object cache membership changed during the full audit")
    matrix = np.stack([record["clip"] for record in records]) if records else np.empty((0, 171))
    result = {
        "mode": "progress" if progress else "complete",
        "complete_integrity_audit": not progress,
        "expected_development_groups": len(expected), "cached_development_groups": len(records),
        "missing_development_groups": len(expected) - len(records), "holdout_groups_used": 0,
        "completed_reports_present": reports, "source_hashes_verified": 1 + len(engine.INHERITED_SOURCES),
        "clip_feature_count": 171, "window_feature_count": 57,
        "sampled_instants": sum(item["samples"] for item in records),
        "two_second_windows": sum(item["windows"] for item in records),
        "accepted_object_predictions": sum(item["detections"] for item in records),
        "anonymous_track_segments": sum(item["tracks"] for item in records),
        "sampled_timestamp_fallbacks": sum(item["fallbacks"] for item in records),
        "unique_clip_vectors": len(np.unique(matrix, axis=0)) if len(matrix) else 0,
        "nonconstant_clip_features": int(np.count_nonzero(np.ptp(matrix, axis=0))) if len(matrix) else 0,
        "semantic_accuracy_verified": False, "operational_integrity_only": True,
        "limitations": list(_LIMITATIONS),
    }
    if write_report and not progress:
        (run / "object_observation_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "video-object-tracks")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--progress", action="store_true", help="Validate available caches without a completion claim")
    args = parser.parse_args(argv)
    try:
        result = audit(args.run_dir, args.protocol, progress=args.progress)
    except Exception:
        raise SystemExit("Private object audit failed; no media, observations or identifiers emitted") from None
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
