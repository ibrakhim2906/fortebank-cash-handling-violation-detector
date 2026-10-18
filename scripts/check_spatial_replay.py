"""Diagnose frozen spatial replay; only aggregate counts reach stdout.

Default inspection reads provenance only. --sample-only decodes the fixed four
centers without VLM calls. --run adds exactly two sequential frozen extractions.
Private samples/observations/request hashes remain in exclusive mode-600 files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import uuid

import joblib
import numpy as np

from src import config
from src import video_qwen_spatial_v3 as engine
from src.video_eval import load_protocol, protocol_groups
from src.video_train import verify_feature_record

_IDENTITY_KEYS = ("requested_sec", "actual_sec", "source_frame_index", "timestamp_source",
                  "source_frame_sha256", "source_shape", "views")
_LIMITATIONS = [
    "Numeric reproducibility and source/view identity only; no semantic or regulatory verification.",
    "The original cache did not record request-body hashes; historical body equality cannot be proved.",
    "Runtime files/model assets are verified on disk; active-process attestation is a separate check.",
    "No threshold, prompt, feature configuration, training or holdout evaluation is changed.",
]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _private(path, value, *, json_output=False):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        if json_output:
            stream.write((json.dumps(value, indent=2) + "\n").encode())
        else:
            joblib.dump(value, stream)


def _validate_record(record, *, cached=False):
    if cached:
        verify_feature_record(record, engine.get_encoder_provenance(), engine.FEATURE_CONFIG)
    clip = np.asarray(record["clip_features"])
    windows, audit = record["windows"], record["observation_audit"]
    _require(clip.shape == (len(engine.CLIP_FEATURE_NAMES),) and np.isfinite(clip).all()
             and len(windows) == len(audit) == engine.WINDOW_COUNT,
             "Spatial replay feature dimensions are invalid")
    _require(record["sampling"]["feature_config"] == engine.FEATURE_CONFIG,
             "Spatial replay sampling configuration changed")
    values = []
    for window, sample in zip(windows, audit, strict=True):
        expected, _ = engine.observation_features(sample["observation"])
        vector = np.asarray(window["features"])
        _require(vector.shape == expected.shape and np.array_equal(vector, expected)
                 and window["timestamp_sec"] == sample["actual_sec"],
                 "Spatial replay window does not match its private observation")
        _require(all(key in sample for key in _IDENTITY_KEYS), "Spatial replay sample provenance is incomplete")
        values.append(vector)
    matrix = np.stack(values)
    statistics = {"max": matrix.max(axis=0), "mean": matrix.mean(axis=0), "std": matrix.std(axis=0)}
    _require(np.array_equal(clip, np.concatenate([statistics[key] for key in engine.CLIP_STATISTICS])),
             "Spatial replay pooled features differ from its windows")


def identity_difference(first, second):
    one, two = first["observation_audit"], second["observation_audit"]
    _require(len(one) == len(two) == engine.WINDOW_COUNT, "Spatial replay sample count changed")
    return {"differing_sample_" + key: sum(a[key] != b[key] for a, b in zip(one, two, strict=True))
            for key in _IDENTITY_KEYS}


def _same_identity(difference):
    return not any(difference.values())


@contextmanager
def _capture_requests(*, observations=None):
    """Capture hashes, preserving the frozen serialized body and real transport."""
    client = engine._observations
    original_builder = client.build_opener
    captures = []
    pending = iter(observations) if observations is not None else None

    class CapturingOpener:
        def __init__(self, delegate):
            self.delegate = delegate

        def open(self, request, *args, **kwargs):
            captures.append({"body_sha256": hashlib.sha256(request.data).hexdigest(),
                             "body_bytes": len(request.data)})
            if pending is not None:
                value = next(pending)
                envelope = {"choices": [{"message": {"content": json.dumps(value)}}]}
                return io.BytesIO(json.dumps(envelope).encode())
            return self.delegate.open(request, *args, **kwargs)

    def build(*handlers):
        return CapturingOpener(None if pending is not None else original_builder(*handlers))

    with patch.object(client, "build_opener", build):
        yield captures


def _decode_probe(source, cached):
    started = time.monotonic()
    deadline = started + 60
    cap, fps, count = engine._open_video(source, deadline)
    rows = []
    try:
        duration = float(count / fps)
        _require(count >= engine.WINDOW_COUNT, "Spatial replay source metadata is incomplete")
        engine.cv2.setNumThreads(2)
        engine.cv2.setRNGSeed(config.SEED)
        cached_observations = [row["observation"] for row in cached["observation_audit"]]
        with _capture_requests(observations=cached_observations) as requests:
            for interval in engine.uniform_windows(duration):
                requested = (interval["start_sec"] + interval["end_sec"]) / 2
                images, entry = engine._sample_center(cap, requested, fps, count, duration, deadline)
                # This invokes frozen serialization with a fake transport, not a model.
                engine._observations.request_observations(images, timeout=45)
                rows.append({**entry, "interval": interval})
    finally:
        cap.release()
    return {"observation_audit": rows, "duration_sec": duration,
            "request_audit": requests, "elapsed_sec": time.monotonic() - started}


def _prepare(run, protocol_path):
    protocol = load_protocol(protocol_path)
    groups = protocol_groups(protocol, "development")
    _require(len(groups) == 32 and len(protocol["holdout_group_ids"]) == 8,
             "Spatial replay requires the frozen development/holdout protocol")
    # Length then content-hash tie break only; labels/annotation times are unused.
    chosen = sorted(groups, key=lambda row: (-row["duration_sec"], row["group_id"]))[0]
    _require(chosen["group_id"] not in protocol["holdout_group_ids"],
             "Reserved footage cannot enter spatial replay")
    source = config.ROOT / chosen["path"]
    _require(source.is_file() and digest(source) == chosen["group_id"],
             "Spatial replay source differs from the frozen content group")
    engine._check_inherited_sources()
    provenance = engine.get_encoder_provenance()
    report = json.loads((run / "development.json").read_text())
    verify_feature_record(report, provenance, engine.FEATURE_CONFIG)
    _require(report["protocol_fingerprint"] == protocol["fingerprint"]
             and report["holdout_evaluated"] is False
             and report["group_ids"] == protocol["development_group_ids"]
             and report["seed"] == config.SEED and report["threshold"] == .5,
             "Spatial replay run policy differs from the frozen comparison")
    cache_path = run / "cache" / (chosen["group_id"] + ".joblib")
    _require(not cache_path.is_symlink(), "Spatial replay cache must not be a symlink")
    cached = joblib.load(cache_path)  # Trusted local operator artifacts only.
    _validate_record(cached, cached=True)
    model_path = run / "models/baseline.joblib"
    manifest = json.loads((run / "model_manifest.json").read_text())
    verify_feature_record(manifest, provenance, engine.FEATURE_CONFIG)
    _require(manifest["model_sha256"] == digest(model_path)
             and manifest["protocol_fingerprint"] == protocol["fingerprint"],
             "Spatial replay classifier manifest differs from the saved model")
    model = engine._load_model(model_path)
    _require(set(model["training_groups"]) == set(protocol["development_group_ids"])
             and len(model["training_groups"]) == len(groups)
             and model["protocol_fingerprint"] == protocol["fingerprint"],
             "Spatial replay model was not trained on the permitted development groups")
    ready = engine._provision.verify_ready()
    _require(ready["weights_sha256"] == engine.WEIGHTS_SHA256
             and ready["projector_sha256"] == engine.FEATURE_CONFIG["projector_sha256"]
             and ready["runtime_release"] == engine.FEATURE_CONFIG["runtime_release"]
             and ready["model_revision"] == engine.FEATURE_CONFIG["model_revision"],
             "Spatial replay runtime assets differ from the frozen recipe")
    assets = {key: ready[key] for key in (
        "weights_sha256", "projector_sha256", "runtime_release", "model_revision", "runtime_files")}
    failure = run / "inference_replay_check.json"
    return {"source": source, "chosen": chosen, "cached": cached, "model": model,
            "cache_sha256": digest(cache_path), "original_failure": failure,
            "original_failure_sha256": digest(failure) if failure.exists() else None,
            "public_provenance": {"script_sha256": digest(__file__), **provenance,
                                  "classifier_sha256": digest(model_path),
                                  "runtime_asset_fingerprint": _fingerprint(assets),
                                  "protocol_fingerprint": protocol["fingerprint"]}}


def _comparison(first, second, model):
    def decision(record):
        clip = np.asarray(record["clip_features"]).reshape(1, -1)
        positive = float(engine._positive_probability(model["clip_model"], clip)[0]) >= .5
        scores = engine._positive_probability(model["window_model"], np.stack([
            item["features"] for item in record["windows"]]))
        return positive, record["windows"][int(np.argmax(scores))]["timestamp_sec"]

    before, after = decision(first), decision(second)
    clip_changes = int(np.count_nonzero(np.asarray(first["clip_features"]) != second["clip_features"]))
    window_changes = int(np.count_nonzero(np.stack([row["features"] for row in first["windows"]])
                                         != np.stack([row["features"] for row in second["windows"]])))
    return {"exact_clip_features_equal": clip_changes == 0,
            "exact_window_features_equal": window_changes == 0,
            "differing_clip_features": clip_changes, "differing_window_features": window_changes,
            "classification_threshold_flips": int(before[0] != after[0]),
            "selected_timestamp_changes": int(before[1] != after[1])}


def diagnose(run_dir, protocol_path=None, *, mode="inspect", output_dir=None):
    _require(mode in {"inspect", "sample-only", "run"}, "Unsupported spatial replay diagnostic mode")
    run = Path(run_dir).resolve()
    _require(not run.is_relative_to(config.DATA_RAW.resolve()), "Spatial replay requires derived artifacts")
    protocol_path = protocol_path or config.DATA_PROC / "fortebank-20261017/video_protocol.json"
    prepared = _prepare(run, protocol_path)
    result = {"mode": mode, "selected_development_groups": 1, "holdout_groups_used": 0,
              "selection_uses_labels": False, "inference_calls_completed": 0,
              "semantic_accuracy_verified": False, "promotion_authorized": False,
              "active_process_attested_by_this_script": False,
              "provenance": prepared["public_provenance"], "limitations": list(_LIMITATIONS)}
    if mode == "inspect":
        return result
    output = (Path(output_dir) if output_dir else run / "replay-diagnostics" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]))
    output = output.resolve()
    _require(output.is_relative_to(config.DATA_PROC.resolve())
             and not output.is_relative_to(config.DATA_RAW.resolve())
             and not output.is_relative_to(run / "cache") and not output.is_relative_to(run / "models"),
             "Spatial diagnostic output must be a new derived directory outside frozen artifacts")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _private(output / "selection.joblib", {key: prepared[key] for key in (
        "chosen", "cache_sha256", "original_failure_sha256")})
    try:
        probe = _decode_probe(prepared["source"], prepared["cached"])
    except Exception:
        result.update(diagnostic_complete=False, sampling_identity_equal=False,
                      failure_stage="sampling_probe", failure_details_suppressed=True)
        _private(output / "replay_diagnostic.json", result, json_output=True)
        return result
    _private(output / "sampling_probe.joblib", probe)
    result["sampling_probe_vs_cache"] = identity_difference(prepared["cached"], probe)
    sampling_ok = _same_identity(result["sampling_probe_vs_cache"])
    result["sampling_identity_equal"] = sampling_ok
    captures, records = [], []
    if mode == "run" and sampling_ok:
        for index in range(2):
            try:
                with _capture_requests() as requests:
                    record = engine.extract_video(prepared["source"], deadline=time.monotonic() + 240)
                _validate_record(record)
                _require(len(requests) == engine.WINDOW_COUNT,
                         "Frozen spatial replay request count changed")
            except Exception:
                result.update(failure_stage="frozen_extraction_or_integrity", failure_details_suppressed=True)
                break
            _private(output / f"replay_{index + 1}.joblib", record)
            _private(output / f"request_audit_{index + 1}.joblib", requests)
            result["inference_calls_completed"] += 1
            differences = identity_difference(prepared["cached"], record)
            result[f"replay_{index + 1}_vs_cache_sampling"] = differences
            sampling_ok = sampling_ok and _same_identity(differences)
            records.append(record)
            captures.append(requests)
            if not sampling_ok:
                break  # Fail closed before another VLM call on mismatched inputs.
    result["sampling_identity_equal"] = sampling_ok
    if len(records) == 2:
        result["cached_vs_replay_1"] = _comparison(prepared["cached"], records[0], prepared["model"])
        result["cached_vs_replay_2"] = _comparison(prepared["cached"], records[1], prepared["model"])
        result["replay_1_vs_replay_2"] = _comparison(records[0], records[1], prepared["model"])
        result["request_bodies_equal_between_replays"] = captures[0] == captures[1]
        result["request_bodies_equal_to_serialization_probe"] = all(
            requests == probe["request_audit"] for requests in captures)
        result["numeric_replay_reproducible"] = sampling_ok and all(
            result[key]["exact_clip_features_equal"] and result[key]["exact_window_features_equal"]
            for key in ("cached_vs_replay_1", "cached_vs_replay_2", "replay_1_vs_replay_2"))
    current_failure = prepared["original_failure"]
    _require((digest(current_failure) if current_failure.exists() else None)
             == prepared["original_failure_sha256"], "Original failed replay report changed during diagnosis")
    engine._check_inherited_sources()
    _require(engine.get_encoder_provenance() == {
        key: prepared["public_provenance"][key] for key in engine.get_encoder_provenance()},
        "Spatial replay extractor source changed during diagnosis")
    _require(digest(run / "models/baseline.joblib")
             == prepared["public_provenance"]["classifier_sha256"]
             and digest(__file__) == prepared["public_provenance"]["script_sha256"],
             "Spatial replay classifier or diagnostic source changed during diagnosis")
    result["original_failed_report_preserved"] = True
    result["diagnostic_complete"] = sampling_ok and (mode != "run" or len(records) == 2)
    _private(output / "replay_diagnostic.json", result, json_output=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "video-qwen-spatial-v3")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Decode-check then perform two sequential frozen model extractions")
    mode.add_argument("--sample-only", action="store_true", help="Decode/re-encode only; no VLM calls")
    args = parser.parse_args(argv)
    try:
        result = diagnose(args.run_dir, args.protocol, mode=("run" if args.run else
                          "sample-only" if args.sample_only else "inspect"), output_dir=args.output_dir)
    except Exception:
        raise SystemExit("Private spatial replay diagnostic failed; no media, model text or identifiers emitted") from None
    print(json.dumps(result), flush=True)
    if result.get("diagnostic_complete") is False or result.get("numeric_replay_reproducible") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
