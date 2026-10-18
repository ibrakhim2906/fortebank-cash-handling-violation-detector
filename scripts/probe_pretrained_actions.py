"""Private, train-fold-only pretrained action proposals; never fit or promote."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import socket
import time
from unittest.mock import patch

import numpy as np

from scripts import provision_movinet
from src import config, video_eval
from src import video_movinet as engine


CLASS_INDICES = {"counting money": 112, "looking at phone": 265,
                 "talking on cell phone": 516, "texting": 525, "writing": 595}
ENCODER_SHA256 = "63323474bfdaa6f7d1aa9cfe038380019da9745329179555fbd8f87575ec5205"
LABELS_SHA256 = "8dfca4202520ea8291ce0242ef33b006d0a0772b409663c090a2876bc1978dcf"
LABELS_SOURCE_COMMIT = "f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2"
DEFAULT_RUN_DIR = config.DATA_PROC / "pretrained-action-probe-v1"
RECIPE = {"version": 1, "seed": config.SEED, "selected_recordings": 8, "fold": 0,
          "pool": "train", "selection": "sha256(f'{seed}:{group_id}'); first eight ascending",
          "window_seconds": 8, "stride_seconds": 2, "sample_fps": 5,
          "classes": CLASS_INDICES, "scores": "stable softmax over all 600 logits; uncalibrated",
          "rank": "descending logit; equal logits ordered by ascending class index; one-based",
          "repeat": "longest selected duration, group_id tie-break; fresh encoder in same process",
          "training": False, "automatic_promotion": False}


class ProbeError(ValueError):
    """Safe fixed-message failure; never contains identifiers or model content."""


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ProbeError(message)


def select_groups(protocol):
    video_eval.validate_protocol(protocol)
    _require(protocol["seed"] == config.SEED and len(protocol["development_group_ids"]) == 32
             and len(protocol["holdout_group_ids"]) == 8, "Probe requires the frozen forty-group protocol")
    fold = protocol["folds"][0]
    eligible = set(fold["train_group_ids"])
    _require(not eligible & set(fold["valid_group_ids"] + protocol["holdout_group_ids"])
             and len(eligible) >= 8, "Probe training pool crosses protected partitions")
    ordered = sorted(eligible, key=lambda group: hashlib.sha256(f"{config.SEED}:{group}".encode()).hexdigest())[:8]
    by_id = {row["group_id"]: row for row in protocol["groups"]}
    return [{"group_id": group, "path": by_id[group]["path"], "source_sha256": group,
             "duration_sec": by_id[group]["duration_sec"]} for group in ordered]


def checked_source(selection):
    relative = PurePosixPath(selection["path"])
    _require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in str(relative),
             "Probe source must have a safe repository-relative path")
    path = config.ROOT.joinpath(*relative.parts)
    boundary = config.DATA_PROC / "fortebank-20261017/open/videos"
    _require(path.resolve().is_relative_to(boundary.resolve()) and path.suffix.lower() == ".mov"
             and not any(parent.is_symlink() for parent in (path, *path.parents))
             and path.is_file() and digest(path) == selection["source_sha256"]
             and selection["source_sha256"] == selection["group_id"],
             "Probe source is outside the approved extraction tree or its checksum changed")
    return path


def _provenance(protocol_path):
    labels_path = config.ROOT / "licenses/movinet/kinetics_600_labels.txt"
    metadata_path = labels_path.with_suffix(".provenance.json")
    _require(digest(labels_path) == LABELS_SHA256, "Official action label map changed")
    labels = labels_path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(metadata_path.read_text())
    _require(len(labels) == 600 and all(labels[index] == label for label, index in CLASS_INDICES.items())
             and metadata.get("sha256") == LABELS_SHA256 and metadata.get("classes") == 600
             and metadata.get("bytes") == labels_path.stat().st_size
             and metadata.get("source_commit") == LABELS_SOURCE_COMMIT,
             "Action labels do not match the official pinned class order")
    _require(digest(engine.__file__) == ENCODER_SHA256, "Frozen temporal encoder source changed")
    ready = provision_movinet.verify_ready()
    return {"probe_source_sha256": digest(__file__), "encoder": engine.get_encoder_provenance(),
            "feature_config": engine.FEATURE_CONFIG, "approved_model": ready,
            "protocol_file_sha256": digest(protocol_path), "labels_sha256": LABELS_SHA256,
            "label_provenance_sha256": digest(metadata_path), "labels_source_commit": LABELS_SOURCE_COMMIT,
            "helper_sources": {"video_eval": digest(video_eval.__file__),
                               "provision_movinet": digest(provision_movinet.__file__)},
            "plan_sha256": digest(config.ROOT / "docs/pretrained_probe.md")}


def _write_bytes(path, payload, *, reuse=False):
    if path.exists():
        _require(reuse and not path.is_symlink() and path.read_bytes() == payload,
                 "Existing probe artifact differs; preserve it and use a new run directory")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def _write_json(path, value, *, reuse=False):
    _write_bytes(path, (json.dumps(value, sort_keys=True, ensure_ascii=False,
                                  indent=2, allow_nan=False) + "\n").encode(), reuse=reuse)


def prepare(run_dir, protocol_path):
    protocol = video_eval.load_protocol(protocol_path)
    selections = select_groups(protocol)
    for row in selections:
        checked_source(row)  # Hash only; no decoder or interpreter is opened.
    manifest = {"schema_version": 1, "purpose": "unreviewed_pretrained_action_proposals",
                "protocol_fingerprint": protocol["fingerprint"], "recipe": RECIPE,
                "selections": selections, "provenance": _provenance(protocol_path),
                "repeat_group_id": sorted(selections, key=lambda row: (-row["duration_sec"], row["group_id"]))[0]["group_id"]}
    output = Path(run_dir).resolve()
    _require(output.is_relative_to(config.DATA_PROC.resolve()) and output != config.DATA_PROC.resolve()
             and not output.is_relative_to((config.DATA_PROC / "fortebank-20261017").resolve()),
             "Probe output must be a separate derived directory")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require((output.stat().st_mode & 0o777) == 0o700, "Probe output directory must be private mode 700")
    _write_json(output / "manifest.json", manifest, reuse=True)
    _write_bytes(output / "probe_source.py", Path(__file__).read_bytes(), reuse=True)
    _write_bytes(output / "encoder_source.py", Path(engine.__file__).read_bytes(), reuse=True)
    return output, protocol, manifest, digest(output / "manifest.json")


def probabilities_and_ranks(logits):
    values = np.asarray(logits, dtype=np.float64)
    _require(values.shape == (600,) and np.isfinite(values).all(), "Pretrained output requires 600 finite logits")
    shifted = np.exp(values - values.max())
    probabilities = shifted / shifted.sum()
    order = np.lexsort((np.arange(600), -values))
    ranks = np.empty(600, dtype=int)
    ranks[order] = np.arange(1, 601)
    return probabilities, ranks


_SAMPLE_FIELDS = ("requested_times_sec", "actual_times_sec", "source_frame_indices", "source_frame_sha256",
                  "source_shapes", "timestamp_sources")


def _validate_sampling(sample, requested, duration):
    expected = np.asarray(requested)
    actual = np.asarray(sample.get("actual_times_sec", []), dtype=float)
    _require(sample.get("feature_config") == engine.FEATURE_CONFIG
             and np.array_equal(np.asarray(sample.get("requested_times_sec", [])), expected)
             and actual.shape == expected.shape and len(actual) > 0 and np.isfinite(actual).all()
             and np.all(np.diff(actual) > 0) and np.all(actual >= expected - 1e-7)
             and np.all(actual < duration) and np.all(actual - expected <= .25), "Pretrained sampling times/configuration changed")
    count = len(expected)
    indices = sample.get("source_frame_indices", [])
    hashes = sample.get("source_frame_sha256", [])
    shapes = sample.get("source_shapes", [])
    _require(len(indices) == count and all(type(index) is int and index >= 0 for index in indices)
             and all(right > left for left, right in zip(indices, indices[1:]))
             and len(hashes) == count and all(isinstance(value, str) and len(value) == 64
                 and all(char in "0123456789abcdef" for char in value) for value in hashes)
             and len(shapes) == count and all(isinstance(shape, list) and len(shape) == 3 and shape[2] == 3
                 and all(type(size) is int and size > 0 for size in shape) for shape in shapes)
             and len(sample.get("timestamp_sources", [])) == count
             and all(value in {"decoder", "source_frame_index_over_fps"} for value in sample["timestamp_sources"]),
             "Pretrained sampling provenance is incomplete")


def private_record(selection, extracted, manifest):
    duration = extracted.get("duration_sec")
    _require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0
             and abs(duration - selection["duration_sec"]) <= .25, "Decoded source duration differs from its frozen metadata")
    requested = [index / 5 for index in range(math.ceil(duration * 5)) if index / 5 < duration - 1e-8]
    sampling = extracted.get("sampling", {})
    _validate_sampling(sampling, requested, duration)
    base_provenance = manifest["provenance"]["encoder"]
    _require(all(extracted.get("encoder_provenance", {}).get(key) == value for key, value in base_provenance.items()),
             "Extracted model/runtime provenance differs from the manifest")
    windows, intervals = [], engine.window_intervals(duration)
    _require(len(extracted.get("windows", [])) == len(intervals), "Dense pretrained window count changed")
    for row, interval in zip(extracted["windows"], intervals, strict=True):
        _require(all(row.get(key) == value for key, value in interval.items()), "Dense pretrained window boundaries changed")
        indices = [index for index, value in enumerate(requested)
                   if interval["start_sec"] <= value < interval["end_sec"] - 1e-8]
        sample = row.get("sampling", {})
        _require(all(sample.get(key) == [sampling[key][index] for index in indices] for key in _SAMPLE_FIELDS)
                 and sample.get("feature_config") == engine.FEATURE_CONFIG,
                 "Window sampling differs from its complete recording audit")
        timestamp = (sample["actual_times_sec"][0] + sample["actual_times_sec"][-1]) / 2
        _require(row.get("timestamp_sec") == timestamp, "Pretrained candidate timestamp changed")
        values = np.asarray(row.get("features"), dtype=float)
        probabilities, ranks = probabilities_and_ranks(values)
        predictions = [{"label": label, "class_index": index, "score": float(probabilities[index]),
                        "rank_1based": int(ranks[index])} for label, index in CLASS_INDICES.items()]
        windows.append({**interval, "timestamp_sec": timestamp, "logits": values.tolist(),
                        "predictions": predictions, "sampling": sample})
    elapsed = extracted.get("extraction_sec")
    _require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
             "Pretrained extraction timing is invalid")
    return {**selection, "decoded_duration_sec": duration, "extraction_sec": elapsed, "windows": windows,
            "sampling": sampling, "encoder_provenance": extracted["encoder_provenance"]}


@contextmanager
def deny_python_connections():
    attempts = {"connect": 0, "connect_ex": 0, "create_connection": 0}

    def denied(name):
        def fail(*args, **kwargs):
            attempts[name] += 1
            raise ProbeError("Python socket connections are blocked during local inference")
        return fail

    with patch.object(socket.socket, "connect", denied("connect")), \
            patch.object(socket.socket, "connect_ex", denied("connect_ex")), \
            patch.object(socket, "create_connection", denied("create_connection")):
        yield attempts


def _assert_frozen(output, manifest, manifest_sha, protocol_path):
    _require(digest(output / "manifest.json") == manifest_sha
             and _provenance(protocol_path) == manifest["provenance"], "Probe manifest/code/model provenance changed during execution")


def aggregate(records):
    windows = [row for record in records for row in record["windows"]]
    result = {"windows_completed": len(windows),
              "distinct_logit_vectors": len({tuple(row["logits"]) for row in windows}), "class_statistics": {}}
    for label in CLASS_INDICES:
        per_record = [[next(prediction for prediction in row["predictions"] if prediction["label"] == label)
                       for row in record["windows"]] for record in records]
        predictions = [item for items in per_record for item in items]
        scores, ranks = [item["score"] for item in predictions], [item["rank_1based"] for item in predictions]
        result["class_statistics"][label] = {
            "windows_top1": sum(rank == 1 for rank in ranks), "windows_top5": sum(rank <= 5 for rank in ranks),
            "recordings_with_top5": sum(any(item["rank_1based"] <= 5 for item in items) for items in per_record),
            "softmax_min": min(scores) if scores else None, "softmax_median": float(np.median(scores)) if scores else None,
            "softmax_max": max(scores) if scores else None, "rank_min": min(ranks) if ranks else None,
            "rank_max": max(ranks) if ranks else None}
    return result


def run_probe(run_dir, protocol_path):
    output, protocol, manifest, manifest_sha = prepare(run_dir, protocol_path)
    _write_json(output / "run_started.json", {"manifest_sha256": manifest_sha, "mode": "inference_only"})
    records, repeat, attempts, stage = [], None, {}, "initial_provenance"
    started = time.monotonic()
    summary = {"schema_version": 1, "status": "failed", "manifest_sha256": manifest_sha,
               "selected_recordings": 8, "completed_recordings": 0, "action_accuracy": None,
               "verified_action_labels": 0, "automatic_promotion": False, "training_performed": False,
               "proposal_status": "unreviewed_model_proposals", "reserved_recordings_untouched": 8,
               "blind_pool_decoded": 0, "blind_pool_recordings": len(protocol["folds"][0]["valid_group_ids"]),
               "network_check": "Python socket entry points only; not OS isolation",
               "repeat_checked": False, "repeatability_passed": False, "repeat_fresh_encoder_same_process": True,
               "limitations": ["Scores are uncalibrated600-way action scores, not bank violations or action absence",
                               "No independently verified labels or action/localization accuracy exists",
                               "Prediction inspection cannot create independent blind training/evaluation labels"]}
    try:
        with deny_python_connections() as attempts:
            _assert_frozen(output, manifest, manifest_sha, protocol_path)
            encoder = engine.MoViNetEncoder()
            for index, selection in enumerate(manifest["selections"]):
                stage = "recording_extraction"
                _assert_frozen(output, manifest, manifest_sha, protocol_path)
                source = checked_source(selection)
                extracted = engine.extract_video(source, deadline=time.monotonic() + 240, encoder=encoder)
                record = private_record(selection, extracted, manifest)
                _require(not any(attempts.values()), "A Python connection was attempted during local inference")
                _write_json(output / f"record_{index + 1:02d}.json", record)
                records.append(record)
            stage = "fresh_instance_repeat"
            selection = next(row for row in manifest["selections"] if row["group_id"] == manifest["repeat_group_id"])
            _assert_frozen(output, manifest, manifest_sha, protocol_path)
            source = checked_source(selection)
            fresh_encoder = engine.MoViNetEncoder()
            _require(fresh_encoder is not encoder, "Repeat requires a fresh encoder instance")
            repeat = private_record(selection, engine.extract_video(source, deadline=time.monotonic() + 240,
                                                                   encoder=fresh_encoder), manifest)
            _write_json(output / "repeat_record.json", repeat)
            original = next(row for row in records if row["group_id"] == selection["group_id"])
            summary["repeat_checked"] = True
            summary["repeat_logits_equal"] = [row["logits"] for row in original["windows"]] == [row["logits"] for row in repeat["windows"]]
            summary["repeat_sampling_equal"] = (fingerprint(original["sampling"]) == fingerprint(repeat["sampling"])
                and [fingerprint(row["sampling"]) for row in original["windows"]]
                == [fingerprint(row["sampling"]) for row in repeat["windows"]])
            _require(summary["repeat_logits_equal"] and summary["repeat_sampling_equal"], "Fresh-instance repeat differs")
            _require(not any(attempts.values()), "A Python connection was attempted during local inference")
            _assert_frozen(output, manifest, manifest_sha, protocol_path)
            summary["repeatability_passed"] = True
            summary["status"] = "completed"
    except Exception as error:
        summary.update(failure_stage=stage, exception_type=type(error).__name__, details_suppressed=True)
    summary.update(completed_recordings=len(records), elapsed_sec=time.monotonic() - started,
                   python_connection_attempts=sum(attempts.values()), **aggregate(records))
    durations = [row["extraction_sec"] for row in records]
    summary["extraction_sec_min"] = min(durations) if durations else None
    summary["extraction_sec_max"] = max(durations) if durations else None
    _write_json(output / "records.json", {"schema_version": 1, "manifest_sha256": manifest_sha,
                                         "status": summary["status"], "records": records})
    summary["records_sha256"] = digest(output / "records.json")
    _write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), default="prepare")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--protocol", type=Path, default=config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    args = parser.parse_args(argv)
    stage = "prepare" if args.stage == "prepare" else "run"
    try:
        if args.stage == "prepare":
            prepare(args.run_dir, args.protocol)
            print(json.dumps({"status": "prepared", "selected_recordings": 8, "training_performed": False,
                              "action_accuracy": None, "reserved_recordings_untouched": 8, "blind_pool_decoded": 0}))
            return 0
        summary = run_probe(args.run_dir, args.protocol)
        safe = ("status", "selected_recordings", "completed_recordings", "windows_completed", "elapsed_sec",
                "python_connection_attempts", "repeat_checked", "repeat_logits_equal", "repeat_sampling_equal",
                "action_accuracy", "verified_action_labels", "automatic_promotion", "training_performed",
                "reserved_recordings_untouched", "blind_pool_decoded", "failure_stage", "exception_type")
        print(json.dumps({key: summary[key] for key in safe if key in summary}), flush=True)
        return 0 if summary["status"] == "completed" else 1
    except Exception as error:
        print(json.dumps({"status": "failed", "failure_stage": stage, "exception_type": type(error).__name__,
                          "safe_reason": str(error) if isinstance(error, ProbeError) else None,
                          "details_suppressed": True, "action_accuracy": None}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
