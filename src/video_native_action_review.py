"""Three-task local native-video action pilot, gated by synthetic motion tests.

Machine draft/critique agreement is not independent truth. No media, proposal,
task identifier or clip-specific description is printed by this module.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from scripts import check_temporal_protocol as _protocol
from scripts import provision_action_model as _model_assets
from scripts.provision_action_model import verify_ready as verify_model
from scripts.provision_video_decoder import verify_ready as verify_decoder
from src import config
from src.video_action_proposals import validate_observations
from src.video_actions import (ACTION_IDS, DEFAULT_ACTION_DIR, DEFAULT_PROTOCOL,
                               append_proposal, load_queue, load_reviews)
from src.video_analysis import _check_deadline, _open_video
from src.video_eval import load_protocol
from src.video_local_action_review import consensus
from src.video_local_observations import ACTION_CHECK_PROMPT, ACTION_DRAFT_PROMPT, ACTION_SCHEMA
from src.video_qwen_actions import _read_frame_at


SAMPLE_FPS = 2.
MAX_FRAMES = 16
MIN_FRAMES = 4
FRAME_MAX_SIDE = 640
JPEG_QUALITY = 90
REQUEST_SECONDS = 45.
TASK_SECONDS = 150.
MAX_VIDEO_BYTES = 32 * 1024 * 1024
MEDIA_ROOT = config.DATA_PROC / "fortebank-20261017/open/videos"
DEFAULT_GATE = config.DATA_PROC / "temporal-protocol-native/temporal_protocol.json"
SOURCE_PATHS = (
    "scripts/check_temporal_protocol.py", "scripts/provision_action_model.py",
    "scripts/provision_video_decoder.py", "scripts/serve_action_model.py", "src/video_analysis.py",
    "src/video_qwen_actions.py", "src/video_actions.py",
    "src/video_local_observations.py", "src/video_local_action_review.py",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_recipe():
    value = {"version": 1, "protocol": "input_video/native paired frames",
             "sample_fps": SAMPLE_FPS, "frame_max_side": FRAME_MAX_SIDE,
             "sample_selection": "contiguous from queued review interval start, strictly before end, first 16 only",
             "source_clock_policy": "reject end beyond source duration+max(0.25 seconds,2/FPS); otherwise clamp and preserve both bounds",
             "pilot_pool": "immutable first three eligible tasks per recipe, bound to queue SHA and protocol; later reviews only remove tasks",
             "frame_pair_policy": "trim last odd sample; require 4..16 even frames; never pad or duplicate",
             "frame_view": "whole source frame, downsize to even dimensions without upsampling",
             "jpeg_quality": JPEG_QUALITY, "video_codec": "MJPG", "video_container": "AVI",
             "helper_video_fps": 2, "helper_timestamp_interval_ms": 0,
             "draft_prompt": ACTION_DRAFT_PROMPT, "critique_prompt": ACTION_CHECK_PROMPT,
             "response_schema": ACTION_SCHEMA,
             "leading_text": "chronological frame index to actual original source seconds",
             "prior_serialization": "vocabulary-order present/unknown states only",
             "agreement": "existing same-model consensus; all records remain proposed",
             "max_tasks": 3, "max_request_seconds": REQUEST_SECONDS, "task_seconds": TASK_SECONDS,
             "temperature": 0, "seed": config.SEED, "max_tokens": 256,
             "source_sha256": {relative: _digest(config.ROOT / relative) for relative in SOURCE_PATHS},
             "module_source_sha256": _digest(__file__)}
    return value, hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def interval_sample_times(start, end):
    if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float))
            or not isinstance(end, (int, float)) or not math.isfinite(start) or not math.isfinite(end)
            or not 0 <= start < end <= 300):
        raise ValueError("Native review interval is unsupported")
    values = np.arange(float(start), float(end), 1 / SAMPLE_FPS)[:MAX_FRAMES]
    if len(values) % 2:
        values = values[:-1]
    if len(values) < MIN_FRAMES:
        raise ValueError("Native review needs at least four distinct even chronological samples")
    return values


def _view(frame):
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
            or frame.dtype != np.uint8 or min(frame.shape[:2]) < 2):
        raise ValueError("Native review requires valid visual source frames")
    height, width = frame.shape[:2]
    factor = min(1., FRAME_MAX_SIDE / max(height, width))
    size = (max(2, (round(width * factor) // 2) * 2), max(2, (round(height * factor) // 2) * 2))
    return frame if (width, height) == size else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def sample_task_video(task, directory, deadline):
    """Decode only the source-verified queued interval; return private audit."""
    _check_deadline(deadline)
    path = (config.ROOT / task["path"]).resolve()
    if (task.get("partition") != "development" or not path.is_relative_to(MEDIA_ROOT.resolve())
            or path.suffix.lower() != ".mov" or _digest(path) != task["source_sha256"]):
        raise ValueError("Native review requires a source-verified development task")
    _check_deadline(deadline)
    cap, fps, count = _open_video(path, deadline)
    frames, records = [], []
    try:
        if count <= 0:
            raise ValueError("Native review requires source frame-count metadata")
        duration = count / fps
        declared_start, declared_end = task["interval"]["start_sec"], task["interval"]["end_sec"]
        # Validate original bounds before clamping; small container-clock
        # differences never become invented source coverage.
        interval_sample_times(declared_start, declared_end)
        if declared_end > duration + max(.25, 2 / fps):
            raise ValueError("Native review interval exceeds the source video")
        sampling_end = min(float(declared_end), duration)
        wanted = interval_sample_times(declared_start, sampling_end)
        for requested in wanted:
            frame = _read_frame_at(cap, float(requested), fps, deadline)
            raw_index = float(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if (not math.isfinite(raw_index) or abs(raw_index - round(raw_index)) > 1e-5
                    or not 0 <= raw_index < count):
                raise ValueError("Native review source frame index is invalid")
            index = int(round(raw_index))
            actual = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            timestamp_source = "decoder"
            if not math.isfinite(actual) or actual < 0 or (actual == 0 and index > 0):
                actual, timestamp_source = index / fps, "source_frame_index_over_fps"
            if (not math.isfinite(actual) or not 0 <= actual < count / fps
                    or abs(actual - requested) > max(.25, 2 / fps)
                    or records and (actual <= records[-1]["actual_sec"] or index <= records[-1]["source_frame_index"])):
                raise ValueError("Native review source sample timing could not be verified")
            view = _view(frame)
            if frames and view.shape != frames[0].shape:
                raise ValueError("Native review cannot change image shape within one video")
            ok, encoded = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise ValueError("Native review source view could not be encoded")
            frames.append(view)
            records.append({"requested_sec": float(requested), "actual_sec": float(actual),
                            "source_frame_index": index, "timestamp_source": timestamp_source,
                            "declared_interval": {"start_sec": float(declared_start), "end_sec": float(declared_end)},
                            "sampling_interval": {"start_sec": float(declared_start), "end_sec": sampling_end},
                            "source_duration_sec": duration, "interval_end_clamped": sampling_end != declared_end,
                            "source_shape": list(frame.shape), "view_shape": list(view.shape),
                            "source_frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                            "jpeg_sha256": hashlib.sha256(encoded.tobytes()).hexdigest()})
            _check_deadline(deadline)
    finally:
        cap.release()
    video_path = Path(directory) / "native-review.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), SAMPLE_FPS,
                             (frames[0].shape[1], frames[0].shape[0]))
    if not writer.isOpened():
        raise ValueError("Native review local AVI encoder is unavailable")
    try:
        for frame in frames:
            _check_deadline(deadline)
            writer.write(frame)
    finally:
        writer.release()
    encoded = cv2.VideoCapture(str(video_path))
    try:
        if (not encoded.isOpened() or int(encoded.get(cv2.CAP_PROP_FRAME_COUNT)) != len(frames)
                or abs(encoded.get(cv2.CAP_PROP_FPS) - SAMPLE_FPS) > 1e-6):
            raise ValueError("Native review AVI timing does not match the fixed protocol")
        for record in records:
            ok, frame = encoded.read()
            if not ok or list(frame.shape) != record["view_shape"]:
                raise ValueError("Native review AVI frames do not match the sampled source views")
            record["avi_decoded_frame_sha256"] = hashlib.sha256(frame.tobytes()).hexdigest()
            _check_deadline(deadline)
        if encoded.read()[0]:
            raise ValueError("Native review AVI contains unexpected extra frames")
    finally:
        encoded.release()
    size = video_path.stat().st_size
    if not 0 < size <= MAX_VIDEO_BYTES:
        raise ValueError("Native review AVI exceeds the bounded payload size")
    payload = video_path.read_bytes()
    _check_deadline(deadline)
    return payload, records


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Duplicate native review response field")
        output[key] = value
    return output


def parse_proposal(response):
    try:
        choices = response["choices"]
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError
        value = json.loads(choices[0]["message"]["content"], object_pairs_hook=_unique_object)
        if (not isinstance(value, dict) or set(value) != {"actions", "visibility"}
                or not isinstance(value["actions"], dict) or set(value["actions"]) != set(ACTION_IDS)
                or any(not isinstance(state, str) or state not in {"present", "unknown"}
                       for state in value["actions"].values())):
            raise ValueError
        value["actions"] = [{"action_id": identifier, "presence": value["actions"][identifier]} for identifier in ACTION_IDS]
        return validate_observations(value)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        raise ValueError("Native review response differs from the fixed action schema") from None


def request_native_review(payload, samples, model, deadline, prior_actions=None):
    if (not isinstance(payload, bytes) or not payload.startswith(b"RIFF") or len(payload) > MAX_VIDEO_BYTES
            or not isinstance(samples, list) or not MIN_FRAMES <= len(samples) <= MAX_FRAMES or len(samples) % 2):
        raise ValueError("Native review requires a bounded even-frame AVI payload")
    mapping, previous = [], -1.
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or "actual_sec" not in sample:
            raise ValueError("Native review samples require their actual source times")
        actual = sample["actual_sec"]
        if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual)
                or not 0 <= actual < 300 or actual <= previous):
            raise ValueError("Native review leading frame times must strictly increase")
        mapping.append(f"{index}={actual:.6f}s")
        previous = actual
    prompt = ACTION_DRAFT_PROMPT
    if prior_actions is not None:
        checked = validate_observations({"actions": prior_actions, "visibility": "partial"})
        states = {item["action_id"]: item["presence"] for item in checked["actions"]}
        if set(states) != set(ACTION_IDS):
            raise ValueError("Native review prior must cover the fixed action vocabulary")
        prior = {identifier: states[identifier] for identifier in ACTION_IDS}
        prompt = ACTION_CHECK_PROMPT + json.dumps(prior, separators=(",", ":"))
    leading = "Chronological video-frame index to actual original source seconds: " + "; ".join(mapping) + ".\n"
    _check_deadline(deadline)
    timeout = min(REQUEST_SECONDS, deadline - time.monotonic())
    body = {"model": model["alias"], "temperature": 0, "seed": config.SEED, "max_tokens": 256,
            "stream": False, "response_format": {"type": "json_schema", "json_schema": {
                "name": "native_action_review", "strict": True, "schema": ACTION_SCHEMA}},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": leading + prompt},
                {"type": "input_video", "input_video": {"data": base64.b64encode(payload).decode("ascii")}},
            ]}]}
    try:
        response = _protocol.request_json("/v1/chat/completions", body, timeout=timeout)
    except (OSError, ValueError, RuntimeError):
        raise RuntimeError("Local native action review request failed") from None
    _check_deadline(deadline)
    return parse_proposal(response)


def _append_audit(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        Path(path).chmod(0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _pilot_selection(directory, recipe_sha, queue, protocol, human):
    """Persist one immutable pool before skipping later reviews/completions."""
    queue_sha = hashlib.sha256(json.dumps(queue, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    context = {"schema_version": 1, "recipe_sha256": recipe_sha,
               "protocol_fingerprint": protocol["fingerprint"], "queue_sha256": queue_sha}
    path = Path(directory) / ("selection-" + recipe_sha + ".json")
    if not path.exists():
        selected = [task["task_id"] for task in queue["tasks"] if task["task_id"] not in human][:3]
        try:
            with path.open("x", encoding="utf-8") as handle:
                path.chmod(0o600)
                handle.write(json.dumps({**context, "task_ids": selected}, sort_keys=True) + "\n")
        except FileExistsError:
            pass  # Another caller may establish the same immutable pool first.
    selection = json.loads(path.read_text())
    if (not isinstance(selection, dict) or set(selection) != set(context) | {"task_ids"}
            or any(selection.get(key) != value for key, value in context.items())):
        raise ValueError("Native pilot selection differs from its frozen recipe, queue or protocol")
    ids = selection["task_ids"]
    by_id = {task["task_id"]: task for task in queue["tasks"]}
    if (not isinstance(ids, list) or len(ids) > 3 or any(not isinstance(item, str) or item not in by_id for item in ids)
            or len(ids) != len(set(ids)) or len(by_id) != len(queue["tasks"])):
        raise ValueError("Native pilot selection has invalid fixed task identifiers")
    return [by_id[identifier] for identifier in ids]


def review_tasks(*, out_dir=DEFAULT_ACTION_DIR, protocol_path=DEFAULT_PROTOCOL,
                 gate_path=DEFAULT_GATE, limit=3):
    if type(limit) is not int or not 1 <= limit <= 3:
        raise ValueError("Native action review is limited to the fixed three-task pilot")
    directory = Path(out_dir) / "native-review-v1"
    if directory.resolve().is_relative_to(config.DATA_RAW.resolve()):
        raise ValueError("Native action review cannot write raw data")
    model, decoder = verify_model(), verify_decoder()  # Exactly once before any media.
    _protocol.verify_temporal_gate(gate_path, model, decoder)
    recipe, recipe_sha = review_recipe()
    queue, protocol = load_queue(out_dir, protocol_path), load_protocol(protocol_path)
    development, holdout = set(protocol["development_group_ids"]), set(protocol["holdout_group_ids"])
    if any(task.get("partition") != "development" or task["group_id"] not in development
           or task["group_id"] in holdout for task in queue["tasks"]):
        raise ValueError("Reserved or unknown footage cannot enter native action review")
    history = load_reviews(out_dir, protocol_path)
    human = {item["task_id"] for item in history if item["provenance"]["kind"] == "human_review"}
    existing = {item["task_id"] for item in history if item["provenance"].get("recipe_sha256") == recipe_sha}
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / "audits.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            if record.get("recipe_sha256") == recipe_sha and record.get("status") == "completed":
                existing.add(record["task_id"])
    pilot = _pilot_selection(directory, recipe_sha, queue, protocol, human)
    tasks = [task for task in pilot if task["task_id"] not in human | existing][:limit]
    saved = failures = present = disagreements = 0
    started = time.monotonic()
    if tasks:
        with _protocol.native_server(model, decoder):
            for task in tasks:
                task_started = time.monotonic()
                deadline = task_started + TASK_SECONDS
                try:
                    with tempfile.TemporaryDirectory(dir=directory) as scratch:
                        payload, samples = sample_task_video(task, scratch, deadline)
                        draft = request_native_review(payload, samples, model, deadline)
                        critique = request_native_review(payload, samples, model, deadline, draft["actions"])
                    merged = consensus(draft, critique)
                    _check_deadline(deadline)
                    count = sum(item["presence"] == "present" for item in merged["actions"])
                    provenance = {"kind": "local_model_proposal", "model_id": model["model_id"],
                                  "weights_sha256": model["weights_sha256"], "projector_sha256": model["projector_sha256"],
                                  "recipe_sha256": recipe_sha, "runtime_release": model["runtime_release"],
                                  "runtime_archive_sha256": _model_assets.ASSETS[2].sha256,
                                  "runtime_files_manifest_sha256": hashlib.sha256(json.dumps(
                                      model.get("runtime_files", {}), sort_keys=True).encode()).hexdigest(),
                                  "decoder_archive_sha256": decoder["archive_sha256"],
                                  "sample_times_sec": [item["actual_sec"] for item in samples],
                                  "frame_sha256s": [item["jpeg_sha256"] for item in samples],
                                  "source_frame_sha256s": [item["source_frame_sha256"] for item in samples],
                                  "native_video_sha256": hashlib.sha256(payload).hexdigest(),
                                  "review_method": "native video draft plus same-model critique; unverified"}
                    record = {"status": "completed", "task_id": task["task_id"], "source_sha256": task["source_sha256"],
                              "recipe": recipe, "recipe_sha256": recipe_sha, "provenance": provenance,
                              "samples": samples, "draft": draft, "critique": critique, "consensus": merged,
                              "consensus_present_count": count, "verified": False,
                              "evidence_sample_indices": [0, (len(samples) - 1) // 2, len(samples) - 1],
                              "elapsed_sec": time.monotonic() - task_started}
                    _append_audit(journal, record)
                    latest = load_reviews(out_dir, protocol_path)
                    if any(item["task_id"] == task["task_id"] and item["provenance"]["kind"] == "human_review" for item in latest):
                        continue
                    append_proposal(out_dir, task["task_id"], merged["actions"], merged["visibility"], provenance,
                                    protocol_path, interval={"start_sec": samples[0]["actual_sec"],
                                                             "end_sec": samples[-1]["actual_sec"]})
                    saved += 1
                    present += count
                    disagreements += sum(first != second for first, second in zip(draft["actions"], critique["actions"]))
                except (OSError, ValueError, RuntimeError, TimeoutError, cv2.error) as error:
                    failures += 1
                    _append_audit(journal, {"status": "failed", "task_id": task["task_id"],
                                           "recipe_sha256": recipe_sha, "exception_type": type(error).__name__,
                                           "elapsed_sec": time.monotonic() - task_started, "verified": False})
    return {"tasks_attempted": len(tasks), "machine_rechecked_proposals": saved, "tasks_failed": failures,
            "consensus_present_observations": present, "pass_disagreements": disagreements,
            "verified_labels_created": 0, "holdout_groups_used": 0, "semantic_accuracy_verified": False,
            "bulk_stopped_for_unknowns": saved > 0 and present == 0,
            "bulk_expansion_allowed": False, "elapsed_sec": time.monotonic() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    args = parser.parse_args(argv)
    try:
        result = review_tasks(limit=args.limit, gate_path=args.gate)
    except (OSError, ValueError, RuntimeError, cv2.error):
        raise SystemExit("Native action review could not start safely; no media or observations emitted") from None
    print(json.dumps(result), flush=True)
    if result["tasks_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
