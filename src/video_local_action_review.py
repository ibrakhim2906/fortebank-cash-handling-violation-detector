"""Review development action drafts twice with the approved local vision model.

Two passes provide machine consistency only. All records remain proposals;
neither model agreement nor an organizer binary target creates verified truth.
Video bytes, observations and per-task audit details are never printed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from scripts.provision_action_model import verify_ready
from src import config
from src.video_actions import (ACTION_IDS, DEFAULT_ACTION_DIR, DEFAULT_PROTOCOL,
                               append_proposal, load_queue, load_reviews)
from src.video_analysis import _check_deadline, _open_video
from src.video_eval import load_protocol
from src.video_qwen_actions import _read_frame_at


FRAME_MAX_SIDE = 640
JPEG_QUALITY = 90


def review_recipe():
    from src.video_local_observations import RECIPE_SHA256
    value = {"version": 3, "client_recipe_sha256": RECIPE_SHA256,
             "samples": 3, "selection": "chronological start/middle/end of development review interval",
             "frame_max_side": FRAME_MAX_SIDE, "jpeg_quality": JPEG_QUALITY,
             "two_pass_policy": "Present only if independently returned present by draft and critique; otherwise unknown",
             "truth_status": "unverified machine consistency; never human verified",
             "seed": config.SEED}
    return value, hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def consensus(draft, critique):
    from src.video_action_proposals import validate_observations
    first, second = validate_observations(draft), validate_observations(critique)
    left = {item["action_id"]: item["presence"] for item in first["actions"]}
    right = {item["action_id"]: item["presence"] for item in second["actions"]}
    if set(left) != set(ACTION_IDS) or set(right) != set(ACTION_IDS):
        raise ValueError("Both local review passes must cover the fixed vocabulary")
    visibility = ("unobservable" if "unobservable" in {first["visibility"], second["visibility"]}
                  else "clear" if first["visibility"] == second["visibility"] == "clear" else "partial")
    actions = [{"action_id": identifier,
                "presence": "present" if visibility != "unobservable"
                and left[identifier] == right[identifier] == "present" else "unknown"}
               for identifier in ACTION_IDS]
    return {"actions": actions, "visibility": visibility}


def sample_interval(task, deadline):
    path = (config.ROOT / task["path"]).resolve()
    allowed = (config.DATA_PROC / "fortebank-20261017/open/videos").resolve()
    if (task.get("partition") != "development" or not path.is_relative_to(allowed)
            or path.suffix.lower() != ".mov"
            or hashlib.sha256(path.read_bytes()).hexdigest() != task["source_sha256"]):
        raise ValueError("Local action review requires a source-verified development task")
    cap, fps, count = _open_video(path, deadline)
    images, times = [], []
    try:
        start = float(task["interval"]["start_sec"])
        end = min(float(task["interval"]["end_sec"]), count / fps - 1 / fps)
        if not 0 <= start < end:
            raise ValueError("Local action review interval is invalid")
        for wanted in np.linspace(start, end, 3):
            frame = _read_frame_at(cap, float(wanted), fps, deadline)
            actual = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not np.isfinite(actual) or actual <= 0:
                actual = max(0.0, (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps)
            if (not np.isfinite(actual) or abs(actual - wanted) > max(0.25, 2 / fps)
                    or not 0 <= actual < count / fps or (times and actual <= times[-1])):
                raise ValueError("Local action review sampling timing could not be verified")
            height, width = frame.shape[:2]
            factor = min(1.0, FRAME_MAX_SIDE / max(height, width))
            if factor < 1:
                frame = cv2.resize(frame, (max(1, round(width * factor)), max(1, round(height * factor))),
                                   interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise ValueError("Local action review sample could not be encoded")
            images.append(encoded.tobytes())
            times.append(float(actual))
            _check_deadline(deadline)
    finally:
        cap.release()
    return images, times


def review_tasks(*, out_dir=DEFAULT_ACTION_DIR, protocol_path=DEFAULT_PROTOCOL, limit=3):
    from src.video_local_observations import request_action_review
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Choose a bounded local review task limit")
    model = verify_ready()  # Check weights and runtime before opening media.
    recipe, recipe_sha = review_recipe()
    queue = load_queue(out_dir, protocol_path)
    protocol = load_protocol(protocol_path)
    directory = Path(out_dir) / "local-review-v3"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.resolve().is_relative_to(config.DATA_RAW.resolve()):
        raise ValueError("Local action review cannot write raw data")
    history = load_reviews(out_dir, protocol_path)
    human = {item["task_id"] for item in history if item["provenance"]["kind"] == "human_review"}
    existing = {item["task_id"] for item in history
                if item["provenance"].get("recipe_sha256") == recipe_sha}
    if limit > 3:
        pilot = [json.loads(path.read_text()) for path in directory.glob("*.json")]
        pilot = [item for item in pilot if item.get("recipe_sha256") == recipe_sha]
        if len(pilot) < 3 or not any(item.get("consensus_present_count", 0) for item in pilot):
            raise ValueError("Bulk local review requires three completed pilot audits with non-unknown observations")
    tasks = [task for task in queue["tasks"] if task["task_id"] not in human | existing][:limit]
    saved = failures = present = disagreements = 0
    for task in tasks:
        if task["group_id"] in protocol["holdout_group_ids"]:
            raise ValueError("Reserved footage cannot enter local action review")
        try:
            deadline = time.monotonic() + 150
            images, timestamps = sample_interval(task, deadline)
            draft = request_action_review(images, timeout=min(45, deadline - time.monotonic()))
            _check_deadline(deadline)
            critique = request_action_review(images, prior_actions=draft["actions"],
                                             timeout=min(45, deadline - time.monotonic()))
            _check_deadline(deadline)
            merged = consensus(draft, critique)
            count = sum(item["presence"] == "present" for item in merged["actions"])
            changed = sum(a != b for a, b in zip(draft["actions"], critique["actions"]))
            provenance = {"kind": "local_model_proposal", "model_id": model["model_id"],
                          "weights_sha256": model["weights_sha256"], "projector_sha256": model["projector_sha256"],
                          "recipe_sha256": recipe_sha, "sample_times_sec": timestamps,
                          "frame_sha256s": [hashlib.sha256(image).hexdigest() for image in images],
                          "review_method": "local draft plus same-model consistency check; unverified"}
            audit = {"task_id": task["task_id"], "source_sha256": task["source_sha256"],
                     "recipe": recipe, "recipe_sha256": recipe_sha, "provenance": provenance,
                     "draft": draft, "critique": critique, "consensus": merged,
                     "consensus_present_count": count, "verified": False}
            target = directory / (task["task_id"] + "-" + recipe_sha[:16] + ".json")
            with target.open("x", encoding="utf-8") as handle:
                target.chmod(0o600)
                handle.write(json.dumps(audit, sort_keys=True) + "\n")
            # Never replace a human record, even if one arrived during inference.
            latest = load_reviews(out_dir, protocol_path)
            if any(item["task_id"] == task["task_id"] and item["provenance"]["kind"] == "human_review"
                   for item in latest):
                continue
            append_proposal(out_dir, task["task_id"], merged["actions"], merged["visibility"],
                            provenance, protocol_path)
            saved += 1; present += count; disagreements += changed
            print(f"Local two-pass action review completed task {saved}/{len(tasks)}", flush=True)
        except (OSError, ValueError, RuntimeError, TimeoutError, cv2.error):
            failures += 1
    return {"tasks_attempted": len(tasks), "machine_rechecked_proposals": saved,
            "tasks_failed": failures, "consensus_present_observations": present,
            "pass_disagreements": disagreements, "verified_labels_created": 0,
            "holdout_groups_used": 0, "semantic_accuracy_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = review_tasks(limit=args.limit)
    except (OSError, ValueError, RuntimeError):
        raise SystemExit("Local action review could not start safely; no media or observations emitted") from None
    print(json.dumps(result), flush=True)
    if result["tasks_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
