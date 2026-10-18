"""Local action-label metadata and append-only review provenance.

This module never opens video footage. Official binary labels/timestamps are
weak review anchors, never action labels. Only development content groups may
enter the queue; proposals are excluded from independently reviewed training.
"""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re

from src import config
from src.video_eval import load_protocol


DEFAULT_PROTOCOL = config.DATA_PROC / "fortebank-20261017/video_protocol.json"
DEFAULT_ACTION_DIR = config.DATA_PROC / "action-labels"
ACTION_VOCABULARY = {
    "object_handoff": "Visible object handoff between people",
    "object_relocation": "Visible object moved between locations",
    "repeated_separate_restack": "Repeated visible separation/restacking",
    "container_open_close": "Visible container, lid or door opening/closing",
    "person_region_transition": "Visible person entering/leaving a defined region",
    "hand_object_contact": "Visible hand contacting an object",
    "handheld_device_interaction": "Visible interaction with a handheld device",
    "unknown_observable_action": "Observable action cannot be classified reliably",
}
ACTION_IDS = tuple(ACTION_VOCABULARY)
TAXONOMY = ACTION_VOCABULARY
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVIEWER = re.compile(r"^[\w .@-]{1,64}$", re.UNICODE)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _directory(out_dir):
    directory = Path(out_dir).resolve()
    if directory.is_relative_to(config.DATA_RAW.resolve()):
        raise ValueError("Action labels must never write to data/raw")
    return directory


def _build_queue(protocol):
    development = set(protocol["development_group_ids"])
    holdout = set(protocol["holdout_group_ids"])
    if development & holdout:
        raise ValueError("Reserved holdout cannot enter the action queue")
    tasks = []
    for group in protocol["groups"]:
        if group["group_id"] not in development:
            continue
        duration = float(group["duration_sec"])
        source_annotations = group["timestamp_annotations"]
        anchors = sorted({float(item["timestamp_sec"]) for item in source_annotations
                          if item["timestamp_sec"] is not None})
        ambiguous = bool(group["target"] and not group["localization_eligible"])
        for anchor in anchors or [None]:
            identifier = {"protocol_fingerprint": protocol["fingerprint"],
                          "group_id": group["group_id"], "anchor_sec": anchor}
            tasks.append({
                "task_id": _digest(identifier), "group_id": group["group_id"],
                "path": group["path"], "source_sha256": group["sha256"],
                "duration_sec": duration, "official_target": int(group["target"]),
                "anchor_sec": anchor,
                "interval": {"start_sec": max(0.0, anchor - 4.0) if anchor is not None else 0.0,
                             "end_sec": min(duration, anchor + 4.0) if anchor is not None else duration},
                "ambiguous_anchor": ambiguous,
                "source_annotations": [dict(item) for item in source_annotations],
                "partition": "development", "initial_status": "unknown", "actions": [],
            })
    # Anchored reviews first; identifiers and order are independent of filenames.
    tasks.sort(key=lambda task: (task["anchor_sec"] is None,
                                task["group_id"], task["anchor_sec"] or 0.0))
    return {"schema_version": 1, "protocol_fingerprint": protocol["fingerprint"],
            "vocabulary_version": 1, "action_vocabulary": dict(ACTION_VOCABULARY),
            "tasks": tasks}


def prepare_queue(protocol_path=DEFAULT_PROTOCOL, out_dir=DEFAULT_ACTION_DIR):
    """Create an immutable development-only queue without opening any footage."""
    expected = _build_queue(load_protocol(protocol_path))
    directory = _directory(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "queue.json"
    if path.exists():
        return load_queue(directory, protocol_path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(expected, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)
    return expected


def load_queue(out_dir=DEFAULT_ACTION_DIR, protocol_path=DEFAULT_PROTOCOL):
    """Check every task against the frozen protocol; no media bytes are read."""
    path = _directory(out_dir) / "queue.json"
    with path.open(encoding="utf-8") as handle:
        queue = json.load(handle)
    if queue != _build_queue(load_protocol(protocol_path)):
        raise ValueError("Action queue differs from frozen development protocol or vocabulary")
    return queue


def _interval(value, duration):
    if not isinstance(value, dict) or set(value) != {"start_sec", "end_sec"}:
        raise ValueError("Review requires start_sec and end_sec")
    try:
        start, end = float(value["start_sec"]), float(value["end_sec"])
    except (TypeError, ValueError):
        raise ValueError("Review interval must contain numeric seconds") from None
    if (isinstance(value["start_sec"], bool) or isinstance(value["end_sec"], bool)
            or not math.isfinite(start) or not math.isfinite(end)
            or not 0 <= start < end <= duration):
        raise ValueError("Review interval must be finite, nonempty and within clip duration")
    return {"start_sec": start, "end_sec": end}


def _review_payload(review, queue):
    if not isinstance(review, dict):
        raise ValueError("Review must be an object")
    by_task = {task["task_id"]: task for task in queue["tasks"]}
    task = by_task.get(review.get("task_id"))
    if task is None:
        raise ValueError("Review task is not in the frozen development queue")
    # Metadata supplied by a caller must agree with the frozen task, never win.
    immutable = {"protocol_fingerprint": queue["protocol_fingerprint"],
                 "group_id": task["group_id"], "source_sha256": task["source_sha256"],
                 "partition": "development"}
    for key, value in immutable.items():
        if key in review and review[key] != value:
            raise ValueError("Review metadata differs from the frozen development task")
    interval = _interval(review.get("interval"), task["duration_sec"])
    coverage = review.get("coverage")
    if coverage not in {"interval", "entire_clip"}:
        raise ValueError("Review coverage must be interval or entire_clip")
    if coverage == "entire_clip" and interval != {"start_sec": 0.0, "end_sec": task["duration_sec"]}:
        raise ValueError("Entire-clip coverage requires the complete duration")
    visibility = review.get("visibility")
    if visibility not in {"clear", "partial", "unobservable"}:
        raise ValueError("Review visibility must be clear, partial or unobservable")
    status = review.get("status")
    if status not in {"verified", "ambiguous", "proposed"}:
        raise ValueError("Review status must be verified, ambiguous or proposed")
    provenance = review.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Review provenance is required")
    kind = provenance.get("kind")
    if kind not in {"human_review", "local_model_proposal"}:
        raise ValueError("Action labels require human or local-model provenance")
    reviewer = review.get("reviewer_id")
    if kind == "human_review":
        if not isinstance(reviewer, str) or not _REVIEWER.fullmatch(reviewer) or not reviewer.strip():
            raise ValueError("Human review requires a valid local reviewer_id")
        if status == "proposed":
            raise ValueError("Human review must be verified or ambiguous")
    else:
        if status != "proposed" or reviewer is not None:
            raise ValueError("Model proposals cannot be verified or identify a human reviewer")
        if not isinstance(provenance.get("model_id"), str) or not provenance["model_id"].strip():
            raise ValueError("Model proposals require a model_id")
        for key in ("weights_sha256", "recipe_sha256"):
            if not _SHA256.fullmatch(str(provenance.get(key, ""))):
                raise ValueError("Model proposals require pinned weights and recipe SHA-256")
        frame_hashes = provenance.get("frame_sha256s", [])
        if (not isinstance(frame_hashes, list)
                or any(not _SHA256.fullmatch(str(value)) for value in frame_hashes)):
            raise ValueError("Model proposal frame hashes must be SHA-256 values")
    actions = review.get("actions")
    if not isinstance(actions, list) or len(actions) > len(ACTION_VOCABULARY):
        raise ValueError("Review actions must be a bounded list")
    validated_actions, seen = [], set()
    for action in actions:
        if not isinstance(action, dict) or set(action) != {"action_id", "presence"}:
            raise ValueError("Each action requires action_id and presence only")
        action_id, presence = action["action_id"], action["presence"]
        if action_id not in ACTION_VOCABULARY or action_id in seen:
            raise ValueError("Action vocabulary IDs must be known and unique per review")
        if presence not in {"present", "absent", "unknown"}:
            raise ValueError("Action presence must be present, absent or unknown")
        if status == "verified" and (presence == "unknown" or action_id == "unknown_observable_action"):
            raise ValueError("Unknown actions must remain ambiguous, not verified truth")
        if status == "verified" and presence == "absent" and visibility != "clear":
            raise ValueError("Verified action absence requires clear interval visibility")
        seen.add(action_id)
        validated_actions.append({"action_id": action_id, "presence": presence})
    if status == "verified" and (not actions or visibility == "unobservable"):
        raise ValueError("Verified labels require observable, nonempty action observations")
    note = review.get("note", "")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("Local review note must be at most 2000 characters")
    return {"schema_version": 1, "task_id": task["task_id"], **immutable,
            "interval": interval, "coverage": coverage, "visibility": visibility,
            "status": status, "reviewer_id": reviewer, "actions": validated_actions,
            "provenance": dict(provenance), "note": note}


def _read_reviews(handle, queue):
    records, per_task = [], {}
    for number, line in enumerate(handle, start=1):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            raise ValueError("Action review journal contains invalid JSON") from None
        canonical = _review_payload(record, queue)
        expected_revision = per_task.get(record["task_id"], 0) + 1
        if (record.get("revision") != expected_revision or record.get("sequence") != number
                or not isinstance(record.get("created_at"), str)):
            raise ValueError("Action review revisions or journal sequence are invalid")
        canonical.update(revision=record["revision"], sequence=record["sequence"],
                         created_at=record["created_at"],
                         previous_record_sha256=record.get("previous_record_sha256"))
        expected_previous = records[-1]["record_sha256"] if records else None
        if (canonical["previous_record_sha256"] != expected_previous
                or record.get("record_sha256") != _digest(canonical)
                or set(record) != set(canonical) | {"record_sha256"}):
            raise ValueError("Action review journal provenance hash chain is invalid")
        per_task[record["task_id"]] = expected_revision
        records.append(record)
    return records


def load_reviews(out_dir=DEFAULT_ACTION_DIR, protocol_path=DEFAULT_PROTOCOL):
    """Read local sensitive annotations; callers must not emit their contents."""
    queue = load_queue(out_dir, protocol_path)
    path = _directory(out_dir) / "reviews.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        return _read_reviews(handle, queue)


def append_review(review, out_dir=DEFAULT_ACTION_DIR, protocol_path=DEFAULT_PROTOCOL):
    """Append one checked revision; no existing annotation is overwritten."""
    queue = load_queue(out_dir, protocol_path)
    record = _review_payload(review, queue)
    path = _directory(out_dir) / "reviews.jsonl"
    # One locked append prevents concurrent local reviewers losing revisions.
    with path.open("a+", encoding="utf-8") as handle:
        path.chmod(0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        history = _read_reviews(handle, queue)
        record.update(
            revision=1 + sum(item["task_id"] == record["task_id"] for item in history),
            sequence=len(history) + 1,
            created_at=datetime.now(timezone.utc).isoformat(),
            previous_record_sha256=history[-1]["record_sha256"] if history else None,
        )
        record["record_sha256"] = _digest(record)
        handle.write(_canonical(record) + "\n")
        handle.flush()
    return record


def append_proposal(out_dir, task_id, actions, visibility, provenance,
                    protocol_path=DEFAULT_PROTOCOL, interval=None):
    """Append a pinned local-model proposal, never a verified action label.

    Per-action visibility from a structured local model may be supplied; an
    unobservable action is stored with unknown presence, never as a fact.
    """
    queue = load_queue(out_dir, protocol_path)
    task = next((item for item in queue["tasks"] if item["task_id"] == task_id), None)
    if task is None:
        raise ValueError("Proposal task is not in the frozen development queue")
    if not isinstance(actions, list):
        raise ValueError("Proposal actions must be a list")
    observations = []
    for action in actions:
        if (not isinstance(action, dict) or not {"action_id", "presence"} <= set(action)
                or set(action) - {"action_id", "presence", "visibility"}):
            raise ValueError("Proposal actions require action_id, presence and optional visibility")
        action_visibility = action.get("visibility", visibility)
        if action_visibility not in {"clear", "partial", "unobservable"}:
            raise ValueError("Proposal action visibility is invalid")
        observations.append({"action_id": action["action_id"],
                             "presence": "unknown" if action_visibility == "unobservable"
                             else action["presence"]})
    if not isinstance(provenance, dict) or provenance.get("kind", "local_model_proposal") != "local_model_proposal":
        raise ValueError("Proposal provenance must identify a local model")
    provenance = {**provenance, "kind": "local_model_proposal"}
    return append_review({
        "task_id": task_id, "status": "proposed", "reviewer_id": None,
        "interval": dict(task["interval"] if interval is None else interval),
        "coverage": "interval", "visibility": visibility,
        "actions": observations, "provenance": provenance,
    }, out_dir, protocol_path)


def _latest_human_reviews(records):
    latest = {}
    for record in records:
        if record["provenance"]["kind"] == "human_review":
            latest[record["task_id"]] = record
    return latest


def verified_training_labels(out_dir=DEFAULT_ACTION_DIR, protocol_path=DEFAULT_PROTOCOL):
    """Export only latest independently reviewed, known action observations.

    This does not fit a model or expand an interval into whole-video labels.
    A later ambiguous human revision revokes an earlier verified annotation.
    Local model proposals cannot overwrite or manufacture reviewed truth.
    """
    records = load_reviews(out_dir, protocol_path)
    output = []
    for record in _latest_human_reviews(records).values():
        if record["status"] != "verified":
            continue
        for action in record["actions"]:
            output.append({
                "task_id": record["task_id"], "group_id": record["group_id"],
                "source_sha256": record["source_sha256"], "partition": "development",
                "protocol_fingerprint": record["protocol_fingerprint"],
                "interval": dict(record["interval"]), "coverage": record["coverage"],
                "action_id": action["action_id"], "target": int(action["presence"] == "present"),
                "review_sha256": record["record_sha256"], "review_revision": record["revision"],
            })
    return output


def aggregate_summary(out_dir=DEFAULT_ACTION_DIR, protocol_path=DEFAULT_PROTOCOL):
    """Return nonsensitive aggregate status only, safe for progress reporting."""
    queue = load_queue(out_dir, protocol_path)
    records = load_reviews(out_dir, protocol_path)
    latest = _latest_human_reviews(records)
    statuses = Counter(record["status"] for record in latest.values())
    training = verified_training_labels(out_dir, protocol_path)
    return {
        "development_groups": len({task["group_id"] for task in queue["tasks"]}),
        "tasks": len(queue["tasks"]),
        "anchored_tasks": sum(task["anchor_sec"] is not None for task in queue["tasks"]),
        "ambiguous_anchor_tasks": sum(task["ambiguous_anchor"] for task in queue["tasks"]),
        "human_reviewed_tasks": len(latest), "verified_tasks": statuses["verified"],
        "ambiguous_tasks": statuses["ambiguous"],
        "unreviewed_tasks": len(queue["tasks"]) - len(latest),
        "proposal_records": sum(record["status"] == "proposed" for record in records),
        "review_revisions": len(records), "verified_action_observations": len(training),
        "verified_action_classes": len({item["action_id"] for item in training}),
        "holdout_groups": 0,
    }
