"""Prepare and validate local human transfer reviews; never open video footage."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re

from src import config, video_actions as actions
from src.video_eval import load_protocol


DEFAULT_TRANSFER_DIR = actions.DEFAULT_ACTION_DIR / "transfer-pilot-v1"
DEFINITION = ("A visible object passes from one person's hand/control to another person's "
              "hand/control. Both sides of the exchange must be observable. This is an action, "
              "not a violation judgment; do not infer ownership, role or intent.")
CHOICES = {
    "transfer": "An object passes between two people",
    "same_person_relocation": "The same person moves an object; no transfer between people",
    "contact_only": "Hand/object contact, but no transfer between people",
    "background": "Other activity; no transfer in the watched interval",
    "uncertain": "I cannot tell reliably",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _directory(out_dir):
    path = Path(out_dir).resolve()
    if not path.is_relative_to(config.DATA_PROC.resolve()):
        raise ValueError("Transfer review output must be within data/processed")
    return path


def _expected(protocol):
    fold = protocol["folds"][0]
    train, blind = set(fold["train_group_ids"]), set(fold["valid_group_ids"])
    tasks = []
    for task in actions._build_queue(protocol)["tasks"]:
        item = {key: task[key] for key in ("task_id", "group_id", "path", "source_sha256",
                "duration_sec", "anchor_sec", "interval", "ambiguous_anchor")}
        item.update(pool="train" if task["group_id"] in train else "blind",
                    partition="development", initial_status="unknown")
        tasks.append(item)
    if ({task["group_id"] for task in tasks} != train | blind
            or train & blind or (train | blind) & set(protocol["holdout_group_ids"])):
        raise ValueError("Transfer queue partitions differ from frozen development fold")
    initial, seen = [], set()
    for pool in ("train", "blind"):
        for task in tasks:
            if task["pool"] == pool and task["group_id"] not in seen:
                initial.append(task["task_id"])
                seen.add(task["group_id"])
                if sum(item["pool"] == pool and item["task_id"] in initial for item in tasks) == 8:
                    break
    order = {task_id: index for index, task_id in enumerate(initial)}
    tasks.sort(key=lambda task: (order.get(task["task_id"], len(initial)), task["task_id"]))
    manifest = {"schema_version": 1, "protocol_fingerprint": protocol["fingerprint"],
                "fold_index": 0, "action_id": "object_handoff", "definition": DEFINITION,
                "choices": CHOICES, "tasks": tasks, "initial_task_ids": initial,
                "selection_requirement": {"train": {"positive": 4, "hard_negative": 4},
                                          "blind": {"positive": 4, "hard_negative": 4}},
                "labels_prepopulated": False}
    return {**manifest, "queue_fingerprint": _digest(manifest)}


def prepare_transfer_queue(out_dir=DEFAULT_TRANSFER_DIR, protocol_path=actions.DEFAULT_PROTOCOL):
    expected = _expected(load_protocol(protocol_path))
    directory = _directory(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "queue.json"
    if path.exists():
        return load_transfer_queue(directory, protocol_path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(expected, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)
    return expected


def load_transfer_queue(out_dir=DEFAULT_TRANSFER_DIR, protocol_path=actions.DEFAULT_PROTOCOL):
    with (_directory(out_dir) / "queue.json").open(encoding="utf-8") as handle:
        queue = json.load(handle)
    if queue != _expected(load_protocol(protocol_path)):
        raise ValueError("Transfer queue differs from frozen development protocol")
    return queue


def _payload(value, queue):
    if not isinstance(value, dict):
        raise ValueError("Transfer review must be an object")
    task = next((task for task in queue["tasks"] if task["task_id"] == value.get("task_id")), None)
    if task is None:
        raise ValueError("Transfer review is not a development task")
    reviewer = value.get("reviewer_id")
    if (not isinstance(reviewer, str) or not reviewer.strip()
            or re.fullmatch(r"[\w .@-]{1,64}", reviewer, re.UNICODE) is None):
        raise ValueError("A local human reviewer name is required")
    choice, visibility = value.get("choice"), value.get("visibility")
    if choice not in CHOICES or visibility not in {"clear", "partial", "unobservable"}:
        raise ValueError("Choose an observable outcome and visibility")
    if value.get("watched") is not True:
        raise ValueError("The human reviewer must confirm watching the selected coverage")
    interval = actions._interval(value.get("interval"), task["duration_sec"])
    coverage = value.get("coverage")
    if coverage not in {"interval", "entire_clip"}:
        raise ValueError("Coverage must be interval or entire_clip")
    if coverage == "entire_clip" and interval != {"start_sec": 0.0, "end_sec": task["duration_sec"]}:
        raise ValueError("Entire-clip coverage must include the complete duration")
    evidence = value.get("evidence_sec")
    if evidence is not None:
        if (isinstance(evidence, bool) or not isinstance(evidence, (int, float))
                or not math.isfinite(evidence)
                or not interval["start_sec"] <= evidence <= interval["end_sec"]):
            raise ValueError("Evidence time must lie within the reviewed interval")
        evidence = float(evidence)
    status = "ambiguous" if choice == "uncertain" else "verified"
    if status == "verified" and visibility != "clear":
        raise ValueError("Unclear coverage must be saved as uncertain")
    if choice == "transfer":
        if interval["end_sec"] - interval["start_sec"] > 8.0:
            raise ValueError("Mark the transfer event within an interval of at most eight seconds")
        if evidence is None or value.get("evidence_confirmed") is not True:
            raise ValueError("Select and confirm an original supporting frame for the transfer")
    elif choice != "uncertain":
        if interval["end_sec"] - interval["start_sec"] < 8.0 or value.get("absence_confirmed") is not True:
            raise ValueError("A negative needs at least eight fully watched seconds and explicit no-transfer confirmation")
    provenance = {"kind": "human_review", "watched": True,
                  "absence_confirmed": value.get("absence_confirmed") is True,
                  "evidence_confirmed": value.get("evidence_confirmed") is True}
    if "provenance" in value and value["provenance"] != provenance:
        raise ValueError("Transfer labels require explicit human-review provenance")
    record = {"schema_version": 1, "task_id": task["task_id"], "group_id": task["group_id"],
              "source_sha256": task["source_sha256"], "partition": "development", "pool": task["pool"],
              "protocol_fingerprint": queue["protocol_fingerprint"],
              "queue_fingerprint": queue["queue_fingerprint"], "action_id": "object_handoff",
              "reviewer_id": reviewer, "choice": choice, "status": status,
              "visibility": visibility, "interval": interval, "coverage": coverage,
              "evidence_sec": evidence, "provenance": provenance}
    for key in ("group_id", "source_sha256", "partition", "pool", "protocol_fingerprint", "queue_fingerprint", "action_id", "status"):
        if key in value and value[key] != record[key]:
            raise ValueError("Transfer review metadata differs from the frozen task")
    return record


def _read(handle, queue):
    records, revisions = [], Counter()
    for sequence, line in enumerate(handle, 1):
        value = json.loads(line)
        supplied = {**value, "watched": value["provenance"]["watched"],
                    "absence_confirmed": value["provenance"]["absence_confirmed"],
                    "evidence_confirmed": value["provenance"]["evidence_confirmed"]}
        record = _payload(supplied, queue)
        revisions[record["task_id"]] += 1
        record.update(sequence=sequence, revision=revisions[record["task_id"]],
                      created_at=value.get("created_at"),
                      previous_record_sha256=records[-1]["record_sha256"] if records else None)
        if (not isinstance(record["created_at"], str) or set(value) != set(record) | {"record_sha256"}
                or any(value.get(key) != item for key, item in record.items())
                or value["record_sha256"] != _digest(record)):
            raise ValueError("Transfer review journal hash chain or revision is invalid")
        records.append(value)
    return records


def load_transfer_reviews(out_dir=DEFAULT_TRANSFER_DIR, protocol_path=actions.DEFAULT_PROTOCOL):
    queue = load_transfer_queue(out_dir, protocol_path)
    path = _directory(out_dir) / "reviews.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        return _read(handle, queue)


def append_transfer_review(value, out_dir=DEFAULT_TRANSFER_DIR, protocol_path=actions.DEFAULT_PROTOCOL):
    queue = load_transfer_queue(out_dir, protocol_path)
    record = _payload(value, queue)
    path = _directory(out_dir) / "reviews.jsonl"
    if path.is_symlink():
        raise ValueError("Transfer review journal must not be a symlink")
    with path.open("a+", encoding="utf-8") as handle:
        path.chmod(0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        history = _read(handle, queue)
        record.update(sequence=len(history) + 1,
                      revision=1 + sum(item["task_id"] == record["task_id"] for item in history),
                      created_at=datetime.now(timezone.utc).isoformat(),
                      previous_record_sha256=history[-1]["record_sha256"] if history else None)
        record["record_sha256"] = _digest(record)
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
    return record


def load_transfer_labels(out_dir=DEFAULT_TRANSFER_DIR, protocol_path=actions.DEFAULT_PROTOCOL,
                         *, include_blind=False, blind_release_fingerprint=None):
    """Return sensitive, independently reviewed intervals for a local selector only."""
    queue = load_transfer_queue(out_dir, protocol_path)
    latest = {record["task_id"]: record for record in load_transfer_reviews(out_dir, protocol_path)}
    records = []
    for record in sorted(latest.values(), key=lambda item: item["task_id"]):
        if record["status"] != "verified":
            continue
        records.append({**record, "target": int(record["choice"] == "transfer"),
                        "counterexample_type": None if record["choice"] == "transfer" else record["choice"],
                        "hard_negative": record["choice"] in {"same_person_relocation", "contact_only"},
                        "review_sha256": record["record_sha256"], "review_revision": record["revision"]})
    quota_counts, blind_selected = {}, []
    for pool in ("train", "blind"):
        candidates = sorted((record for record in records if record["pool"] == pool),
                            key=lambda record: (record["group_id"], record["task_id"]))
        positive, seen = [], set()
        for record in candidates:
            if record["target"] == 1 and record["group_id"] not in seen:
                positive.append(record)
                seen.add(record["group_id"])
        selected_positive = positive[:4]
        seen = {record["group_id"] for record in selected_positive}
        negative = []
        for record in candidates:
            if record["hard_negative"] and record["group_id"] not in seen:
                negative.append(record)
                seen.add(record["group_id"])
        quota_counts[pool] = {"positive_recordings": len(positive),
                              "hard_negative_recordings_after_positive_selection": len(negative)}
        if pool == "blind" and len(selected_positive) == 4 and len(negative) >= 4:
            blind_selected = selected_positive + negative[:4]
    commitment = ([{"task_id": record["task_id"], "review_sha256": record["review_sha256"]}
                   for record in sorted(blind_selected, key=lambda record: record["task_id"])])
    if include_blind and (not isinstance(blind_release_fingerprint, str)
                          or re.fullmatch(r"[0-9a-f]{64}", blind_release_fingerprint) is None):
        raise ValueError("Blind outcomes require the independently frozen recipe/artifact manifest fingerprint")
    return {"schema_version": 1, "protocol_fingerprint": queue["protocol_fingerprint"],
            "queue_fingerprint": queue["queue_fingerprint"], "fold_index": 0,
            "annotation_fingerprint": _digest({"queue": queue["queue_fingerprint"], "records": records}),
            "quota_counts": quota_counts,
            "blind_selection_fingerprint": _digest(commitment) if blind_selected else None,
            "blind_release_fingerprint": blind_release_fingerprint if include_blind else None,
            "records": records if include_blind else [record for record in records if record["pool"] == "train"]}


def transfer_summary(out_dir=DEFAULT_TRANSFER_DIR, protocol_path=actions.DEFAULT_PROTOCOL):
    queue = load_transfer_queue(out_dir, protocol_path)
    history = load_transfer_reviews(out_dir, protocol_path)
    latest = {record["task_id"]: record for record in history}
    labels = [record for record in latest.values() if record["status"] == "verified"]
    pools = {}
    for pool in ("train", "blind"):
        positive = {record["group_id"] for record in labels if record["pool"] == pool and record["choice"] == "transfer"}
        negative = {record["group_id"] for record in labels if record["pool"] == pool and record["choice"] in {"same_person_relocation", "contact_only"}}
        pools[pool] = {"positive_groups": len(positive), "hard_negative_groups": len(negative),
                       "required_each": 4}
    return {"development_groups": len({task["group_id"] for task in queue["tasks"]}),
            "tasks": len(queue["tasks"]), "human_reviewed_tasks": len(latest),
            "verified_tasks": sum(record["status"] == "verified" for record in latest.values()),
            "ambiguous_tasks": sum(record["status"] == "ambiguous" for record in latest.values()),
            "unreviewed_tasks": len(queue["tasks"]) - len(latest), "review_revisions": len(history),
            "pools": pools, "holdout_groups": 0,
            "quota_counts_are_not_selection": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TRANSFER_DIR)
    parser.add_argument("--protocol", type=Path, default=actions.DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    prepare_transfer_queue(args.out_dir, args.protocol)
    print(json.dumps(transfer_summary(args.out_dir, args.protocol), sort_keys=True))


if __name__ == "__main__":
    main()
