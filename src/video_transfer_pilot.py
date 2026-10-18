"""Fixed, human-label-gated object-handoff feasibility pilot; not a violation model.

The default CLI only prepares a private immutable selection. It cannot decode or
train until sixteen independent, eligible human reviews satisfy the frozen fold.
All printed output is aggregate. Model guesses and organizer binary targets are
never used to create action labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.video_eval import load_protocol, validate_protocol, video_metrics


SCHEMA_VERSION = 1
ACTION_ID = "object_handoff"
FOLD_INDEX = 0
FEATURE_COUNT = 600
WINDOW_SECONDS = 8.0
SAMPLE_FPS = 5
THRESHOLD = 0.5
HARD_NEGATIVES = frozenset({"same_person_relocation", "contact_only"})
_SHA = re.compile(r"^[0-9a-f]{64}$")
RECIPE = {
    "schema_version": SCHEMA_VERSION,
    "task": "object_handoff_feasibility_not_bank_violation",
    "fold_index": FOLD_INDEX,
    "per_pool": {"positive": 4, "human_confirmed_hard_negative": 4},
    "selection": "group_id_then_task_id; positives_first; distinct_recordings",
    "blind_release": "hidden_commitment_then_eight_train_only_fit_and_artifact_sha_freeze",
    "hard_negative_types": sorted(HARD_NEGATIVES),
    "positive_window": "full_event_contained; centered_then_clamped_to_clip",
    "negative_window": "8_seconds_entirely_inside_explicit_absence_review",
    "window_seconds": WINDOW_SECONDS,
    "sample_fps": SAMPLE_FPS,
    "encoder_module": "src.video_movinet",
    "features": "frozen_stream_logits_600; independent_zero_state_per_window",
    "classifier": {"scaler": "StandardScaler", "model": "LogisticRegression",
                   "C": 1.0, "solver": "liblinear", "max_iter": 2000,
                   "random_state": config.SEED},
    "threshold": THRESHOLD,
    "gross_failure_gate": {"positive_correct_min": 3, "negative_correct_min": 3},
    "evidence": "human_confirmed_source_time; sampling_alignment_is_not_semantic_verification",
}


class PilotError(ValueError):
    """Safe fixed-message failure, containing no clip-derived content."""


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


RECIPE_SHA256 = fingerprint(RECIPE)


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_export(protocol, labels):
    validate_protocol(protocol)
    if (not isinstance(labels, dict) or labels.get("schema_version") != SCHEMA_VERSION
            or labels.get("protocol_fingerprint") != protocol["fingerprint"]
            or labels.get("fold_index") != FOLD_INDEX
            or not isinstance(labels.get("records"), list)
            or not all(_SHA.fullmatch(str(labels.get(key, "")))
                       for key in ("queue_fingerprint", "annotation_fingerprint"))):
        raise PilotError("Human review export does not match the frozen pilot protocol")


def _example(record, group, pool):
    """Accept only explicit independent human semantics, never weak video labels."""
    if (record.get("source_sha256") != group["group_id"]
            or record.get("partition") != "development" or record.get("pool") != pool):
        raise PilotError("Review source or pool disagrees with the frozen development split")
    provenance = record.get("provenance", {})
    if (record.get("status") != "verified" or record.get("visibility") != "clear"
            or record.get("action_id") != ACTION_ID
            or not isinstance(provenance, dict) or provenance.get("kind") != "human_review"
            or provenance.get("watched") is not True
            or not isinstance(record.get("reviewer_id"), str) or not record["reviewer_id"].strip()):
        return None, "unverified_or_unclear"
    if (not isinstance(record.get("task_id"), str) or not record["task_id"]
            or not _SHA.fullmatch(str(record.get("review_sha256", "")))
            or not isinstance(record.get("review_revision"), int)
            or isinstance(record["review_revision"], bool) or record["review_revision"] < 1):
        raise PilotError("Verified review lacks immutable human provenance")
    target = record.get("target")
    if type(target) is not int or target not in (0, 1):
        return None, "missing_action_semantics"
    interval = record.get("interval", {})
    if not isinstance(interval, dict):
        raise PilotError("Human review interval is invalid")
    start, end, duration = interval.get("start_sec"), interval.get("end_sec"), group["duration_sec"]
    if (not _number(start) or not _number(end) or not 0 <= start < end <= duration
            or record.get("coverage") not in {"interval", "entire_clip"}):
        raise PilotError("Human review interval is outside its original recording")
    if duration < WINDOW_SECONDS:
        return None, "recording_too_short"
    evidence = record.get("evidence_sec")
    if target:
        if (end - start > WINDOW_SECONDS + 1e-8
                or provenance.get("evidence_confirmed") is not True
                or not _number(evidence) or not start <= evidence <= end):
            return None, "positive_interval_or_evidence_unsupported"
        window_start = min(max((start + end - WINDOW_SECONDS) / 2, 0.0), duration - WINDOW_SECONDS)
        if window_start > start + 1e-8 or window_start + WINDOW_SECONDS < end - 1e-8:
            return None, "positive_event_not_fully_contained"
    else:
        if (record.get("hard_negative") is not True
                or record.get("counterexample_type") not in HARD_NEGATIVES
                or provenance.get("absence_confirmed") is not True):
            return None, "not_a_confirmed_hard_negative"
        if end - start < WINDOW_SECONDS - 1e-8:
            return None, "negative_absence_interval_too_short"
        window_start = (start + end - WINDOW_SECONDS) / 2
        evidence = None
    selected = {
        "group_id": group["group_id"], "source_sha256": group["group_id"],
        "task_id": record["task_id"], "pool": pool, "target": target,
        "review_sha256": record["review_sha256"], "review_revision": record["review_revision"],
        "interval": {"start_sec": float(start), "end_sec": float(end)},
        "window": {"start_sec": float(window_start), "end_sec": float(window_start + WINDOW_SECONDS)},
        "evidence_sec": float(evidence) if evidence is not None else None,
        "counterexample_type": record.get("counterexample_type") if not target else None,
    }
    selected["example_sha256"] = fingerprint(selected)
    return selected, None


def _blind_commitment(examples):
    return fingerprint(sorted(({"task_id": row["task_id"], "review_sha256": row["review_sha256"]}
                               for row in examples), key=lambda row: row["task_id"]))


def build_selection(protocol, labels, *, released=False):
    """Pure metadata gate. A blocked return cannot import an encoder or open media."""
    _check_export(protocol, labels)
    groups = {row["group_id"]: row for row in protocol["groups"]}
    fold = protocol["folds"][FOLD_INDEX]
    pools = {"train": set(fold["train_group_ids"]), "blind": set(fold["valid_group_ids"])}
    holdout = set(protocol["holdout_group_ids"])
    eligible, rejected, seen = {"train": [], "blind": []}, Counter(), set()
    for record in labels["records"]:
        if not isinstance(record, dict):
            raise PilotError("Human review export contains an invalid record")
        group_id = record.get("group_id")
        if group_id in holdout or group_id not in set(protocol["development_group_ids"]):
            raise PilotError("Only frozen development recordings may enter the action pilot")
        identity = (record.get("task_id"), record.get("review_revision"))
        if identity in seen:
            raise PilotError("Human review export repeats a review revision")
        seen.add(identity)
        pool = "train" if group_id in pools["train"] else "blind"
        if pool == "blind" and not released:
            raise PilotError("Blind action outcomes must remain hidden until the trained artifact is frozen")
        example, reason = _example(record, groups[group_id], pool)
        if example is None:
            rejected[reason] += 1
        else:
            eligible[pool].append(example)
    selected, counts = [], {}
    for pool in (("train", "blind") if released else ("train",)):
        rows = sorted(eligible[pool], key=lambda row: (row["group_id"], row["task_id"], row["example_sha256"]))
        positives, used = [], set()
        for row in rows:
            if row["target"] == 1 and row["group_id"] not in used:
                positives.append(row)
                used.add(row["group_id"])
        chosen_positives = positives[:4]
        used = {row["group_id"] for row in chosen_positives}
        negatives = []
        for row in rows:
            if row["target"] == 0 and row["group_id"] not in used:
                negatives.append(row)
                used.add(row["group_id"])
        counts[pool] = {"positive_recordings": len(positives),
                        "hard_negative_recordings_after_positive_selection": len(negatives)}
        selected.extend(chosen_positives + negatives[:4])
    if not released:
        quota = labels.get("quota_counts", {}).get("blind", {})
        if (set(quota) != {"positive_recordings", "hard_negative_recordings_after_positive_selection"}
                or any(type(value) is not int or value < 0 for value in quota.values())):
            raise PilotError("Human review export lacks outcome-hidden blind quota counts")
        counts["blind"] = quota
    sufficient = all(counts[pool]["positive_recordings"] >= 4
                     and counts[pool]["hard_negative_recordings_after_positive_selection"] >= 4
                     for pool in counts)
    result = {"schema_version": SCHEMA_VERSION, "status": "ready" if sufficient else "blocked",
              "scope": RECIPE["task"], "protocol_fingerprint": protocol["fingerprint"],
              "queue_fingerprint": labels["queue_fingerprint"],
              "annotation_fingerprint": labels["annotation_fingerprint"],
              "blind_selection_fingerprint": labels.get("blind_selection_fingerprint"),
              "recipe": RECIPE, "recipe_sha256": RECIPE_SHA256,
              "counts": counts, "excluded_counts": dict(sorted(rejected.items())),
              "holdout_opened": 0,
              "required_action": None if sufficient else "Review four action-positive and four explicit hard-negative distinct recordings in each fixed pool locally"}
    if sufficient:
        if not _SHA.fullmatch(str(result["blind_selection_fingerprint"])):
            raise PilotError("Blind selection has not been committed independently of the classifier")
        if released and _blind_commitment([row for row in selected if row["pool"] == "blind"]) != result["blind_selection_fingerprint"]:
            raise PilotError("Released blind selection disagrees with its pretraining commitment")
        result["examples"] = selected
    return result


def encoder_provenance():
    engine = importlib.import_module(RECIPE["encoder_module"])
    if not hasattr(engine, "FEATURE_CONFIG") or not hasattr(engine, "extract_interval"):
        raise PilotError("The frozen temporal encoder contract is not available")
    extra = engine.get_encoder_provenance() if hasattr(engine, "get_encoder_provenance") else {}
    return {"module": RECIPE["encoder_module"], "source_sha256": _digest(engine.__file__),
            "feature_config": engine.FEATURE_CONFIG, "asset_provenance": extra}


def _source_provenance():
    helper = importlib.import_module("scripts.prepare_transfer_review")
    return {"pilot_source_sha256": _digest(__file__),
            "review_helper_sha256": _digest(helper.__file__),
            "video_eval_sha256": _digest(importlib.import_module("src.video_eval").__file__)}


def freeze_manifest(path, selection, encoder_info, sources):
    if selection.get("status") != "ready":
        raise PilotError("Insufficient verified action labels; no pilot manifest or training created")
    manifest = {**selection, "encoder": encoder_info, "sources": sources}
    manifest["manifest_sha256"] = fingerprint(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def validate_manifest(manifest, protocol, labels, encoder_info=None, sources=None):
    _check_export(protocol, labels)
    if (manifest.get("status") != "ready" or manifest.get("recipe") != RECIPE
            or manifest.get("recipe_sha256") != RECIPE_SHA256
            or manifest.get("protocol_fingerprint") != protocol["fingerprint"]
            or manifest.get("queue_fingerprint") != labels["queue_fingerprint"]
            or manifest.get("manifest_sha256") != fingerprint({k: v for k, v in manifest.items() if k != "manifest_sha256"})):
        raise PilotError("Pilot manifest is changed or disagrees with the frozen recipe")
    if encoder_info is not None and manifest.get("encoder") != encoder_info:
        raise PilotError("Temporal encoder provenance changed; preserve this run and use a fresh run")
    if sources is not None and manifest.get("sources") != sources:
        raise PilotError("Pilot or review source changed; preserve this run and use a fresh run")
    records = {(row.get("task_id"), row.get("review_sha256")): row for row in labels["records"]}
    groups = {row["group_id"]: row for row in protocol["groups"]}
    fold = protocol["folds"][FOLD_INDEX]
    allowed = {"train": set(fold["train_group_ids"]), "blind": set(fold["valid_group_ids"])}
    examples = manifest.get("examples", [])
    if (len(examples) != 8 or len({row["group_id"] for row in examples}) != 8
            or any(row["pool"] != "train" for row in examples)
            or Counter(row["target"] for row in examples) != {0: 4, 1: 4}):
        raise PilotError("The prepared manifest must contain only eight balanced training sources")
    if manifest.get("blind_selection_fingerprint") != labels.get("blind_selection_fingerprint"):
        raise PilotError("The independently committed blind selection changed after the recipe freeze")
    for example in examples:
        if example["group_id"] not in allowed.get(example["pool"], set()):
            raise PilotError("Pilot source crosses its frozen fold boundary")
        record = records.get((example["task_id"], example["review_sha256"]))
        if record is None:
            raise PilotError("A selected human review is missing, changed or superseded")
        current, _ = _example(record, groups[example["group_id"]], example["pool"])
        if current != example:
            raise PilotError("Selected human semantics changed after the immutable pilot freeze")
    return manifest


def release_blind_selection(manifest, protocol, labels, release_fingerprint):
    if labels.get("blind_release_fingerprint") != release_fingerprint or not _SHA.fullmatch(str(release_fingerprint)):
        raise PilotError("Blind labels were not released against this frozen trained artifact")
    validate_manifest(manifest, protocol, labels)
    selection = build_selection(protocol, labels, released=True)
    if selection["status"] != "ready":
        raise PilotError("Committed blind labels are no longer supported by human reviews")
    training = [row for row in selection["examples"] if row["pool"] == "train"]
    if training != manifest["examples"]:
        raise PilotError("Training example selection changed after fitting")
    blind = [row for row in selection["examples"] if row["pool"] == "blind"]
    if set(row["group_id"] for row in blind) & set(row["group_id"] for row in training):
        raise PilotError("Training and blind recordings overlap")
    return blind


def _validate_features(example, record, encoder_info=None):
    if (record.get("source_sha256") != example["source_sha256"]
            or record.get("example_sha256") != example["example_sha256"]):
        raise PilotError("Feature record does not match the selected human-reviewed source interval")
    features = np.asarray(record.get("features"), dtype=float)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise PilotError("Temporal feature vector must contain exactly 600 finite logits")
    sampling = record.get("sampling", {})
    requested = np.asarray(sampling.get("requested_times_sec", []), dtype=float)
    actual = np.asarray(sampling.get("actual_times_sec", []), dtype=float)
    indices = sampling.get("source_frame_indices", [])
    hashes = sampling.get("source_frame_sha256", [])
    shapes = sampling.get("source_shapes", [])
    timestamp_sources = sampling.get("timestamp_sources", [])
    expected = example["window"]["start_sec"] + np.arange(40) / SAMPLE_FPS
    if (requested.shape != (40,) or actual.shape != (40,)
            or not np.isfinite(requested).all() or not np.isfinite(actual).all()
            or not np.allclose(requested, expected, rtol=0, atol=1e-6)
            or not np.all(np.diff(actual) > 0) or np.max(np.abs(actual - requested)) > 0.25
            or np.any(actual < requested - 1e-7) or np.any(actual >= example["window"]["end_sec"])
            or len(indices) != 40 or any(type(index) is not int or index < 0 for index in indices)
            or any(right <= left for left, right in zip(indices, indices[1:]))
            or len(hashes) != 40 or any(not _SHA.fullmatch(str(value)) for value in hashes)
            or len(shapes) != 40 or any(not isinstance(shape, list) or len(shape) != 3
                                         or shape[2] != 3 or any(type(value) is not int or value <= 0 for value in shape)
                                         for shape in shapes)
            or len(timestamp_sources) != 40
            or any(value not in {"decoder", "source_frame_index_over_fps"} for value in timestamp_sources)):
        raise PilotError("Temporal sampling audit is missing, misaligned or fabricated")
    if encoder_info is not None:
        expected_provenance = encoder_info["asset_provenance"]
        if (sampling.get("feature_config") != encoder_info["feature_config"]
                or any(record.get("encoder_provenance", {}).get(key) != value
                       for key, value in expected_provenance.items())):
            raise PilotError("Temporal feature audit disagrees with the frozen encoder provenance")
    if example["target"]:
        interval = example["interval"]
        within = np.flatnonzero((actual >= interval["start_sec"]) & (actual <= interval["end_sec"]))
        if not len(within):
            raise PilotError("No actual sampled source frame lies inside the reviewed positive event")
    return features


def extract_features(manifest, protocol, *, root=config.ROOT, extractor=None, encoder=None, deadline=None):
    """Only an explicitly invoked evaluate stage opens the selected 16 sources."""
    if extractor is None:
        engine = importlib.import_module(RECIPE["encoder_module"])
        extractor = engine.extract_interval
        encoder = engine.MoViNetEncoder() if encoder is None else encoder
    groups = {row["group_id"]: row for row in protocol["groups"]}
    records = {}
    for example in manifest["examples"]:
        if example["group_id"] not in set(protocol["development_group_ids"]):
            raise PilotError("Reserved recordings cannot be extracted by this pilot")
        if deadline is not None and time.monotonic() >= deadline:
            raise PilotError("Pilot extraction reached its fixed wall-clock budget")
        source = Path(root) / groups[example["group_id"]]["path"]
        if _digest(source) != example["source_sha256"]:
            raise PilotError("Selected development source checksum changed")
        try:
            result = extractor(source, example["window"]["start_sec"], example["window"]["end_sec"],
                               deadline=deadline, encoder=encoder)
        except Exception:
            raise PilotError("Local temporal extraction failed; inspect its private local audit") from None
        record = {"features": np.asarray(result["features"], dtype=float).tolist(),
                  "sampling": result["sampling"], "source_sha256": example["source_sha256"],
                  "example_sha256": example["example_sha256"],
                  "encoder_provenance": result.get("encoder_provenance", {})}
        _validate_features(example, record, manifest["encoder"])
        records[example["group_id"]] = record
    return records


def _matrix(examples, records, encoder_info):
    if set(records) != {row["group_id"] for row in examples}:
        raise PilotError("Features must match exactly the selected source recordings")
    return np.stack([_validate_features(row, records[row["group_id"]], encoder_info) for row in examples])


def _classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, solver="liblinear", random_state=config.SEED, max_iter=2000))


def fit_training_head(manifest, records):
    """No blind examples/outcomes are accepted by this fitting interface."""
    train = manifest["examples"]
    if (len(train) != 8 or any(row["pool"] != "train" for row in train)
            or len({row["group_id"] for row in train}) != 8
            or Counter(row["target"] for row in train) != {0: 4, 1: 4}):
        raise PilotError("Head fitting accepts only eight independently reviewed balanced training sources")
    x_train = _matrix(train, records, manifest["encoder"])
    model = _classifier()
    model.fit(x_train, [row["target"] for row in train])
    return model


def evaluate_blind_head(manifest, blind, records, model):
    """Score the already fitted, artifact-frozen model without any fitting."""
    train_ids = {row["group_id"] for row in manifest["examples"]}
    if (len(blind) != 8 or any(row["pool"] != "blind" for row in blind)
            or len({row["group_id"] for row in blind}) != 8
            or train_ids & {row["group_id"] for row in blind}
            or Counter(row["target"] for row in blind) != {0: 4, 1: 4}
            or _blind_commitment(blind) != manifest["blind_selection_fingerprint"]):
        raise PilotError("Blind scoring violates its independent balanced pretraining commitment")
    x_blind = _matrix(blind, records, manifest["encoder"])
    probabilities = model.predict_proba(x_blind)[:, 1]
    repeated = model.predict_proba(x_blind)[:, 1]
    metrics = video_metrics([row["target"] for row in blind], probabilities, threshold=THRESHOLD)
    classification_pass = metrics["tp"] >= 3 and metrics["tn"] >= 3
    report = {"schema_version": SCHEMA_VERSION, "scope": RECIPE["task"],
              "manifest_sha256": manifest["manifest_sha256"], "recipe_sha256": RECIPE_SHA256,
              "train_recordings": 8, "blind_recordings": 8, "holdout_opened": 0,
              "action_metrics": metrics, "classification_feasibility_pass": classification_pass,
              "head_repeatability_pass": bool(np.array_equal(probabilities, repeated)),
              "encoder_repeatability_checked": False,
              "reviewed_interval_sampling_alignment_checked": True,
              "evidence_sampling_alignment_checked": False,
              "evidence_semantic_review_required": True,
              "promotion_authorized": False,
              "limitations": ["Eight blind examples are a feasibility check, not a reliable accuracy estimate",
                              "Blind means action outcomes withheld until fitting; these are previously used development recordings",
                              "Action presence is not a banking violation or public-rule finding",
                              "Human-selected event windows do not evaluate continuous event detection or full-video runtime",
                              "Separate fresh encoder repeatability and exact human-confirmed evidence checks remain required"]}
    return report


def _freeze_head(run, manifest, model, features_path):
    import joblib
    model_path = run / "action_head.joblib"
    with model_path.open("xb") as handle:
        joblib.dump({"model": model, "manifest_sha256": manifest["manifest_sha256"],
                     "recipe": RECIPE, "encoder": manifest["encoder"], "scope": RECIPE["task"]}, handle)
    artifact = {"schema_version": SCHEMA_VERSION, "manifest_sha256": manifest["manifest_sha256"],
                "model_sha256": _digest(model_path), "train_features_sha256": _digest(features_path),
                "recipe_sha256": RECIPE_SHA256, "fit_recordings": 8, "blind_seen_during_fit": 0}
    artifact["release_fingerprint"] = fingerprint(artifact)
    _write_new(run / "trained_artifact.json", artifact)
    return artifact


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _arguments(argv, *, sanity=False):
    parser = argparse.ArgumentParser(description=__doc__)
    if not sanity:
        parser.add_argument("--stage", choices=("prepare", "evaluate"), default="prepare")
    parser.add_argument("--protocol", type=Path, default=config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    parser.add_argument("--action-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "video-transfer-pilot-v1")
    return parser.parse_args(argv)


def _inputs(args):
    run = args.run_dir.resolve()
    if not run.is_relative_to(config.DATA_PROC.resolve()):
        raise PilotError("Pilot outputs must stay inside data/processed")
    protocol = load_protocol(args.protocol)
    helper = importlib.import_module("scripts.prepare_transfer_review")
    kwargs = {"protocol_path": args.protocol}
    if args.action_dir is not None:
        kwargs["out_dir"] = args.action_dir
    labels = helper.load_transfer_labels(**kwargs)
    selection = build_selection(protocol, labels)
    return run, protocol, helper, kwargs, labels, selection


def _print_blocked(selection):
    print(json.dumps({key: selection[key] for key in ("status", "scope", "counts", "excluded_counts",
                                                       "holdout_opened", "required_action")}, sort_keys=True))


def _print_failure(error):
    message = str(error) if isinstance(error, PilotError) else "Pilot failed; inspect private local provenance and preserve this run"
    print(json.dumps({"status": "failed", "error": message, "holdout_opened": 0}, sort_keys=True))


def _verify_artifact(run, manifest):
    artifact = json.loads((run / "trained_artifact.json").read_text(encoding="utf-8"))
    if (artifact.get("manifest_sha256") != manifest["manifest_sha256"]
            or artifact.get("recipe_sha256") != RECIPE_SHA256
            or artifact.get("fit_recordings") != 8 or artifact.get("blind_seen_during_fit") != 0
            or artifact.get("model_sha256") != _digest(run / "action_head.joblib")
            or artifact.get("train_features_sha256") != _digest(run / "train_features.json")
            or artifact.get("release_fingerprint") != fingerprint({key: value for key, value in artifact.items() if key != "release_fingerprint"})):
        raise PilotError("Trained artifact or training-feature freeze changed; blind release is forbidden")
    return artifact


def main(argv=None):
    args = _arguments(argv)
    try:
        run, protocol, helper, kwargs, labels, selection = _inputs(args)
        if selection["status"] == "blocked":
            _print_blocked(selection)
            return 2
        encoder_info, sources = encoder_provenance(), _source_provenance()
        manifest_path = run / "manifest.json"
        if manifest_path.exists():
            manifest = validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), protocol, labels,
                                         encoder_info=encoder_info, sources=sources)
        else:
            manifest = freeze_manifest(manifest_path, selection, encoder_info, sources)
        if args.stage == "prepare":
            print(json.dumps({"status": "ready", "train_recordings": 8, "blind_recordings": 8,
                              "holdout_opened": 0, "training_performed": False}, sort_keys=True))
            return 0
        if any((run / name).exists() for name in ("result.json", "train_features.json", "blind_features.json", "action_head.joblib", "trained_artifact.json")):
            raise PilotError("This pilot run was already evaluated or interrupted; preserve it and use a fresh run")
        started = time.monotonic()
        records = extract_features(manifest, protocol, deadline=started + 1800)
        _write_new(run / "train_features.json", {"manifest_sha256": manifest["manifest_sha256"], "records": records})
        model = fit_training_head(manifest, records)
        _freeze_head(run, manifest, model, run / "train_features.json")
        artifact = _verify_artifact(run, manifest)
        # This is the first call permitted to expose blind outcomes, after fit.
        released = helper.load_transfer_labels(**kwargs, include_blind=True,
                                                blind_release_fingerprint=artifact["release_fingerprint"])
        blind = release_blind_selection(manifest, protocol, released, artifact["release_fingerprint"])
        _write_new(run / "blind_selection.json", {"examples": blind, "release_fingerprint": artifact["release_fingerprint"]})
        blind_records = extract_features({**manifest, "examples": blind}, protocol, deadline=started + 1800)
        _write_new(run / "blind_features.json", {"manifest_sha256": manifest["manifest_sha256"], "records": blind_records})
        report = evaluate_blind_head(manifest, blind, blind_records, model)
        report["elapsed_sec"] = time.monotonic() - started
        report["release_fingerprint"] = artifact["release_fingerprint"]
        report["blind_features_sha256"] = _digest(run / "blind_features.json")
        report["blind_selection_sha256"] = _digest(run / "blind_selection.json")
        _write_new(run / "result.json", report)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["classification_feasibility_pass"] else 3
    except Exception as error:
        # Library errors and private paths/labels/model responses must not escape.
        _print_failure(error)
        return 1


def sanity_features(manifest, train_records, blind, blind_records):
    """Twenty fixed-seed train-label permutations; blind targets never fit a model."""
    if ({row["group_id"] for row in manifest["examples"]} & {row["group_id"] for row in blind}
            or len(manifest["examples"]) != 8 or len(blind) != 8
            or any(row["pool"] != "train" for row in manifest["examples"])
            or any(row["pool"] != "blind" for row in blind)
            or Counter(row["target"] for row in manifest["examples"]) != {0: 4, 1: 4}
            or Counter(row["target"] for row in blind) != {0: 4, 1: 4}
            or _blind_commitment(blind) != manifest["blind_selection_fingerprint"]):
        raise PilotError("Shuffled sanity requires the same independent committed pilot pools")
    x_train = _matrix(manifest["examples"], train_records, manifest["encoder"])
    x_blind = _matrix(blind, blind_records, manifest["encoder"])
    train_y = np.asarray([row["target"] for row in manifest["examples"]])
    blind_y = [row["target"] for row in blind]
    rng, scores = np.random.default_rng(config.SEED), []
    for _ in range(20):
        fitted = _classifier().fit(x_train, rng.permutation(train_y))
        scores.append(video_metrics(blind_y, fitted.predict_proba(x_blind)[:, 1])["roc_auc"])
    mean = float(np.mean(scores))
    return {"scope": RECIPE["task"], "shuffled_trials": 20, "shuffled_auc_mean": mean,
            "shuffled_auc_std": float(np.std(scores)), "chance_tolerance": 0.1,
            "chance_check_pass": abs(mean - 0.5) <= 0.1,
            "shared_recordings": 0, "holdout_opened": 0, "source_hashes_verified": False,
            "limitations": "Eight blind examples make this a coarse leakage diagnostic, not proof of generalization"}


def sanity_main(argv=None):
    args = _arguments(argv, sanity=True)
    try:
        run, protocol, helper, kwargs, labels, selection = _inputs(args)
        if selection["status"] == "blocked":
            _print_blocked(selection)
            return 2
        if not (run / "result.json").is_file():
            print(json.dumps({"status": "blocked", "required_action": "Run the authorized fixed pilot before sanity; no scores exist", "holdout_opened": 0}))
            return 2
        manifest = validate_manifest(json.loads((run / "manifest.json").read_text(encoding="utf-8")), protocol, labels,
                                     encoder_info=encoder_provenance(), sources=_source_provenance())
        artifact = _verify_artifact(run, manifest)
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        if (result.get("manifest_sha256") != manifest["manifest_sha256"]
                or result.get("release_fingerprint") != artifact["release_fingerprint"]
                or result.get("blind_features_sha256") != _digest(run / "blind_features.json")
                or result.get("blind_selection_sha256") != _digest(run / "blind_selection.json")):
            raise PilotError("Saved blind evaluation changed; preserve the frozen run")
        released = helper.load_transfer_labels(**kwargs, include_blind=True,
                                                blind_release_fingerprint=artifact["release_fingerprint"])
        blind = release_blind_selection(manifest, protocol, released, artifact["release_fingerprint"])
        train_records = json.loads((run / "train_features.json").read_text(encoding="utf-8"))["records"]
        blind_records = json.loads((run / "blind_features.json").read_text(encoding="utf-8"))["records"]
        groups = {row["group_id"]: row for row in protocol["groups"]}
        for row in manifest["examples"] + blind:
            if _digest(config.ROOT / groups[row["group_id"]]["path"]) != row["source_sha256"]:
                raise PilotError("A selected development source checksum changed")
        import joblib
        saved = joblib.load(run / "action_head.joblib")
        if saved.get("manifest_sha256") != manifest["manifest_sha256"] or saved.get("recipe") != RECIPE:
            raise PilotError("Saved head disagrees with its frozen pilot metadata")
        replay = evaluate_blind_head(manifest, blind, blind_records, saved["model"])
        if replay["action_metrics"] != result["action_metrics"]:
            raise PilotError("Saved head does not reproduce its original blind result")
        report = sanity_features(manifest, train_records, blind, blind_records)
        report["source_hashes_verified"] = True
        _write_new(run / "sanity.json", report)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["chance_check_pass"] else 3
    except Exception as error:
        _print_failure(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
