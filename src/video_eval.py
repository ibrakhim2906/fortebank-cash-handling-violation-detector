"""Frozen, content-grouped video validation; never opens video footage.

This is the approved video adapter. The generic ``src.eval`` stays unchanged.
The reserved holdout is for a later, frozen-policy evaluation, not P2 tuning.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import config


HOLDOUT_SIZE = 8
N_VIDEO_FOLDS = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _fingerprint(protocol):
    payload = {key: value for key, value in protocol.items() if key != "fingerprint"}
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"Expected a boolean inventory value, received {value!r}")


def _inventory_groups(inventory_path):
    frame = pd.read_csv(inventory_path)
    required = {"video_id", "path", "sha256", "target", "timestamp_sec", "duration_sec"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Inventory is missing columns: {sorted(missing)}")
    if frame.empty or frame[list(required - {"timestamp_sec"})].isna().any().any():
        raise ValueError("Inventory identifiers, labels and durations must be complete")
    if frame.video_id.duplicated().any() or frame.path.duplicated().any():
        raise ValueError("Inventory video identifiers and paths must be unique")
    labels = pd.to_numeric(frame.target, errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Video targets must be binary 0/1")
    frame["target"] = labels.astype(int)
    frame["timestamp_sec"] = pd.to_numeric(frame.timestamp_sec, errors="raise")
    frame["duration_sec"] = pd.to_numeric(frame.duration_sec, errors="raise")
    if not np.isfinite(frame.duration_sec).all() or (frame.duration_sec <= 0).any():
        raise ValueError("Video durations must be finite and positive")
    if not frame.sha256.map(lambda value: bool(_SHA256.fullmatch(str(value)))).all():
        raise ValueError("Inventory requires full lowercase SHA-256 content hashes")
    if "group_id" in frame and not frame.group_id.eq(frame.sha256).all():
        raise ValueError("Inventory group_id must equal the complete content hash")
    positive = frame.target.eq(1)
    valid_times = np.isfinite(frame.timestamp_sec) & frame.timestamp_sec.ge(0)
    valid_times &= frame.timestamp_sec.le(frame.duration_sec)
    if (positive & ~valid_times).any():
        raise ValueError("Positive timestamps must be finite and within clip duration")
    if (~positive & frame.timestamp_sec.notna()).any():
        raise ValueError("Negative videos must not carry event timestamps")

    groups = []
    for group_id, aliases in frame.groupby("sha256", sort=True):
        aliases = aliases.sort_values(["video_id", "path"], kind="stable")
        if aliases.target.nunique() != 1:
            raise ValueError(f"Duplicate content has conflicting target labels: {group_id}")
        if not np.allclose(aliases.duration_sec, aliases.duration_sec.iloc[0], atol=1e-6, rtol=0):
            raise ValueError(f"Duplicate content has conflicting durations: {group_id}")
        target = int(aliases.target.iloc[0])
        times = aliases.timestamp_sec.dropna().unique()
        eligible = bool(target == 1 and len(times) == 1)
        if "localization_eligible" in aliases:
            eligible = eligible and all(aliases.localization_eligible.map(_boolean))
        groups.append({
            "group_id": str(group_id),
            "sha256": str(group_id),
            "video_ids": aliases.video_id.astype(str).tolist(),
            "paths": aliases.path.astype(str).tolist(),
            "path": str(aliases.path.iloc[0]),
            "target": target,
            "duration_sec": float(aliases.duration_sec.iloc[0]),
            "timestamp_sec": float(times[0]) if eligible else None,
            "localization_eligible": eligible,
            "timestamp_annotations": [
                {"video_id": str(row.video_id),
                 "timestamp_sec": float(row.timestamp_sec) if pd.notna(row.timestamp_sec) else None}
                for row in aliases.itertuples()
            ],
        })
    return groups


def create_or_load_protocol(inventory_path, output_path):
    """Create a deterministic 8-group holdout and three development folds once.

    Labels are used only to stratify partitions. Exact duplicates become one
    prediction unit. Conflicting timestamps remain recorded but unscored.
    A changed inventory, seed or partition never overwrites an existing freeze.
    """
    inventory_path, output_path = Path(inventory_path), Path(output_path)
    groups = _inventory_groups(inventory_path)
    targets = np.asarray([group["target"] for group in groups])
    if set(targets) != {0, 1} or min(Counter(targets).values()) < 7:
        raise ValueError("Need at least seven unique videos of each class for holdout and three folds")
    development, holdout = train_test_split(
        np.arange(len(groups)), test_size=HOLDOUT_SIZE,
        stratify=targets, random_state=config.SEED,
    )
    if Counter(targets[holdout]) != {0: 4, 1: 4}:
        raise ValueError("The approved holdout requires four positives and four negatives")
    development = np.sort(development)
    holdout = np.sort(holdout)
    splitter = StratifiedKFold(n_splits=N_VIDEO_FOLDS, shuffle=True, random_state=config.SEED)
    folds = []
    for training, validation in splitter.split(development, targets[development]):
        folds.append({
            "train_group_ids": [groups[index]["group_id"] for index in development[training]],
            "valid_group_ids": [groups[index]["group_id"] for index in development[validation]],
        })
    protocol = {
        "schema_version": 1,
        "seed": int(config.SEED),
        "raw_inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "groups": groups,
        "development_group_ids": [groups[index]["group_id"] for index in development],
        "holdout_group_ids": [groups[index]["group_id"] for index in holdout],
        "folds": folds,
        "holdout_policy": "Reserved; do not score before model and threshold policy freeze.",
        "metric_note": "Internal diagnostics only; official aggregate and time tolerance unspecified.",
    }
    protocol["fingerprint"] = _fingerprint(protocol)
    validate_protocol(protocol)
    if output_path.exists():
        frozen = load_protocol(output_path)
        if frozen["fingerprint"] != protocol["fingerprint"]:
            raise ValueError("Frozen video protocol differs from inventory, seed or partitions; do not overwrite")
        return frozen
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return protocol


def load_protocol(path):
    """Load and check the metadata fingerprint and every fold boundary."""
    with Path(path).open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    """Reject corrupt fingerprints, duplicate groups and leaked fold membership."""
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported video protocol schema")
    if not isinstance(protocol.get("seed"), int):
        raise ValueError("Video protocol seed must be an integer")
    if not _SHA256.fullmatch(str(protocol.get("raw_inventory_sha256", ""))):
        raise ValueError("Video protocol requires an inventory SHA-256 fingerprint")
    groups = protocol.get("groups", [])
    group_ids = [group["group_id"] for group in groups]
    if not group_ids or len(set(group_ids)) != len(group_ids):
        raise ValueError("Video protocol groups must be nonempty and unique")
    aliases, paths = [], []
    for group in groups:
        if not _SHA256.fullmatch(group["group_id"]) or group["sha256"] != group["group_id"]:
            raise ValueError("Protocol group IDs must be full content hashes")
        if group["target"] not in (0, 1):
            raise ValueError("Protocol target must be binary")
        if not group["video_ids"] or len(group["paths"]) != len(group["video_ids"]):
            raise ValueError("Protocol must preserve every duplicate alias and path")
        if group["path"] not in group["paths"]:
            raise ValueError("Canonical path must belong to the content group")
        aliases.extend(group["video_ids"])
        paths.extend(group["paths"])
        if not np.isfinite(group["duration_sec"]) or group["duration_sec"] <= 0:
            raise ValueError("Protocol durations must be finite and positive")
        if not isinstance(group["localization_eligible"], bool):
            raise ValueError("Protocol localization eligibility must be boolean")
        annotations = group["timestamp_annotations"]
        if [annotation["video_id"] for annotation in annotations] != group["video_ids"]:
            raise ValueError("Protocol must preserve every alias timestamp annotation")
        source_times = [annotation["timestamp_sec"] for annotation in annotations]
        if group["target"] == 1:
            if any(value is None or not np.isfinite(value) or not 0 <= value <= group["duration_sec"]
                   for value in source_times):
                raise ValueError("Positive source annotations must be within clip duration")
        elif any(value is not None for value in source_times):
            raise ValueError("Negative source annotations must not contain event timestamps")
        timestamp = group["timestamp_sec"]
        if group["localization_eligible"]:
            if (group["target"] != 1 or timestamp is None or not np.isfinite(timestamp)
                    or not 0 <= timestamp <= group["duration_sec"]):
                raise ValueError("Eligible timestamp must be a positive event within clip duration")
            if set(source_times) != {timestamp}:
                raise ValueError("Conflicting source annotations cannot be localization eligible")
        elif timestamp is not None:
            raise ValueError("Ineligible localization must not select an annotation")
    if len(aliases) != len(set(aliases)):
        raise ValueError("Video aliases cannot cross content groups")
    if len(paths) != len(set(paths)):
        raise ValueError("Video paths cannot cross content groups")
    development = protocol["development_group_ids"]
    holdout = protocol["holdout_group_ids"]
    if len(set(development)) != len(development) or len(set(holdout)) != len(holdout):
        raise ValueError("Video partitions cannot contain repeated groups")
    if set(development) & set(holdout) or set(development) | set(holdout) != set(group_ids):
        raise ValueError("Development and holdout must partition all groups without overlap")
    by_id = {group["group_id"]: group for group in groups}
    if len(holdout) != HOLDOUT_SIZE or Counter(by_id[key]["target"] for key in holdout) != {0: 4, 1: 4}:
        raise ValueError("Holdout must contain eight unique groups, four of each target")
    if len(protocol["folds"]) != N_VIDEO_FOLDS:
        raise ValueError("Video protocol requires three development folds")
    validated = []
    for fold in protocol["folds"]:
        training, validation = fold["train_group_ids"], fold["valid_group_ids"]
        if not training or not validation:
            raise ValueError("Training and validation folds must be nonempty")
        if len(set(training)) != len(training) or len(set(validation)) != len(validation):
            raise ValueError("A fold cannot repeat content groups")
        if set(training) & set(validation):
            raise ValueError("Training and validation content groups overlap")
        if (set(training) | set(validation)) != set(development):
            raise ValueError("Every fold must partition development only; holdout cannot enter a fold")
        validated.extend(validation)
    if Counter(validated) != Counter(development):
        raise ValueError("Each development group must be validated exactly once")
    if protocol.get("fingerprint") != _fingerprint(protocol):
        raise ValueError("Video protocol metadata fingerprint does not match its contents")


def protocol_groups(protocol, partition="development"):
    """Return stable group rows, without opening any video or extracting features."""
    if partition == "all":
        return list(protocol["groups"])
    if partition not in {"development", "holdout"}:
        raise ValueError("Partition must be development, holdout or all")
    by_id = {group["group_id"]: group for group in protocol["groups"]}
    return [by_id[key] for key in protocol[f"{partition}_group_ids"]]


def video_metrics(y_true, probabilities, threshold=0.5):
    """Video-level diagnostic metrics, not an invented official jury score."""
    y_true, probabilities = np.asarray(y_true), np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or probabilities.shape != y_true.shape or not len(y_true):
        raise ValueError("Targets and probabilities must be nonempty matching vectors")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Targets must be binary 0/1")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values between zero and one")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between zero and one")
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    recalls = [(tp / (tp + fn))] if tp + fn else []
    if tn + fp:
        recalls.append(tn / (tn + fp))
    return {
        "n": len(y_true), "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(np.mean(recalls)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
    }


def localization_metrics(y_true_sec, predicted_sec, eligible=None, duration_sec=None):
    """Timestamp diagnostics only on eligible positive ground truth.

    Pass a mask excluding negative and disputed labels. Predictions for these
    excluded rows are ignored; eligible missing predictions fail explicitly.
    These tolerances are diagnostics and are not the unspecified official rule.
    """
    actual, predicted = np.asarray(y_true_sec, dtype=float), np.asarray(predicted_sec, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Timestamp targets and predictions must be matching vectors")
    mask = np.ones(len(actual), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if mask.shape != actual.shape:
        raise ValueError("Localization eligibility must match timestamp vectors")
    truth, estimate = actual[mask], predicted[mask]
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("Eligible localization timestamps must be finite")
    if (truth < 0).any() or (estimate < 0).any():
        raise ValueError("Eligible localization timestamps must be nonnegative")
    if duration_sec is not None:
        durations = np.asarray(duration_sec, dtype=float)
        if durations.shape != actual.shape or not np.isfinite(durations[mask]).all():
            raise ValueError("Durations must match timestamps and be finite on eligible rows")
        if (durations[mask] <= 0).any() or (truth > durations[mask]).any() or (estimate > durations[mask]).any():
            raise ValueError("Localization timestamps must be within clip durations")
    errors = np.abs(estimate - truth)
    return {
        "n": len(errors),
        "mean_absolute_error_sec": float(errors.mean()) if len(errors) else None,
        "median_absolute_error_sec": float(np.median(errors)) if len(errors) else None,
        "within_1_sec": float(np.mean(errors <= 1)) if len(errors) else None,
        "within_3_sec": float(np.mean(errors <= 3)) if len(errors) else None,
    }
