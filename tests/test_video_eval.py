"""Synthetic metadata checks: these tests never read bank video footage."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.video_eval import (
    create_or_load_protocol, load_protocol, localization_metrics, protocol_groups,
    validate_protocol, video_metrics,
)


@pytest.fixture
def inventory(tmp_path):
    rows = []
    for index in range(40):
        target = int(index < 21)
        rows.append({
            "video_id": f"Х-{index + 1}",
            "path": f"local/videos/Х-{index + 1}.mov",
            "sha256": hashlib.sha256(f"synthetic-video-{index}".encode()).hexdigest(),
            "target": target,
            "timestamp_sec": 10.0 if target else np.nan,
            "duration_sec": 100.0,
            "localization_eligible": bool(target),
        })
    # Keep both raw timestamps; neither can become localization ground truth.
    rows[3]["timestamp_sec"] = 57.0
    rows.append({**rows[3], "video_id": "Х-47", "path": "local/videos/Х-47.mov",
                 "timestamp_sec": 68.0})
    rows.append({**rows[4], "video_id": "Х-48", "path": "local/videos/Х-48.mov"})
    path = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_protocol_is_frozen_deduplicated_and_holdout_is_untouched(inventory, tmp_path):
    path = tmp_path / "protocol.json"
    protocol = create_or_load_protocol(inventory, path)
    before = path.read_bytes()
    assert create_or_load_protocol(inventory, path) == protocol
    assert path.read_bytes() == before
    assert load_protocol(path) == protocol
    assert len(protocol["groups"]) == 40
    assert len(protocol_groups(protocol)) == 32
    assert len(protocol_groups(protocol, "holdout")) == 8
    assert sum(group["target"] for group in protocol_groups(protocol, "holdout")) == 4
    assert sum(len(group["video_ids"]) for group in protocol["groups"]) == 42
    holdout = set(protocol["holdout_group_ids"])
    validation_ids = []
    for fold in protocol["folds"]:
        assert not holdout & set(fold["train_group_ids"] + fold["valid_group_ids"])
        assert not set(fold["train_group_ids"]) & set(fold["valid_group_ids"])
        validation_ids.extend(fold["valid_group_ids"])
    assert sorted(validation_ids) == sorted(protocol["development_group_ids"])
    conflict = next(group for group in protocol["groups"] if "Х-47" in group["video_ids"])
    assert set(conflict["video_ids"]) == {"Х-4", "Х-47"}
    assert conflict["timestamp_sec"] is None
    assert not conflict["localization_eligible"]
    assert {annotation["timestamp_sec"] for annotation in conflict["timestamp_annotations"]} == {57.0, 68.0}
    clear_duplicate = next(group for group in protocol["groups"] if "Х-48" in group["video_ids"])
    assert clear_duplicate["localization_eligible"]
    assert clear_duplicate["timestamp_sec"] == 10


def test_canonical_names_and_membership_stable_under_csv_reordering(inventory, tmp_path):
    original = create_or_load_protocol(inventory, tmp_path / "first.json")
    pd.read_csv(inventory).sample(frac=1, random_state=config.SEED).to_csv(inventory, index=False)
    reordered = create_or_load_protocol(inventory, tmp_path / "second.json")
    assert original["groups"] == reordered["groups"]
    assert original["development_group_ids"] == reordered["development_group_ids"]
    assert original["holdout_group_ids"] == reordered["holdout_group_ids"]
    assert original["folds"] == reordered["folds"]
    assert original["raw_inventory_sha256"] != reordered["raw_inventory_sha256"]


@pytest.mark.parametrize("mutation", ["seed", "inventory"])
def test_changed_inputs_never_overwrite_frozen_protocol(inventory, tmp_path, monkeypatch, mutation):
    output = tmp_path / "protocol.json"
    create_or_load_protocol(inventory, output)
    original = output.read_bytes()
    if mutation == "seed":
        monkeypatch.setattr(config, "SEED", config.SEED + 1)
    else:
        frame = pd.read_csv(inventory)
        frame.loc[0, "path"] = "local/videos/a-new-name.mov"
        frame.to_csv(inventory, index=False)
    with pytest.raises(ValueError, match="Frozen video protocol differs"):
        create_or_load_protocol(inventory, output)
    assert output.read_bytes() == original


def test_conflicting_duplicate_targets_are_blocked(inventory, tmp_path):
    frame = pd.read_csv(inventory)
    frame.loc[frame.video_id.eq("Х-48"), ["target", "timestamp_sec"]] = [0, np.nan]
    frame.to_csv(inventory, index=False)
    with pytest.raises(ValueError, match="conflicting target labels"):
        create_or_load_protocol(inventory, tmp_path / "protocol.json")


@pytest.mark.parametrize("timestamp", [-1, 101, float("inf"), float("nan")])
def test_out_of_bounds_or_missing_positive_timestamp_is_blocked(inventory, tmp_path, timestamp):
    frame = pd.read_csv(inventory)
    frame.loc[0, "timestamp_sec"] = timestamp
    frame.to_csv(inventory, index=False)
    with pytest.raises(ValueError, match="within clip duration"):
        create_or_load_protocol(inventory, tmp_path / "protocol.json")


@pytest.mark.parametrize("mutation, message", [
    ("overlap", "overlap"),
    ("holdout_in_fold", "holdout cannot enter"),
    ("repeat_validation", "validated exactly once"),
    ("duplicate_group", "nonempty and unique"),
    ("metadata", "fingerprint"),
])
def test_corrupt_protocols_are_rejected(inventory, tmp_path, mutation, message):
    protocol = deepcopy(create_or_load_protocol(inventory, tmp_path / "protocol.json"))
    if mutation == "overlap":
        protocol["folds"][0]["train_group_ids"].append(protocol["folds"][0]["valid_group_ids"][0])
    elif mutation == "holdout_in_fold":
        protocol["folds"][0]["train_group_ids"].append(protocol["holdout_group_ids"][0])
    elif mutation == "repeat_validation":
        protocol["folds"][1] = deepcopy(protocol["folds"][0])
    elif mutation == "duplicate_group":
        protocol["groups"].append(deepcopy(protocol["groups"][0]))
    else:
        protocol["groups"][0]["path"] = protocol["groups"][0]["paths"][-1]
        protocol["groups"][0]["duration_sec"] += 1
    with pytest.raises(ValueError, match=message):
        validate_protocol(protocol)


def test_disk_metadata_fingerprint_checked(inventory, tmp_path):
    path = tmp_path / "protocol.json"
    create_or_load_protocol(inventory, path)
    payload = json.loads(path.read_text())
    payload["seed"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_protocol(path)


def test_disputed_timestamp_cannot_be_selected_in_loaded_protocol(inventory, tmp_path):
    protocol = create_or_load_protocol(inventory, tmp_path / "protocol.json")
    conflict = next(group for group in protocol["groups"] if "Х-47" in group["video_ids"])
    conflict["localization_eligible"] = True
    conflict["timestamp_sec"] = 57.0
    with pytest.raises(ValueError, match="Conflicting source annotations"):
        validate_protocol(protocol)


def test_video_metrics_have_known_confusion_and_ranking_values():
    result = video_metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9])
    assert {key: result[key] for key in ("tn", "fp", "fn", "tp")} == dict(tn=1, fp=1, fn=1, tp=1)
    for metric in ("precision", "recall", "f1", "balanced_accuracy"):
        assert result[metric] == 0.5
    assert result["roc_auc"] == 0.75
    assert video_metrics([0, 0], [0.2, 0.3])["roc_auc"] is None


@pytest.mark.parametrize("probabilities", [[0.1], [float("nan"), 0.2], [0.1, 1.2]])
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError):
        video_metrics([0, 1], probabilities)


def test_localization_excludes_unreviewed_and_negative_annotations():
    result = localization_metrics(
        [10, 20, np.nan, np.nan], [11, 24, 99, np.nan], [True, True, False, False],
        duration_sec=[100, 100, 100, 100],
    )
    assert result == {
        "n": 2, "mean_absolute_error_sec": 2.5, "median_absolute_error_sec": 2.5,
        "within_1_sec": 0.5, "within_3_sec": 0.5,
    }
    empty = localization_metrics([np.nan], [np.nan], [False])
    assert empty["n"] == 0
    assert empty["median_absolute_error_sec"] is None
    with pytest.raises(ValueError, match="within clip durations"):
        localization_metrics([10], [101], duration_sec=[100])
    with pytest.raises(ValueError, match="must be finite"):
        localization_metrics([10], [np.nan])
