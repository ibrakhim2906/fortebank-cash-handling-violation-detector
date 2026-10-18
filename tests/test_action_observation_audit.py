"""Operational cache audits use only synthetic numeric records and metadata."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts import audit_action_observations as A
from src import video_qwen_actions as V
from src.video_actions import prepare_queue
from src.video_eval import create_or_load_protocol
from src.video_train import engine_provenance


@pytest.fixture
def run(tmp_path):
    rows = [{"video_id": f"synthetic-{index}", "path": f"local/absent-{index}.mov",
             "sha256": hashlib.sha256(f"audit-synthetic-{index}".encode()).hexdigest(),
             "target": index % 2, "timestamp_sec": 5.0 if index % 2 else np.nan,
             "duration_sec": 10.0} for index in range(40)]
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    protocol_path = tmp_path / "protocol.json"
    protocol = create_or_load_protocol(inventory, protocol_path)
    actions = tmp_path / "actions"
    prepare_queue(protocol_path, actions)
    run_dir = tmp_path / "run"
    cache = run_dir / "cache"
    cache.mkdir(parents=True)
    features, _ = V.observation_features({"actions": [], "visibility": "clear"})
    matrix = np.stack([features] * V.WINDOW_COUNT)
    clip = np.concatenate([matrix.max(axis=0), matrix.mean(axis=0), matrix.std(axis=0)])
    record = {**engine_provenance("qwen-actions", V), "feature_config": V.FEATURE_CONFIG,
              "clip_features": clip,
              "windows": [{"timestamp_sec": index + 1.0, "features": features.copy()}
                          for index in range(V.WINDOW_COUNT)], "duration_sec": 10.0}
    for group_id in protocol["development_group_ids"]:
        joblib.dump(record, cache / f"{group_id}.joblib")
    return run_dir, protocol_path, actions, protocol, record


def call_audit(run, **kwargs):
    run_dir, protocol_path, actions, _, _ = run
    return A.audit(run_dir, protocol_path=protocol_path, action_dir=actions, **kwargs)


def test_constant_observations_report_integrity_without_claiming_accuracy(run, monkeypatch):
    original_open = Path.open

    def no_media(self, *args, **kwargs):
        assert self.suffix.lower() != ".mov", "Audit must never open footage"
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_media)
    result = call_audit(run)
    assert result["cached_groups"] == result["expected_development_groups"] == 32
    assert result["unique_clip_vectors"] == 1 and result["nonconstant_features"] == 0
    assert result["operational_integrity_only"] is True and result["partial"] is False
    assert result["holdout_groups_used"] == result["verified_action_observations"] == 0
    assert json.loads((run[0] / "action_observation_audit.json").read_text()) == result
    assert all(isinstance(value, (int, bool)) for value in result.values())


def test_missing_cache_requires_explicit_partial_mode(run):
    next((run[0] / "cache").glob("*.joblib")).unlink()
    with pytest.raises(ValueError, match="every development"):
        call_audit(run)
    partial = call_audit(run, allow_partial=True)
    assert partial["partial"] is True and partial["cached_groups"] == 31
    assert partial["missing_development_groups"] == 1


def test_holdout_file_is_rejected_before_loading_even_in_partial_mode(run, monkeypatch):
    path = run[0] / "cache" / f"{run[3]['holdout_group_ids'][0]}.joblib"
    path.write_bytes(b"Must never be unpickled")
    monkeypatch.setattr(A.joblib, "load", lambda path: pytest.fail("Reserved cache must be rejected before any load"))
    with pytest.raises(ValueError, match="reserved"):
        call_audit(run, allow_partial=True)


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "a" * 64), ("weights_sha256", "b" * 64),
    ("inherited_sources", {}), ("feature_config", {}),
    ("protocol_fingerprint", "a" * 64),
    ("clip_features", np.zeros(1)), ("clip_features", np.full(57, np.nan)),
    ("duration_sec", float("inf")), ("windows", []),
])
def test_hash_schema_or_finite_value_mismatch_is_rejected(run, field, value):
    record = deepcopy(run[4])
    record[field] = value
    path = next((run[0] / "cache").glob("*.joblib"))
    joblib.dump(record, path)
    with pytest.raises(ValueError):
        call_audit(run)


@pytest.mark.parametrize("kind", ["all_unobservable", "zero_filled", "wrong_summary"])
def test_unobservable_failures_cannot_be_disguised_as_numeric_records(run, kind):
    record = deepcopy(run[4])
    if kind == "all_unobservable":
        features, _ = V.observation_features({"actions": [], "visibility": "unobservable"})
        for window in record["windows"]:
            window["features"] = features.copy()
    elif kind == "zero_filled":
        for window in record["windows"]:
            window["features"] = np.zeros(19)
    else:
        record["clip_features"][0] = 1
    joblib.dump(record, next((run[0] / "cache").glob("*.joblib")))
    with pytest.raises(ValueError):
        call_audit(run)


def test_run_manifest_protocol_mismatch_is_rejected(run):
    (run[0] / "model_manifest.json").write_text(json.dumps({
        "protocol_fingerprint": "wrong", "training_groups": run[3]["development_group_ids"]}))
    with pytest.raises(ValueError, match="manifest differs"):
        call_audit(run)


def test_completed_manifest_encoder_provenance_is_checked(run):
    manifest = {**engine_provenance("qwen-actions", V), "feature_config": V.FEATURE_CONFIG,
                "protocol_fingerprint": run[3]["fingerprint"],
                "training_groups": run[3]["development_group_ids"]}
    path = run[0] / "model_manifest.json"
    path.write_text(json.dumps(manifest))
    assert call_audit(run)["cached_groups"] == 32
    manifest["weights_sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="stale"):
        call_audit(run)


def test_distinct_valid_vectors_are_counted_without_exposing_their_actions(run):
    record = deepcopy(run[4])
    features, _ = V.observation_features({"actions": [{"action_id": V.ACTION_IDS[0], "presence": "present"}], "visibility": "clear"})
    for window in record["windows"]:
        window["features"] = features.copy()
    matrix = np.stack([features] * V.WINDOW_COUNT)
    record["clip_features"] = np.concatenate([matrix.max(axis=0), matrix.mean(axis=0), matrix.std(axis=0)])
    joblib.dump(record, next((run[0] / "cache").glob("*.joblib")))
    result = call_audit(run)
    assert result["unique_clip_vectors"] == 2 and result["nonconstant_features"] == 4
    assert not any(identifier in json.dumps(result) for identifier in V.ACTION_IDS)
