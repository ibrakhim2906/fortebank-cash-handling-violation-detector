"""Synthetic private-cache integrity tests; never load bank media or real caches."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import numpy as np
import pytest

from scripts import audit_spatial_observations as spatial


GROUPS = ("synthetic-development-a", "synthetic-development-b")


def features(observation):
    vector = np.zeros(28, dtype=np.float32)
    vector[0] = observation["synthetic_value"]
    return vector, "clear"


def synthetic_record(offset=0):
    observations, windows = [], []
    for index in range(4):
        timestamp = float(index + 1)
        observation = {"synthetic_value": float(offset + index + 1)}
        vector, _ = features(observation)
        views = [{"crop_rect_xyxy": rectangle, "sha256": "b" * 64}
                 for rectangle in ([0, 0, 8, 12], [0, 0, 8, 8], [0, 4, 8, 12])]
        observations.append({"observation": observation, "actual_sec": timestamp,
                             "requested_sec": timestamp, "source_frame_index": 10 * (index + 1),
                             "source_shape": [12, 8, 3], "source_frame_sha256": "d" * 64,
                             "views": views})
        windows.append({"features": vector, "timestamp_sec": timestamp})
    matrix = np.stack([item["features"] for item in windows])
    return {"sampling": {"source_fps": 10.0}, "duration_sec": 10.0,
            "observation_audit": observations, "windows": windows,
            "clip_features": np.concatenate((matrix.max(axis=0), matrix.mean(axis=0), matrix.std(axis=0))),
            "extraction_sec": 1.25 + offset}


@pytest.fixture
def run(tmp_path, monkeypatch):
    directory = tmp_path / "synthetic-spatial-run"
    cache = directory / "cache"
    cache.mkdir(parents=True)
    records = {identifier: synthetic_record(index) for index, identifier in enumerate(GROUPS)}
    for identifier, record in records.items():
        joblib.dump(record, cache / f"{identifier}.joblib")
    engine = SimpleNamespace(FEATURE_CONFIG={"synthetic": True},
                             CLIP_STATISTICS=("max", "mean", "std"),
                             observation_features=features)
    protocol = {"development_group_ids": list(GROUPS), "holdout_group_ids": ["synthetic-holdout"]}
    provenance = {"engine": "synthetic", "source_sha256": "c" * 64}
    verify = Mock()
    monkeypatch.setattr(spatial, "load_protocol", lambda *a: protocol)
    monkeypatch.setattr(spatial, "load_engine", lambda *a: engine)
    monkeypatch.setattr(spatial, "engine_provenance", lambda *a: provenance)
    monkeypatch.setattr(spatial, "verify_feature_record", verify)
    return {"directory": directory, "cache": cache, "records": records,
            "protocol": protocol, "engine": engine, "provenance": provenance, "verify": verify}


def save(run, record, identifier=GROUPS[0]):
    joblib.dump(record, run["cache"] / f"{identifier}.joblib")


def test_complete_audit_returns_and_saves_only_aggregate_integrity(run):
    result = spatial.audit(run["directory"], "synthetic-protocol")
    assert result == {
        "cached_development_groups": 2, "holdout_groups_used": 0,
        "sampled_instants": 8, "sampled_views": 24, "unique_clip_vectors": 2,
        "nonconstant_features": 2, "maximum_extraction_sec": 2.25,
        "sampling_alignment_verified": True, "semantic_accuracy_verified": False,
        "operational_integrity_only": True,
    }
    assert run["verify"].call_count == 2
    for call in run["verify"].call_args_list:
        assert call.args[1] == run["provenance"]
        assert call.args[2] == run["engine"].FEATURE_CONFIG
    output = run["directory"] / "spatial_observation_audit.json"
    assert json.loads(output.read_text()) == result
    assert not any(identifier in output.read_text() for identifier in GROUPS)
    assert "synthetic_value" not in output.read_text()


def test_cli_emits_only_aggregate_counts_without_per_clip_observations(run, capsys):
    spatial.main(["--run-dir", str(run["directory"])])
    output = capsys.readouterr().out
    value = json.loads(output)
    assert value["sampled_instants"] == 8
    assert value["semantic_accuracy_verified"] is False
    assert not any(identifier in output for identifier in GROUPS)
    assert all(word not in output for word in ["synthetic_value", "source_shape", "crop_rect_xyxy",
                                               "requested_sec", "source_frame_index"])


@pytest.mark.parametrize("kind", ["missing", "extra", "holdout", "symlink"])
def test_membership_and_links_rejected_before_unpickling(run, monkeypatch, kind):
    if kind == "missing":
        (run["cache"] / f"{GROUPS[0]}.joblib").unlink()
    elif kind == "symlink":
        path = run["cache"] / f"{GROUPS[0]}.joblib"
        path.unlink()
        target = run["directory"] / "synthetic-target.joblib"
        joblib.dump(synthetic_record(), target)
        path.symlink_to(target)
    else:
        identifier = "synthetic-holdout" if kind == "holdout" else "unexpected-record"
        joblib.dump(synthetic_record(), run["cache"] / f"{identifier}.joblib")
    load = Mock(side_effect=AssertionError("No unpickle may precede membership checks"))
    monkeypatch.setattr(spatial.joblib, "load", load)
    with pytest.raises(ValueError, match="every permitted development record"):
        spatial.audit(run["directory"])
    load.assert_not_called()


def test_changed_window_feature_rejected_against_private_observation(run):
    record = copy.deepcopy(run["records"][GROUPS[0]])
    record["windows"][0]["features"][0] += 1
    save(run, record)
    with pytest.raises(ValueError, match="observation and classifier feature values differ"):
        spatial.audit(run["directory"])


@pytest.mark.parametrize("kind", ["nonfinite_actual", "nonfinite_fps", "nonpositive_fps",
                                 "outside_duration", "requested_mismatch", "window_mismatch"])
def test_invalid_timing_rejected(run, kind):
    record = copy.deepcopy(run["records"][GROUPS[0]])
    first = record["observation_audit"][0]
    if kind == "nonfinite_actual":
        first["actual_sec"] = float("nan")
    elif kind == "nonfinite_fps":
        record["sampling"]["source_fps"] = float("inf")
    elif kind == "nonpositive_fps":
        record["sampling"]["source_fps"] = 0
    elif kind == "outside_duration":
        first["actual_sec"] = record["duration_sec"]
    elif kind == "requested_mismatch":
        first["requested_sec"] += 1
    else:
        record["windows"][0]["timestamp_sec"] += 0.01
    save(run, record)
    with pytest.raises(ValueError, match="timing differs"):
        spatial.audit(run["directory"])


@pytest.mark.parametrize("kind", ["duplicate_time", "duplicate_index", "reversed_index"])
def test_samples_must_be_chronological(run, kind):
    record = copy.deepcopy(run["records"][GROUPS[0]])
    first, second = record["observation_audit"][:2]
    if kind == "duplicate_time":
        second["actual_sec"] = second["requested_sec"] = first["actual_sec"]
        record["windows"][1]["timestamp_sec"] = first["actual_sec"]
    else:
        second["source_frame_index"] = first["source_frame_index"] - int(kind == "reversed_index")
    save(run, record)
    with pytest.raises(ValueError, match="strictly chronological"):
        spatial.audit(run["directory"])


@pytest.mark.parametrize("kind", ["wrong_crop", "missing_view", "missing_hash", "nonhex_hash",
                                 "bad_source_hash"])
def test_source_aligned_crop_and_hash_metadata_required(run, kind):
    record = copy.deepcopy(run["records"][GROUPS[0]])
    views = record["observation_audit"][0]["views"]
    if kind == "wrong_crop":
        views[1]["crop_rect_xyxy"][0] = 1
    elif kind == "missing_view":
        views.pop()
    elif kind == "missing_hash":
        views[0]["sha256"] = "incomplete"
    elif kind == "nonhex_hash":
        views[0]["sha256"] = "z" * 64
    else:
        record["observation_audit"][0]["source_frame_sha256"] = "z" * 64
    save(run, record)
    with pytest.raises(ValueError, match="crop metadata|source-aligned view|hash provenance"):
        spatial.audit(run["directory"])


@pytest.mark.parametrize("kind", ["changed_pool", "nonfinite_clip", "wrong_clip_dimensions"])
def test_clip_must_match_finite_fixed_dimension_pooling(run, kind):
    record = copy.deepcopy(run["records"][GROUPS[0]])
    if kind == "changed_pool":
        record["clip_features"][0] += 0.5
    elif kind == "nonfinite_clip":
        record["clip_features"][0] = float("nan")
    else:
        record["clip_features"] = record["clip_features"][:-1]
    save(run, record)
    with pytest.raises(ValueError, match="clip pooling differs"):
        spatial.audit(run["directory"])


@pytest.mark.parametrize("field", ["observation_audit", "windows"])
def test_exactly_four_complete_samples_required(run, field):
    record = copy.deepcopy(run["records"][GROUPS[0]])
    record[field].pop()
    save(run, record)
    with pytest.raises(ValueError, match="incomplete sampling"):
        spatial.audit(run["directory"])


def test_cli_failure_never_prints_private_exception_or_observation_details(run, monkeypatch, capsys):
    monkeypatch.setattr(spatial, "audit", Mock(side_effect=ValueError("synthetic-private-observation")))
    with pytest.raises(SystemExit, match="no media or observations emitted") as error:
        spatial.main(["--run-dir", str(run["directory"])])
    assert "synthetic-private-observation" not in str(error.value)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("field,value", [("extraction_sec", float("nan")),
                                        ("extraction_sec", float("inf")),
                                        ("extraction_sec", -0.1),
                                        ("duration_sec", float("nan")),
                                        ("duration_sec", float("inf")),
                                        ("duration_sec", 0), ("duration_sec", 301)])
def test_runtime_and_duration_must_be_finite_and_within_bounds(run, field, value):
    record = copy.deepcopy(run["records"][GROUPS[0]])
    record[field] = value
    save(run, record)
    with pytest.raises(ValueError, match="duration or runtime is invalid"):
        spatial.audit(run["directory"])


@pytest.mark.parametrize("shape", [[0, 8, 3], [-12, 8, 3], [12, 8.5, 3],
                                  [True, 8, 3], [12, 8], [12, 8, 4]])
def test_source_shape_requires_positive_integer_rgb_dimensions(run, shape):
    record = copy.deepcopy(run["records"][GROUPS[0]])
    record["observation_audit"][0]["source_shape"] = shape
    save(run, record)
    with pytest.raises(ValueError, match="source dimensions are invalid"):
        spatial.audit(run["directory"])


@pytest.mark.parametrize("index", [True, -1, 0.5, float("nan"), float("inf"), 100])
def test_source_frame_index_requires_bounded_nonnegative_integer(run, index):
    record = copy.deepcopy(run["records"][GROUPS[0]])
    record["observation_audit"][0]["source_frame_index"] = index
    save(run, record)
    with pytest.raises(ValueError, match="source frame index is invalid"):
        spatial.audit(run["directory"])
