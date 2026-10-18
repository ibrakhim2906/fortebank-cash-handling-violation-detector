"""Synthetic private-cache integrity tests; no decoder, model or bank data used."""
import hashlib
import json

import cv2
import joblib
import numpy as np
import pandas as pd
import pytest

from scripts import audit_object_observations as A
from src import config
from src.video_eval import create_or_load_protocol


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    rows = [{"video_id": f"synthetic-{index}", "path": f"synthetic-{index}.mov",
             "sha256": hashlib.sha256(f"synthetic-cache-group-{index}".encode()).hexdigest(),
             "target": index % 2, "timestamp_sec": 1.0 if index % 2 else np.nan,
             "duration_sec": 2.0} for index in range(40)]
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    protocol_path = tmp_path / "protocol.json"
    protocol = create_or_load_protocol(inventory, protocol_path)
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    provenance = A.engine.get_encoder_provenance()
    for group_number, identifier in enumerate(protocol["development_group_ids"]):
        tracker = A.engine.AnonymousTracker()
        observations = []
        for index, timestamp in enumerate([0., .5, 1., 1.5]):
            x = .1 + group_number * .001 + index * .002
            detections = A.engine.validate_detections([
                {"class_id": 0, "score": .9, "box_xyxy": [x, .1, x + .3, .8]},
                {"class_id": 67, "score": .8, "box_xyxy": [.6, .5, .7, .6]},
            ])
            observations.append({"timestamp_sec": timestamp, "requested_sec": timestamp,
                                 "source_frame_index": int(timestamp * 10), "timestamp_source": "decoder",
                                 "source_shape": [64, 96, 3],
                                 "source_frame_sha256": hashlib.sha256(
                                     f"synthetic-pixels-{group_number}-{index}".encode()).hexdigest(),
                                 "detections": tracker.update(detections, timestamp)})
        windows, clip = A.engine.summarize_windows(observations)
        record = {**provenance, "feature_config": A.engine.FEATURE_CONFIG,
                  "clip_features": clip, "windows": windows, "object_audit": observations,
                  "duration_sec": 2., "extraction_sec": .2,
                  "sampling": {"sampled_frames": 4, "decoded_frames": 20, "source_fps": 10.,
                               "timestamp_fallbacks": 0, "elapsed_sec": .19,
                               "sample_times_sec": [0., .5, 1., 1.5],
                               "feature_config": A.engine.FEATURE_CONFIG}}
        joblib.dump(record, run / "cache" / f"{identifier}.joblib")
    (run / "models").mkdir()
    # Hash-only model verification must never unpickle this placeholder.
    model_bytes = b"Synthetic model placeholder for manifest-integrity testing"
    (run / "models/baseline.joblib").write_bytes(model_bytes)
    common = {**provenance, "feature_config": A.engine.FEATURE_CONFIG,
              "protocol_fingerprint": protocol["fingerprint"], "cache_dir": str(run / "cache")}
    development = {**common, "group_ids": protocol["development_group_ids"],
                   "holdout_evaluated": False, "threshold": .5, "seed": config.SEED}
    manifest = {**common, "training_groups": protocol["development_group_ids"],
                "stage": "development-only", "model_sha256": hashlib.sha256(model_bytes).hexdigest()}
    (run / "development.json").write_text(json.dumps(development))
    (run / "model_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(A.engine._assets, "verify_ready", lambda: {
        "weights_sha256": A.engine.WEIGHTS_SHA256,
        "model_revision": A.engine.FEATURE_CONFIG["model_revision"]})
    return run, protocol_path, protocol


def mutate_record(synthetic_run, change):
    run, _, protocol = synthetic_run
    path = run / "cache" / f"{protocol['development_group_ids'][0]}.joblib"
    record = joblib.load(path)
    change(record)
    joblib.dump(record, path)


def test_full_audit_recomputes_all_features_and_outputs_only_aggregate_counts(synthetic_run, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Cache audit must not decode videos or instantiate/call models")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", forbidden)
    monkeypatch.setattr(A.engine._objects, "ObjectDetector", forbidden)
    run, protocol_path, protocol = synthetic_run
    result = A.audit(run, protocol_path)
    assert result["complete_integrity_audit"] is True
    assert result["cached_development_groups"] == 32
    assert result["holdout_groups_used"] == 0
    assert result["sampled_instants"] == 128
    assert result["two_second_windows"] == 32
    assert result["accepted_object_predictions"] == 256
    assert result["anonymous_track_segments"] == 64
    assert result["clip_feature_count"] == 171 and result["window_feature_count"] == 57
    assert result["source_hashes_verified"] == 4
    assert result["semantic_accuracy_verified"] is False
    output = json.dumps(result)
    assert all(identifier not in output for identifier in protocol["development_group_ids"])
    assert "class_id" not in output and "box_xyxy" not in output
    assert str(run) not in output
    assert json.loads((run / "object_observation_audit.json").read_text()) == result


def test_progress_allows_partial_records_but_never_claims_complete_or_writes_final(synthetic_run):
    run, protocol_path, protocol = synthetic_run
    (run / "cache" / f"{protocol['development_group_ids'][0]}.joblib").unlink()
    (run / "model_manifest.json").unlink()
    result = A.audit(run, protocol_path, progress=True)
    assert result["cached_development_groups"] == 31
    assert result["missing_development_groups"] == 1
    assert result["completed_reports_present"] == 1
    assert result["complete_integrity_audit"] is False
    assert not (run / "object_observation_audit.json").exists()
    with pytest.raises(A.AuditError, match="all development caches"):
        A.audit(run, protocol_path)


def test_complete_progress_snapshot_still_does_not_claim_full_audit(synthetic_run):
    run, protocol_path, _ = synthetic_run
    assert A.audit(run, protocol_path, progress=True)["complete_integrity_audit"] is False


def test_full_audit_requires_both_completed_reports(synthetic_run):
    run, protocol_path, _ = synthetic_run
    (run / "development.json").unlink()
    with pytest.raises(A.AuditError, match="development and model manifests"):
        A.audit(run, protocol_path)


def test_extra_reserved_cache_rejected_before_any_pickle_is_loaded(synthetic_run, monkeypatch):
    run, protocol_path, protocol = synthetic_run
    (run / "cache" / f"{protocol['holdout_group_ids'][0]}.joblib").write_bytes(b"must never load")

    def forbidden(*args, **kwargs):
        pytest.fail("Reserved membership must fail before any pickle is loaded")

    monkeypatch.setattr(A.joblib, "load", forbidden)
    with pytest.raises(A.AuditError, match="reserved or unknown"):
        A.audit(run, protocol_path)


def test_symlink_cache_rejected_before_loading(synthetic_run):
    run, protocol_path, protocol = synthetic_run
    path = run / "cache" / f"{protocol['development_group_ids'][0]}.joblib"
    original = run / "outside.joblib"
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(A.AuditError, match="unsupported entries"):
        A.audit(run, protocol_path)


@pytest.mark.parametrize("change,expected", [
    (lambda record: record.update(source_sha256="wrong"), "provenance"),
    (lambda record: record["sampling"].update(feature_config={}), "sampling configuration"),
    (lambda record: record["sampling"].update(sampled_frames=3), "two-FPS"),
    (lambda record: record["sampling"].update(decoded_frames=19), "duration"),
    (lambda record: record["sampling"].update(elapsed_sec=9), "timing"),
    (lambda record: record["object_audit"][1].update(requested_sec=.6), "timing"),
    (lambda record: record["object_audit"][1].update(timestamp_sec=.8), "timing"),
    (lambda record: record["object_audit"][1].update(source_frame_index=0), "chronological"),
    (lambda record: record["object_audit"][0].update(source_shape=[64, 96, 1]), "dimensions"),
    (lambda record: record["object_audit"][0].update(source_frame_sha256="bad"), "frame-hash"),
    (lambda record: record["object_audit"][0]["detections"][0].update(track_id=99), "IoU"),
    (lambda record: record["object_audit"][1]["detections"][0].update(class_id=63), "IoU"),
    (lambda record: record["object_audit"][0]["detections"][0].update(score=.2), "threshold"),
    (lambda record: record["object_audit"][0]["detections"][0].update(box_xyxy=[0, 0, 2, 1]), "boxes"),
    (lambda record: record["windows"][0]["features"].__setitem__(0, 99), "window features"),
    (lambda record: record["clip_features"].__setitem__(0, np.nan), "clip pooling"),
])
def test_private_metadata_or_feature_corruption_is_detected(synthetic_run, change, expected):
    mutate_record(synthetic_run, change)
    with pytest.raises(A.AuditError, match=expected):
        A.audit(synthetic_run[0], synthetic_run[1])


def test_valid_sampled_timestamp_fallback_recomputes_and_counts(synthetic_run):
    def modify(record):
        record["object_audit"][1]["timestamp_source"] = "source_frame_index_over_fps"
        record["sampling"]["timestamp_fallbacks"] = 1

    mutate_record(synthetic_run, modify)
    assert A.audit(synthetic_run[0], synthetic_run[1])["sampled_timestamp_fallbacks"] == 1


def test_fallback_timestamp_requires_exact_frame_over_fps(synthetic_run):
    def modify(record):
        row = record["object_audit"][1]
        row.update(timestamp_source="source_frame_index_over_fps", timestamp_sec=.51)
        record["sampling"]["timestamp_fallbacks"] = 1

    mutate_record(synthetic_run, modify)
    with pytest.raises(A.AuditError, match="fallback"):
        A.audit(synthetic_run[0], synthetic_run[1])


def test_modified_model_bytes_fail_manifest_hash_check(synthetic_run):
    run, protocol_path, _ = synthetic_run
    (run / "models/baseline.joblib").write_bytes(b"changed")
    with pytest.raises(A.AuditError, match="model bytes"):
        A.audit(run, protocol_path)


def test_expired_tracks_cannot_reuse_old_ids_after_a_gap(synthetic_run):
    def modify(record):
        record["object_audit"][1]["detections"] = []
        record["object_audit"][2]["detections"] = []
        # The original IDs at 1.5 seconds are now stale after a 1.5-second gap.

    mutate_record(synthetic_run, modify)
    with pytest.raises(A.AuditError, match="IoU"):
        A.audit(synthetic_run[0], synthetic_run[1])


def test_changed_source_hash_blocks_audit_before_any_cache_load(synthetic_run, monkeypatch):
    monkeypatch.setattr(A.engine, "INHERITED_SOURCES", {"src/video_analysis.py": "wrong"})

    def forbidden(*args, **kwargs):
        pytest.fail("Changed source must be rejected before cache loading")

    monkeypatch.setattr(A.joblib, "load", forbidden)
    with pytest.raises(ValueError, match="helper source changed"):
        A.audit(synthetic_run[0], synthetic_run[1])


def test_cli_failure_sanitizes_private_exception_text(synthetic_run, monkeypatch, capsys):
    def private_failure(*args, **kwargs):
        raise ValueError("PRIVATE-FILENAME-AND-DETECTION-VALUES")

    monkeypatch.setattr(A, "audit", private_failure)
    with pytest.raises(SystemExit) as exc:
        A.main(["--run-dir", str(synthetic_run[0])])
    assert "PRIVATE-FILENAME" not in str(exc.value)
    assert "no media" in str(exc.value)
    assert capsys.readouterr().out == ""
