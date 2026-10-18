"""Synthetic metadata/logits only: no bank media, pretrained runtime or downloads."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import config
from src import video_transfer_pilot as pilot
from src.video_eval import create_or_load_protocol


@pytest.fixture
def case(tmp_path):
    rows = []
    for index in range(40):
        source = tmp_path / "videos" / f"synthetic-{index}.mov"
        source.parent.mkdir(exist_ok=True)
        payload = f"synthetic bytes, not a video, {index}".encode()
        source.write_bytes(payload)
        rows.append({"video_id": f"synthetic-{index}", "path": str(source.relative_to(tmp_path)),
                     "sha256": hashlib.sha256(payload).hexdigest(), "target": int(index < 21),
                     "timestamp_sec": 10 if index < 21 else np.nan, "duration_sec": 20.0})
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    protocol_path = tmp_path / "protocol.json"
    protocol = create_or_load_protocol(inventory, protocol_path)
    groups = {group["group_id"]: group for group in protocol["groups"]}
    by_pool = {"train": protocol["folds"][0]["train_group_ids"],
               "blind": protocol["folds"][0]["valid_group_ids"]}
    records = []
    for pool, ids in by_pool.items():
        for index, group_id in enumerate(sorted(ids)[:8]):
            target = int(index < 4)
            record = {"task_id": hashlib.sha256(f"task-{group_id}".encode()).hexdigest(),
                      "group_id": group_id, "source_sha256": group_id, "partition": "development",
                      "pool": pool, "interval": {"start_sec": 5.0 if target else 2.0,
                                                 "end_sec": 7.0 if target else 14.0},
                      "coverage": "interval", "action_id": pilot.ACTION_ID,
                      "target": target, "hard_negative": not target,
                      "counterexample_type": None if target else "same_person_relocation",
                      "visibility": "clear", "status": "verified", "reviewer_id": "local-human",
                      "evidence_sec": 6.0 if target else None,
                      "provenance": {"kind": "human_review", "watched": True,
                                     "evidence_confirmed": bool(target), "absence_confirmed": not target},
                      "review_revision": 1}
            record["review_sha256"] = pilot.fingerprint(record)
            records.append(record)
    return SimpleNamespace(root=tmp_path, protocol=protocol, protocol_path=protocol_path,
                           groups=groups, records=records)


def export(case, records=None, *, released=False, token=None):
    records = deepcopy(case.records if records is None else records)
    blind = sorted([row for row in records if row["pool"] == "blind"], key=lambda row: (row["group_id"], row["task_id"]))
    # Fixtures contain exactly the committed balanced blind eight.
    commitment = pilot._blind_commitment(blind) if len(blind) == 8 else None
    counts = {}
    for pool in ("train", "blind"):
        relevant = [row for row in records if row["pool"] == pool]
        counts[pool] = {"positive_recordings": len({row["group_id"] for row in relevant if row["target"]}),
                        "hard_negative_recordings_after_positive_selection": len({row["group_id"] for row in relevant if not row["target"]})}
    return {"schema_version": 1, "protocol_fingerprint": case.protocol["fingerprint"],
            "queue_fingerprint": "b" * 64, "annotation_fingerprint": pilot.fingerprint(records),
            "fold_index": 0, "quota_counts": counts, "blind_selection_fingerprint": commitment,
            "blind_release_fingerprint": token if released else None,
            "records": records if released else [row for row in records if row["pool"] == "train"]}


ENCODER = {"module": "synthetic-only", "source_sha256": "a" * 64,
           "feature_config": {"synthetic": True}, "asset_provenance": {"weights_sha256": "c" * 64}}
SOURCES = {"pilot_source_sha256": "d" * 64, "review_helper_sha256": "e" * 64,
           "video_eval_sha256": "f" * 64}


def manifest(case, tmp_path):
    selected = pilot.build_selection(case.protocol, export(case))
    return pilot.freeze_manifest(tmp_path / "manifest.json", selected, ENCODER, SOURCES)


def feature_records(examples, *, offset=0.0):
    records = {}
    for row in examples:
        requested = row["window"]["start_sec"] + np.arange(40) / 5
        features = np.zeros(600)
        features[0] = 1.0 if row["target"] else -1.0
        features[1] = offset
        records[row["group_id"]] = {
            "source_sha256": row["source_sha256"], "example_sha256": row["example_sha256"],
            "features": features.tolist(), "encoder_provenance": ENCODER["asset_provenance"],
            "sampling": {"requested_times_sec": requested.tolist(), "actual_times_sec": (requested + .01).tolist(),
                         "source_frame_indices": [int(row["window"]["start_sec"] * 30) + index * 6 for index in range(40)],
                         "source_frame_sha256": [hashlib.sha256(f"synthetic-frame-{index}".encode()).hexdigest() for index in range(40)],
                         "source_shapes": [[20, 30, 3] for _ in range(40)],
                         "timestamp_sources": ["decoder"] * 40,
                         "feature_config": ENCODER["feature_config"]}}
    return records


def released_blind(case, frozen):
    token = "1" * 64
    return pilot.release_blind_selection(frozen, case.protocol, export(case, released=True, token=token), token)


def test_zero_labels_block_without_examples_or_model(case):
    result = pilot.build_selection(case.protocol, export(case, []))
    assert result["status"] == "blocked"
    assert "examples" not in result
    assert result["counts"]["train"]["positive_recordings"] == 0
    assert result["holdout_opened"] == 0


def test_prepare_has_exactly_eight_training_labels_and_hidden_blind_commitment(case):
    result = pilot.build_selection(case.protocol, export(case))
    assert result["status"] == "ready"
    assert len(result["examples"]) == 8
    assert {row["pool"] for row in result["examples"]} == {"train"}
    assert sum(row["target"] for row in result["examples"]) == 4
    assert not set(row["group_id"] for row in result["examples"]) & set(case.protocol["holdout_group_ids"])


def test_blind_outcomes_cannot_enter_preparation(case):
    with pytest.raises(pilot.PilotError, match="remain hidden"):
        pilot.build_selection(case.protocol, export(case, released=True, token="1" * 64))


@pytest.mark.parametrize("change", ["model", "unclear", "unwatched", "unverified", "missing_evidence"])
def test_nonhuman_or_unverified_positive_cannot_fill_quota(case, change):
    rows = deepcopy(case.records)
    row = next(row for row in rows if row["pool"] == "train" and row["target"])
    if change == "model":
        row["provenance"]["kind"] = "model_proposal"
    elif change == "unclear":
        row["visibility"] = "partial"
    elif change == "unwatched":
        row["provenance"]["watched"] = False
    elif change == "unverified":
        row["status"] = "proposed"
    else:
        row["provenance"]["evidence_confirmed"] = False
    result = pilot.build_selection(case.protocol, export(case, rows))
    assert result["status"] == "blocked"
    assert result["counts"]["train"]["positive_recordings"] == 3


@pytest.mark.parametrize("change", ["background", "no_absence", "short_interval"])
def test_negatives_require_explicit_watched_hard_counterexamples(case, change):
    rows = deepcopy(case.records)
    row = next(row for row in rows if row["pool"] == "train" and not row["target"])
    if change == "background":
        row["counterexample_type"] = "background"
    elif change == "no_absence":
        row["provenance"]["absence_confirmed"] = False
    else:
        row["interval"] = {"start_sec": 2.0, "end_sec": 9.0}
    assert pilot.build_selection(case.protocol, export(case, rows))["status"] == "blocked"


def test_complete_positive_event_must_fit_eight_seconds(case):
    rows = deepcopy(case.records)
    row = next(row for row in rows if row["pool"] == "train" and row["target"])
    row["interval"] = {"start_sec": 1.0, "end_sec": 10.0}
    assert pilot.build_selection(case.protocol, export(case, rows))["status"] == "blocked"


@pytest.mark.parametrize("interval,expected", [((0.0, 1.0), (0.0, 8.0)), ((19.0, 20.0), (12.0, 20.0))])
def test_positive_context_clamps_without_truncating_event(case, interval, expected):
    row = deepcopy(next(row for row in case.records if row["target"]))
    row["interval"] = dict(zip(("start_sec", "end_sec"), interval))
    row["evidence_sec"] = sum(interval) / 2
    result, reason = pilot._example(row, case.groups[row["group_id"]], row["pool"])
    assert reason is None
    assert tuple(result["window"].values()) == expected


def test_weak_organizer_target_does_not_assign_action_semantics(case):
    row = deepcopy(next(row for row in case.records if not row["target"]))
    group = {**case.groups[row["group_id"]], "target": 1}
    example, reason = pilot._example(row, group, row["pool"])
    assert reason is None and example["target"] == 0


def test_reserved_source_is_rejected_even_if_claimed_verified(case):
    rows = deepcopy(case.records)
    rows[0]["group_id"] = case.protocol["holdout_group_ids"][0]
    rows[0]["source_sha256"] = rows[0]["group_id"]
    with pytest.raises(pilot.PilotError, match="Only frozen development"):
        pilot.build_selection(case.protocol, export(case, rows))


def test_multiple_intervals_from_one_recording_do_not_multiply_support(case):
    rows = deepcopy(case.records)
    positives = [row for row in rows if row["pool"] == "train" and row["target"]]
    for index, row in enumerate(positives):
        row["group_id"] = positives[0]["group_id"]
        row["source_sha256"] = row["group_id"]
        row["task_id"] = f"synthetic-duplicate-{index}"
    result = pilot.build_selection(case.protocol, export(case, rows))
    assert result["status"] == "blocked"
    assert result["counts"]["train"]["positive_recordings"] == 1


def test_another_fold_cannot_be_chosen_after_label_counts(case):
    labels = export(case)
    labels["fold_index"] = 1
    with pytest.raises(pilot.PilotError, match="frozen pilot protocol"):
        pilot.build_selection(case.protocol, labels)


def test_frozen_manifest_cannot_be_overwritten(case, tmp_path):
    frozen = manifest(case, tmp_path)
    before = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        pilot.freeze_manifest(tmp_path / "manifest.json", pilot.build_selection(case.protocol, export(case)), ENCODER, SOURCES)
    assert (tmp_path / "manifest.json").read_bytes() == before
    assert pilot.validate_manifest(frozen, case.protocol, export(case), ENCODER, SOURCES) == frozen


def test_selected_human_revision_cannot_change_after_freeze(case, tmp_path):
    frozen = manifest(case, tmp_path)
    rows = deepcopy(case.records)
    chosen = next(row for row in rows if row["task_id"] == frozen["examples"][0]["task_id"])
    chosen["interval"]["start_sec"] += .1
    with pytest.raises(pilot.PilotError, match="semantics changed"):
        pilot.validate_manifest(frozen, case.protocol, export(case, rows))


def test_blind_release_requires_matching_fitted_artifact_identity(case, tmp_path):
    frozen = manifest(case, tmp_path)
    with pytest.raises(pilot.PilotError, match="not released"):
        pilot.release_blind_selection(frozen, case.protocol, export(case, released=True, token="1" * 64), "2" * 64)
    blind = released_blind(case, frozen)
    assert len(blind) == 8 and sum(row["target"] for row in blind) == 4
    assert not {row["group_id"] for row in blind} & {row["group_id"] for row in frozen["examples"]}


def test_changed_hidden_selection_is_not_silently_replaced(case, tmp_path):
    frozen = manifest(case, tmp_path)
    labels = export(case, released=True, token="1" * 64)
    labels["blind_selection_fingerprint"] = "2" * 64
    with pytest.raises(pilot.PilotError, match="blind selection changed"):
        pilot.release_blind_selection(frozen, case.protocol, labels, "1" * 64)


@pytest.mark.parametrize("change", ["misaligned", "reordered", "reused_index", "bad_hash", "bad_shape", "wrong_config"])
def test_temporal_audit_rejects_misalignment_or_false_provenance(case, tmp_path, change):
    frozen = manifest(case, tmp_path)
    row = frozen["examples"][0]
    record = feature_records([row])[row["group_id"]]
    audit = record["sampling"]
    if change == "misaligned":
        audit["actual_times_sec"][2] += .3
    elif change == "reordered":
        audit["actual_times_sec"][0], audit["actual_times_sec"][1] = audit["actual_times_sec"][1], audit["actual_times_sec"][0]
    elif change == "reused_index":
        audit["source_frame_indices"][1] = audit["source_frame_indices"][0]
    elif change == "bad_hash":
        audit["source_frame_sha256"][0] = "unverified"
    elif change == "bad_shape":
        audit["source_shapes"][0] = [0, 30, 3]
    else:
        audit["feature_config"] = {"other_model": True}
    with pytest.raises(pilot.PilotError):
        pilot._validate_features(row, record, ENCODER)


@pytest.mark.parametrize("bad", [[0] * 599, [float("nan")] * 600])
def test_feature_shape_and_finiteness_are_checked(case, tmp_path, bad):
    frozen = manifest(case, tmp_path)
    row = frozen["examples"][0]
    record = feature_records([row])[row["group_id"]]
    record["features"] = bad
    with pytest.raises(pilot.PilotError, match="600 finite"):
        pilot._validate_features(row, record, ENCODER)


def test_scaler_and_head_fit_only_training_eight_then_blind_is_prediction_only(case, tmp_path):
    frozen = manifest(case, tmp_path)
    blind = released_blind(case, frozen)
    train_records = feature_records(frozen["examples"])
    blind_records = feature_records(blind, offset=10000)
    model = pilot.fit_training_head(frozen, train_records)
    assert model[0].mean_[1] == 0  # Would be 5000 if blind rows had entered fit.
    assert model[-1].C == 1 and model[-1].max_iter == 2000 and model[-1].random_state == config.SEED
    model.fit = lambda *args, **kwargs: pytest.fail("Blind evaluation must never call fit")
    report = pilot.evaluate_blind_head(frozen, blind, blind_records, model)
    assert report["action_metrics"]["tp"] == report["action_metrics"]["tn"] == 4
    assert report["classification_feasibility_pass"]
    assert not report["promotion_authorized"]
    assert report["evidence_semantic_review_required"]
    assert report["reviewed_interval_sampling_alignment_checked"]
    assert not report["evidence_sampling_alignment_checked"]
    assert "violation_metrics" not in report


def test_head_fit_interface_rejects_even_one_blind_record(case, tmp_path):
    frozen = manifest(case, tmp_path)
    changed = deepcopy(frozen)
    changed["examples"][0]["pool"] = "blind"
    with pytest.raises(pilot.PilotError, match="training sources"):
        pilot.fit_training_head(changed, feature_records(changed["examples"]))


def test_extraction_opens_only_selected_development_training_sources(case, tmp_path):
    frozen = manifest(case, tmp_path)
    expected = feature_records(frozen["examples"])
    called = []
    def extractor(path, start, end, **kwargs):
        group_id = hashlib.sha256(path.read_bytes()).hexdigest()
        called.append(group_id)
        record = expected[group_id]
        return {"features": record["features"], "sampling": record["sampling"],
                "encoder_provenance": record["encoder_provenance"]}
    records = pilot.extract_features(frozen, case.protocol, root=case.root, extractor=extractor, encoder=object())
    assert set(records) == set(called) == {row["group_id"] for row in frozen["examples"]}
    assert len(called) == 8
    assert not set(called) & set(case.protocol["folds"][0]["valid_group_ids"] + case.protocol["holdout_group_ids"])


def test_extractor_failure_does_not_echo_private_details(case, tmp_path):
    frozen = manifest(case, tmp_path)
    def broken(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_PRIVATE_DETAIL")
    with pytest.raises(pilot.PilotError) as error:
        pilot.extract_features(frozen, case.protocol, root=case.root, extractor=broken, encoder=object())
    assert "SYNTHETIC_PRIVATE_DETAIL" not in str(error.value)


def test_prepare_cli_with_zero_labels_never_loads_encoder(case, tmp_path, monkeypatch, capsys):
    selected = pilot.build_selection(case.protocol, export(case, []))
    monkeypatch.setattr(pilot, "_inputs", lambda args: (tmp_path, case.protocol, None, {}, export(case, []), selected))
    monkeypatch.setattr(pilot, "encoder_provenance", lambda: pytest.fail("No labels must gate before encoder import"))
    assert pilot.main([]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "blocked" and "score" not in output and "examples" not in output
    assert not list(tmp_path.glob("*.joblib"))
    assert not (tmp_path / "manifest.json").exists()


def test_sanity_with_zero_labels_is_blocked_without_scores(case, tmp_path, monkeypatch, capsys):
    selected = pilot.build_selection(case.protocol, export(case, []))
    monkeypatch.setattr(pilot, "_inputs", lambda args: (tmp_path, case.protocol, None, {}, export(case, []), selected))
    assert pilot.sanity_main([]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "blocked"
    assert not any("auc" in key or "score" in key for key in output)


def test_actual_entrypoint_freezes_fitted_head_before_requesting_blind_labels(case, tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    hidden = export(case)
    selected = pilot.build_selection(case.protocol, hidden)
    released_tokens = []
    def load_labels(**kwargs):
        assert kwargs["include_blind"] is True
        frozen = json.loads((run / "manifest.json").read_text())
        artifact = pilot._verify_artifact(run, frozen)
        assert kwargs["blind_release_fingerprint"] == artifact["release_fingerprint"]
        assert len(frozen["examples"]) == 8 and {row["pool"] for row in frozen["examples"]} == {"train"}
        released_tokens.append(artifact["release_fingerprint"])
        return export(case, released=True, token=artifact["release_fingerprint"])
    helper = SimpleNamespace(load_transfer_labels=load_labels)
    monkeypatch.setattr(pilot, "_inputs", lambda args: (run, case.protocol, helper, {}, hidden, selected))
    monkeypatch.setattr(pilot, "encoder_provenance", lambda: ENCODER)
    monkeypatch.setattr(pilot, "_source_provenance", lambda: SOURCES)
    monkeypatch.setattr(pilot, "extract_features", lambda manifest, *args, **kwargs: feature_records(manifest["examples"]))
    assert pilot.main(["--stage", "evaluate"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["classification_feasibility_pass"]
    assert len(released_tokens) == 1
    before = (run / "trained_artifact.json").read_bytes()
    assert pilot.main(["--stage", "evaluate"]) == 1
    assert (run / "trained_artifact.json").read_bytes() == before
    assert len(released_tokens) == 1
    capsys.readouterr()
    # The real sanity entry point rechecks immutable bytes and replays the saved
    # fitted head before twenty synthetic shuffled-target fits.
    monkeypatch.setattr(config, "ROOT", case.root)
    assert pilot.sanity_main([]) in (0, 3)
    sanity = json.loads(capsys.readouterr().out)
    assert sanity["shuffled_trials"] == 20
    assert sanity["source_hashes_verified"] is True
    assert sanity["shared_recordings"] == sanity["holdout_opened"] == 0
    assert (run / "trained_artifact.json").read_bytes() == before


def test_twenty_shuffle_fits_are_deterministic_and_do_not_modify_labels(case, tmp_path):
    frozen = manifest(case, tmp_path)
    blind = released_blind(case, frozen)
    before = deepcopy(frozen)
    records, blind_records = feature_records(frozen["examples"]), feature_records(blind)
    first = pilot.sanity_features(frozen, records, blind, blind_records)
    second = pilot.sanity_features(frozen, records, blind, blind_records)
    assert first == second
    assert first["shuffled_trials"] == 20
    assert 0 <= first["shuffled_auc_mean"] <= 1
    assert frozen == before
