import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.make_synthetic import generate
from src import artifacts as A
from src import config as C
from src import impact, train


@pytest.fixture
def run_context(tmp_path, monkeypatch):
    for name in (*A.SETTINGS, "TRAIN_FILE", "TEST_FILE", "REPORTS", "MODELS", "DATA_PROC"):
        monkeypatch.setattr(C, name, getattr(C, name))
    monkeypatch.setenv("DATATHON_THREADS", "2")
    output = tmp_path / "run"
    output.mkdir()
    generate(output / "input", 2400, 200)
    A.write_json(output / "config.json", {
        "TRAIN_FILE": "input/train.csv", "TEST_FILE": "input/test.csv", "N_FOLDS": 2,
        "LABEL_GAP_DAYS": 10, "VALIDATION_WINDOWS": [["2025-08-01", "2025-11-01"], ["2025-11-01", "2026-02-01"]],
        "HOLDOUT_WINDOW": ["2026-02-01", "2026-03-01"], "GROUP_COL": None,
    })
    A.configure_run(output)
    for path in (C.REPORTS, C.MODELS, C.DATA_PROC):
        path.mkdir()
    return output


def freeze(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["src.impact"])
    impact.main()


def test_generator_is_deterministic_and_refuses_overwrites(tmp_path):
    first, test = generate(tmp_path / "first", 40, 10)
    second, _ = generate(tmp_path / "second", 40, 10)
    pd.testing.assert_frame_equal(first, second)
    assert "target" not in test
    with pytest.raises(ValueError, match="empty"):
        generate(tmp_path / "first", 40, 10)
    with pytest.raises(ValueError, match="outside data/raw"):
        generate(C.DATA_RAW / "forbidden")


def test_generator_rejects_raw_symlink(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    shortcut = tmp_path / "shortcut"
    shortcut.symlink_to(raw, target_is_directory=True)
    from unittest.mock import patch
    with patch.object(C, "DATA_RAW", raw):
        with pytest.raises(ValueError, match="outside data/raw"):
            generate(shortcut)


def test_schema_is_fit_only_on_training_categories():
    earlier = pd.DataFrame({"category": ["known", "known"], "number": [1, 2]})
    later = pd.DataFrame({"category": ["unseen"], "number": [3]})
    schema = A.fit_schema(earlier)
    assert schema["categories"] == {"category": ["known"]}
    assert A.align_frame(later, schema).category.isna().all()


def test_runtime_seed_is_used_by_model(monkeypatch):
    from src import models
    monkeypatch.setattr(C, "SEED", 123)
    assert models.make_lgbm().get_params()["random_state"] == C.SEED


def test_development_excludes_holdout_predictions_and_labels(run_context):
    manifest = train.develop(["dummy", "linear", "lgbm"], "test synthetic")
    frame, _ = train.build_dataset()
    heldout = pd.to_datetime(frame.application_date).ge("2026-02-01")
    assert np.isnan(np.load(C.REPORTS / "oof_lgbm.npy")[heldout]).all()
    assert np.isnan(np.load(C.REPORTS / "oof_target.npy")[heldout]).all()
    assert manifest["state"] == "developed"
    assert not list(C.MODELS.iterdir())
    assert len(A.development_rows(frame)) == int((~heldout).sum())
    assert not (C.ROOT / "experiments.md").samefile(A.experiment_path())


def test_finalization_order_and_heldout_labels(run_context, monkeypatch):
    train.develop(["lgbm"], "test synthetic")
    freeze(monkeypatch)
    full, _ = train.build_dataset()
    heldout_indices = set(np.flatnonzero(pd.to_datetime(full.application_date).ge("2026-02-01")))
    events = []
    fit_original, predict_original, score_original = train.fit_fixed, A.predict, train.score

    def checked_fit(frame, labels, manifest):
        events.append(("fit", set(frame.index)))
        if len(events) == 1:
            assert not heldout_indices & set(labels.index)
        else:
            assert any(event[0] == "score" for event in events)
            assert (C.REPORTS / "holdout.json").exists()
        return fit_original(frame, labels, manifest)

    def checked_predict(bundle, frame):
        events.append(("predict", set(frame.index)))
        return predict_original(bundle, frame)

    def checked_score(labels, predictions):
        assert events[-1][0] == "predict"
        events.append(("score", set(labels.index)))
        return score_original(labels, predictions)

    monkeypatch.setattr(train, "fit_fixed", checked_fit)
    monkeypatch.setattr(A, "predict", checked_predict)
    monkeypatch.setattr(train, "score", checked_score)
    result = train.finalize()
    assert [event[0] for event in events[:4]] == ["fit", "predict", "score", "fit"]
    assert events[2][1] == heldout_indices
    assert result["holdout"]["threshold"] == result["policy"]["threshold"]
    assert result["state"] == "finalized"
    with pytest.raises(ValueError, match="only once"):
        train.finalize()


def test_frozen_policy_required_and_stale_files_rejected(run_context, monkeypatch):
    train.develop(["lgbm"], "test synthetic")
    with pytest.raises(ValueError, match="frozen policy"):
        train.finalize()
    freeze(monkeypatch)
    with pytest.raises(ValueError, match="already frozen"):
        impact.main()
    manifest = A.load_manifest()
    manifest["policy"]["threshold"] = 0.99
    A.save_manifest(manifest)
    with pytest.raises(ValueError, match="Frozen policy"):
        A.load_manifest()


def test_artifact_and_source_configuration_integrity(run_context):
    train.develop(["lgbm"], "test synthetic")
    original = C.VALUE_TP
    C.VALUE_TP += 1
    with pytest.raises(ValueError, match="Stale run"):
        A.load_manifest()
    C.VALUE_TP = original
    with (C.REPORTS / "oof_lgbm.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="incompatible artifact"):
        A.load_manifest()


def test_comparison_cannot_silently_become_delivery(run_context):
    with pytest.raises(ValueError, match="Include lgbm"):
        train.develop(["cat"], "test")


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "order", "extra_column", "nan", "range"])
def test_submission_contract(run_context, mutation):
    test = pd.DataFrame({C.ID_COL: [1, 2]})
    submission = pd.DataFrame({C.ID_COL: [1, 2], "prediction": [0.2, 0.8]})
    if mutation == "duplicate":
        submission[C.ID_COL] = [1, 1]
    elif mutation == "missing":
        submission[C.ID_COL] = [1, None]
    elif mutation == "order":
        submission[C.ID_COL] = [2, 1]
    elif mutation == "extra_column":
        submission["index"] = [0, 1]
    elif mutation == "nan":
        submission.loc[0, "prediction"] = np.nan
    elif mutation == "range":
        submission.loc[0, "prediction"] = 1.2
    with pytest.raises(ValueError):
        A.validate_submission(submission, test)


def test_regression_outputs_are_not_limited_to_probabilities(run_context, monkeypatch):
    monkeypatch.setattr(C, "TASK", "regression")
    monkeypatch.setattr(C, "METRIC", "rmse")
    monkeypatch.setattr(C, "TIME_COL", None)
    monkeypatch.setattr(C, "VALIDATION_WINDOWS", None)
    monkeypatch.setattr(C, "HOLDOUT_WINDOW", None)
    train.develop(["dummy", "linear", "lgbm"], "regression smoke")
    result = train.finalize()
    assert "not configured" in result["holdout"]["status"]
    A.validate_submission(pd.DataFrame({C.ID_COL: [1], "prediction": [-12.0]}), pd.DataFrame({C.ID_COL: [1]}))


def test_demo_uses_reloaded_model_without_fitting(run_context, monkeypatch):
    train.develop(["lgbm"], "test synthetic")
    freeze(monkeypatch)
    train.finalize()
    monkeypatch.setenv("DATATHON_RUN_DIR", str(run_context))
    monkeypatch.setattr(sys, "argv", ["src.explain"])
    from src import explain
    explain.main()
    from scripts.rehearse import app_check
    # Streamlit installs each AppTest script as a new global __main__ module.
    # Restore the test runner so later spawn workers do not re-execute app.py.
    original_main = sys.modules["__main__"]
    try:
        app_check(run_context)
    finally:
        sys.modules["__main__"] = original_main
    assert sys.modules["__main__"] is original_main
    result = json.loads((C.REPORTS / "app_check.json").read_text())
    assert result["no_training"]
    assert result["csv_max_error"] < 1e-12
    assert {row["example"] for row in result["examples"]} == {"Low risk", "High risk", "Borderline"}
