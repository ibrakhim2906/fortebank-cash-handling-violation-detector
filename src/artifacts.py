"""Run configuration, training-only schemas and checked artifact loading."""
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.features import feature_columns

SETTINGS = ("ID_COL", "TARGET", "TIME_COL", "GROUP_COL", "DROP_COLS", "TASK", "METRIC",
            "N_FOLDS", "SEED", "LABEL_GAP_DAYS", "VALIDATION_WINDOWS", "HOLDOUT_WINDOW",
            "EXCLUDE_VALIDATION_GROUPS", "VALUE_TP", "COST_FP")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def configure_run(directory=None):
    directory = directory or os.environ.get("DATATHON_RUN_DIR")
    if not directory:
        return
    directory = Path(directory).resolve()
    settings_path = directory / "config.json"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        unknown = set(settings) - set(SETTINGS) - {"TRAIN_FILE", "TEST_FILE"}
        if unknown:
            raise ValueError(f"Unsupported runtime settings: {sorted(unknown)}")
        for name, value in settings.items():
            if name in ("TRAIN_FILE", "TEST_FILE") and value is not None:
                value = (directory / value).resolve()
            setattr(C, name, value)
    C.REPORTS = directory / "reports"
    C.MODELS = directory / "models"
    C.DATA_PROC = directory / "processed"


def experiment_path():
    return C.REPORTS.parent / "experiments.md" if C.REPORTS != C.ROOT / "reports" else C.ROOT / "experiments.md"


def fingerprint():
    source_root = Path(__file__).resolve().parent
    return {
        "settings": json.loads(json.dumps({name: getattr(C, name) for name in SETTINGS})),
        "inputs": {name: digest(path) if path is not None and Path(path).exists() else None
                   for name, path in (("train", C.TRAIN_FILE), ("test", C.TEST_FILE))},
        "raw_history": {str(path.relative_to(C.DATA_RAW)): digest(path)
                        for path in sorted(C.DATA_RAW.rglob("*")) if path.is_file()},
        "source": {path.name: digest(path) for path in sorted(source_root.glob("*.py"))},
        "threads": os.environ.get("DATATHON_THREADS", "-1"),
    }


def save_manifest(manifest):
    write_json(C.REPORTS / "manifest.json", manifest)


def load_manifest():
    path = C.REPORTS / "manifest.json"
    if not path.exists():
        raise ValueError("No run manifest; run the development stage first")
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1 or manifest.get("selected_model") != "lgbm":
        raise ValueError("Unsupported manifest or delivered model")
    if manifest["fingerprint"] != fingerprint():
        raise ValueError("Stale run: source, input data or configuration changed")
    for relative, expected in manifest["artifacts"].items():
        path = C.REPORTS.parent / relative
        if not path.exists() or digest(path) != expected:
            raise ValueError(f"Missing or incompatible artifact: {relative}")
    if "policy" in manifest:
        frozen = json.loads((C.REPORTS / "policy.json").read_text())
        if frozen != {"policy": manifest["policy"], "model_params": manifest["model_params"]}:
            raise ValueError("Frozen policy/model settings changed")
    return manifest


def record_artifact(manifest, path):
    path = Path(path)
    relative = str(path.relative_to(C.REPORTS.parent))
    manifest["artifacts"][relative] = digest(path)


def freeze_policy(manifest):
    path = C.REPORTS / "policy.json"
    write_json(path, {"policy": manifest["policy"], "model_params": manifest["model_params"]})
    record_artifact(manifest, path)


def inputs(frame):
    columns = [name for name in feature_columns(frame)
               if not pd.api.types.is_datetime64_any_dtype(frame[name])]
    return frame[columns].copy()


def fit_schema(frame):
    return {"columns": list(frame.columns), "categories": {
        name: sorted(frame[name].dropna().astype(str).unique().tolist())
        for name in frame.columns if not pd.api.types.is_numeric_dtype(frame[name])
    }}


def align_frame(frame, schema):
    missing = set(schema["columns"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}")
    result = frame[schema["columns"]].copy()
    for name in result:
        if name in schema["categories"]:
            values = result[name].astype("string")
            result[name] = pd.Categorical(values.where(values.isin(schema["categories"][name])),
                                          categories=schema["categories"][name])
        elif result[name].dtype == bool:
            result[name] = result[name].astype(int)
    return result


def development_rows(frame):
    if C.HOLDOUT_WINDOW is None:
        return frame.copy()
    dates = pd.to_datetime(frame[C.TIME_COL]).dt.normalize()
    start, end = map(pd.Timestamp, C.HOLDOUT_WINDOW)
    return frame.loc[~((dates >= start) & (dates < end))].copy()


def load_bundle():
    manifest = load_manifest()
    if manifest["state"] != "finalized":
        raise ValueError("No final model; freeze the policy and finalize this run first")
    bundle = joblib.load(C.MODELS / "lgbm.joblib")
    if bundle["run_id"] != manifest["run_id"] or bundle["model_name"] != manifest["selected_model"]:
        raise ValueError("Model/run identity mismatch")
    return bundle


def predict(bundle, frame):
    aligned = align_frame(frame, bundle["schema"])
    return (bundle["model"].predict_proba(aligned)[:, 1] if bundle["task"] == "binary"
            else bundle["model"].predict(aligned))


def validate_submission(submission, test):
    if list(submission.columns) != [C.ID_COL, "prediction"]:
        raise ValueError("Submission columns do not match the configured ID and prediction")
    for frame in (submission, test):
        if frame[C.ID_COL].isna().any() or frame[C.ID_COL].duplicated().any():
            raise ValueError("Missing or duplicate prediction IDs")
    if submission[C.ID_COL].tolist() != test[C.ID_COL].tolist():
        raise ValueError("Submission IDs/order do not exactly match test")
    values = submission.prediction.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Predictions must be finite")
    if C.TASK == "binary" and not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Binary predictions must be probabilities in [0, 1]")
