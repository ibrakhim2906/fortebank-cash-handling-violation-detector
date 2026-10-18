"""Run an isolated synthetic rehearsal, then replay in a clean source snapshot."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from src import artifacts as A
from src import config as C

ROOT = Path(__file__).resolve().parents[1]


def protected_files(root):
    names = ["src/eval.py", "src/config.py", "docs/brief.md", "docs/context.md", "docs/jury.md",
             "CHECKLIST.md", "experiments.md", "pyproject.toml", "uv.lock"]
    names += [str(path.relative_to(root)) for path in (root / "data/raw").rglob("*") if path.is_file()]
    return {name: A.digest(root / name) for name in names}


def run_command(arguments, cwd, environment, logs, label):
    command = ["uv", "run", "--offline", *arguments]
    started = time.perf_counter()
    print(f"RUN {label}: {shlex.join(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    path = logs / f"{label}.txt"
    path.write_text(result.stdout)
    record = {"command": command, "cwd": str(cwd), "exit_code": result.returncode,
              "seconds": time.perf_counter() - started, "log": str(path)}
    A.write_json(logs / f"{label}.json", record)
    print(result.stdout[-5000:], flush=True)
    if result.returncode:
        raise RuntimeError(f"{label} failed; stop and inspect {path}")
    return record


def app_check(directory):
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    from src.features import build_dataset
    A.configure_run(directory)
    manifest = A.load_manifest()
    bundle = A.load_bundle()
    _, test = build_dataset()
    expected = A.predict(bundle, A.inputs(test))
    submission = pd.read_csv(C.REPORTS / "submission.csv")
    A.validate_submission(submission, test)
    np.testing.assert_allclose(expected, submission.prediction, rtol=0, atol=1e-12)
    started = time.perf_counter()
    with patch("lightgbm.LGBMClassifier.fit", side_effect=AssertionError("App must not train")):
        application = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
        initial_seconds = time.perf_counter() - started
        if application.exception or application.error:
            raise AssertionError(f"App startup failed: {application.exception} {application.error}")
        checked = []
        for example in ("Low risk", "High risk", "Borderline"):
            application.selectbox[0].select(example).run()
            if application.exception:
                raise AssertionError(str(application.exception))
            selected = application.selectbox[1].value
            position = test[C.ID_COL].tolist().index(selected)
            value = float(expected[position])
            assert application.metric[0].value == f"{value:.1%}"
            decision = "Flag for review" if value >= manifest["policy"]["threshold"] else "OK"
            assert application.metric[1].value == decision
            assert len(application.dataframe[0].value) == min(5, len(bundle["columns"]))
            checked.append({"example": example, "id": int(selected), "prediction": value, "decision": decision})
        restarted = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
        assert not restarted.exception
    if initial_seconds >= 10:
        raise AssertionError(f"App startup {initial_seconds:.3f}s exceeds 10s target")
    result = {"initial_seconds": initial_seconds, "examples": checked, "csv_max_error": float(np.max(np.abs(expected - submission.prediction))),
              "no_training": True, "browser_clickthrough": "pending; automated Streamlit UI exercised"}
    A.write_json(C.REPORTS / "app_check.json", result)
    print(json.dumps(result, indent=2))


def baseline(source, output, config, environment, logs):
    directory = output / "baseline"
    directory.mkdir()
    for name in ("reports", "models", "processed"):
        (directory / name).mkdir()
    (directory / "experiments.md").write_text("# Synthetic baseline experiments\n")
    code = (
        "import json,runpy,sys; from pathlib import Path; from src import config as C; "
        f"settings=json.loads({json.dumps(json.dumps(config))}); "
        "[(setattr(C,key,Path(value) if key in ('TRAIN_FILE','TEST_FILE') else value)) for key,value in settings.items()]; "
        f"C.ROOT=Path({str(directory)!r}); C.REPORTS=C.ROOT/'reports'; C.MODELS=C.ROOT/'models'; C.DATA_PROC=C.ROOT/'processed'; "
        "from src import models as M; M.LGB_PARAMS['n_jobs']=2; "
        "from src.features import build_dataset,to_model_frame; from src.eval import cross_validate,make_folds; "
        "from src.train import log_experiment; import numpy as np; "
        "training,test=build_dataset(); training=training.reset_index(drop=True); inputs,_=to_model_frame(training,test); "
        "labels=training[C.TARGET]; folds=make_folds(training); result=cross_validate(M.fit_lgbm,inputs,labels,folds); "
        "np.save(C.REPORTS/'oof_lgbm.npy',result['oof']); "
        "log_experiment('SYNTHETIC before hardening [lgbm]',result['score'],'development OOF only; no legacy full refit'); "
        "print('BASELINE development-only',result['score'],result['fold_scores'])"
    )
    baseline_environment = dict(environment)
    baseline_environment.pop("DATATHON_RUN_DIR", None)
    baseline_environment["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".venv")
    run_command(["python", "-c", code], source, baseline_environment, logs, "baseline")
    predictions = np.load(directory / "reports/oof_lgbm.npy")
    labels = pd.read_csv(config["TRAIN_FILE"])[C.TARGET].to_numpy()
    from src.eval import score
    mask = np.isfinite(predictions)
    return {"score": score(labels[mask], predictions[mask]), "oof": predictions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--app-check", type=Path)
    args = parser.parse_args()
    if args.app_check:
        app_check(args.app_check)
        return
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    if output == C.DATA_RAW.resolve() or C.DATA_RAW.resolve() in output.parents:
        raise ValueError("Rehearsal output must be outside data/raw")
    if output.exists():
        raise ValueError("Use a fresh rehearsal output directory")
    before = protected_files(ROOT)
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    environment = dict(os.environ, DATATHON_RUN_DIR=str(output), DATATHON_THREADS="2",
                       UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR", "/tmp/datathon-uv-cache"),
                       MPLCONFIGDIR="/tmp/datathon-matplotlib", OMP_NUM_THREADS="2")
    records = []
    records.append(run_command(["python", "-m", "scripts.make_synthetic", "--output", str(output / "input")], ROOT, environment, logs, "generate"))
    settings = {"TRAIN_FILE": "input/train.csv", "TEST_FILE": "input/test.csv", "GROUP_COL": None,
                "N_FOLDS": 2, "LABEL_GAP_DAYS": 30, "VALIDATION_WINDOWS": [["2025-10-01", "2025-12-01"], ["2025-12-01", "2026-02-01"]],
                "HOLDOUT_WINDOW": ["2026-02-01", "2026-03-01"], "EXCLUDE_VALIDATION_GROUPS": False}
    A.write_json(output / "config.json", settings)
    A.write_json(output / "synthetic.json", {"seed": C.SEED, "note": "Synthetic rehearsal only; gap/costs are illustrative"})
    frame = pd.read_csv(output / "input/train.csv")
    A.write_json(output / "overview.json", {"train_shape": list(frame.shape), "target_rate": float(frame.target.mean()),
                                            "start": frame.application_date.min(), "end": frame.application_date.max(),
                                            "missing_cells": int(frame.isna().sum().sum())})
    old = None
    if args.baseline_source:
        baseline_settings = {**settings, "TRAIN_FILE": str(output / "input/train.csv"), "TEST_FILE": str(output / "input/test.csv")}
        old = baseline(args.baseline_source.resolve(), output, baseline_settings, environment, logs)
    for label, arguments in [
        ("develop", ["-m", "src.train", "--models", "dummy,linear,lgbm,cat", "--note", "SYNTHETIC rehearsal"]),
        ("sanity", ["-m", "src.sanity"]),
        ("impact", ["-m", "src.impact", "--oof", "lgbm"]),
        ("finalize", ["-m", "src.train", "--stage", "finalize"]),
        ("explain", ["-m", "src.explain"]),
        ("app", ["-m", "scripts.rehearse", "--app-check", str(output)]),
    ]:
        records.append(run_command(["python", *arguments], ROOT, environment, logs, label))
    manifest = json.loads((output / "reports/manifest.json").read_text())
    result = {"synthetic": True, "development": manifest["development"], "holdout": manifest["holdout"],
              "commands": records, "app": json.loads((output / "reports/app_check.json").read_text())}
    if old is not None:
        current = np.load(output / "reports/oof_lgbm.npy")
        np.testing.assert_allclose(old["oof"], current, rtol=0, atol=1e-12, equal_nan=True)
        result["before_after"] = {"before": old["score"], "after": manifest["development"]["lgbm"]["score"], "oof_identical": True}
    if not args.skip_replay:
        replay_root = Path(tempfile.mkdtemp(prefix="datathon-replay-"))
        source = replay_root / "source"
        shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(".git", ".venv", "data", "models", "reports", "__pycache__", ".pytest_cache", ".codex", ".agents"))
        replay_output = replay_root / "run"
        replay_environment = dict(environment, UV_PROJECT_ENVIRONMENT=str(ROOT / ".venv"))
        replay_environment.pop("DATATHON_RUN_DIR", None)
        run_command(["python", "-m", "scripts.rehearse", "--output", str(replay_output), "--skip-replay"], source, replay_environment, logs, "clean-replay")
        for filename in ("oof_lgbm.npy", "holdout_predictions.npy"):
            np.testing.assert_allclose(np.load(output / "reports" / filename), np.load(replay_output / "reports" / filename), rtol=0, atol=1e-12, equal_nan=True)
        first = pd.read_csv(output / "reports/submission.csv")
        second = pd.read_csv(replay_output / "reports/submission.csv")
        pd.testing.assert_frame_equal(first, second, rtol=0, atol=1e-12)
        result["replay"] = {"source": str(source), "output": str(replay_output), "identical_predictions": True,
                            "environment": "Locked existing uv environment reused offline; not a fresh package download"}
    if before != protected_files(ROOT):
        raise AssertionError("Protected raw/config/case files changed")
    result["protected_files_unchanged"] = before
    A.write_json(output / "result.json", result)
    print("REHEARSAL COMPLETE", json.dumps({"development": result["development"], "holdout": result["holdout"], "before_after": result.get("before_after"), "replay": result.get("replay")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
