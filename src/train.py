"""Development comparisons followed by explicitly frozen holdout/final fitting."""
import argparse
import datetime as dt
import uuid

import joblib
import numpy as np
import pandas as pd

from src import artifacts as A
from src import config as C
from src import models as M
from src.eval import cross_validate, make_folds, score
from src.features import build_dataset


def log_experiment(change, value, notes="", keep="comparison"):
    path = A.experiment_path()
    if not path.exists():
        path.write_text("# Experiments\n\n| Time | Change | Score | Keep/drop | Notes |\n|---|---|---|---|---|\n")
    with path.open("a") as stream:
        stream.write(f"| {dt.datetime.now():%m-%d %H:%M} | {change} | {C.METRIC}={value:.5f} | {keep} | {notes} |\n")


def fold_predict(name):
    def fit_predict(train_frame, train_labels, valid_frame, valid_labels):
        schema = A.fit_schema(train_frame)
        return M.FIT[name](A.align_frame(train_frame, schema), train_labels,
                           A.align_frame(valid_frame, schema), valid_labels)
    return fit_predict


def fold_table(train, folds):
    rows = []
    for fold_index, (training, validation) in enumerate(folds):
        earlier, later = train.iloc[training], train.iloc[validation]
        rows.append({"fold": fold_index, "train_rows": len(training), "valid_rows": len(validation),
                     "train_dates": f"{earlier[C.TIME_COL].min()}..{earlier[C.TIME_COL].max()}" if C.TIME_COL else None,
                     "valid_dates": f"{later[C.TIME_COL].min()}..{later[C.TIME_COL].max()}" if C.TIME_COL else None,
                     "train_target_rate": float(earlier[C.TARGET].mean()),
                     "valid_target_rate": float(later[C.TARGET].mean()),
                     "shared_groups": len(set(earlier[C.GROUP_COL]) & set(later[C.GROUP_COL])) if C.GROUP_COL else None})
    print(pd.DataFrame(rows).to_string(index=False))
    return rows


def develop(names, note):
    if "lgbm" not in names:
        raise ValueError("Include lgbm: it is the supported delivered model; other models are comparisons")
    if (C.REPORTS / "manifest.json").exists():
        raise ValueError("Run already exists; use a fresh run directory")
    train, _ = build_dataset()
    train = train.reset_index(drop=True)
    frame, labels = A.inputs(train), train[C.TARGET]
    folds = make_folds(train)
    table = fold_table(train, folds)
    M.LGB_BEST_ITERS.clear()
    results = {}
    manifest = {"version": 1, "run_id": uuid.uuid4().hex, "selected_model": "lgbm",
                "fingerprint": A.fingerprint(), "state": "developed", "artifacts": {}, "development": {}}
    for name in names:
        result = cross_validate(fold_predict(name), frame, labels, folds)
        results[name] = result
        path = C.REPORTS / f"oof_{name}.npy"
        np.save(path, result["oof"])
        A.record_artifact(manifest, path)
        manifest["development"][name] = {"score": result["score"], "fold_scores": result["fold_scores"]}
        print(f"{name:7s} {C.METRIC}={result['score']:.5f} folds={result['fold_scores']}")
        log_experiment(f"{note} [{name}]", result["score"], "development only",
                       "keep: delivery" if name == "lgbm" else "comparison only")
    if "cat" in results:
        blend = (results["lgbm"]["oof"] + results["cat"]["oof"]) / 2
        mask = np.isfinite(blend)
        value = score(labels.to_numpy()[mask], blend[mask])
        np.save(C.REPORTS / "oof_blend.npy", blend)
        A.record_artifact(manifest, C.REPORTS / "oof_blend.npy")
        manifest["development"]["blend"] = {"score": value}
        log_experiment(f"{note} [blend]", value, "development comparison; not deployed")
        print(f"blend   {C.METRIC}={value:.5f} (comparison only)")
    selected_mask = np.isfinite(results["lgbm"]["oof"])
    np.save(C.REPORTS / "oof_target.npy", labels.where(selected_mask).to_numpy())
    A.record_artifact(manifest, C.REPORTS / "oof_target.npy")
    manifest["iterations"] = max(50, int(np.mean(M.LGB_BEST_ITERS) * 1.1))
    manifest["model_params"] = {**M.LGB_PARAMS, "random_state": C.SEED, "n_estimators": manifest["iterations"]}
    manifest["folds"] = table
    if C.TASK != "binary":
        manifest["policy"] = {"threshold": None, "model": "lgbm", "run_id": manifest["run_id"]}
        manifest["state"] = "policy_frozen"
        A.freeze_policy(manifest)
    A.save_manifest(manifest)
    return manifest


def fit_fixed(frame, labels, manifest):
    schema = A.fit_schema(frame)
    model = M.make_lgbm(**manifest["model_params"])
    model.fit(A.align_frame(frame, schema), labels)
    return {"model": model, "schema": schema, "columns": schema["columns"], "task": C.TASK,
            "run_id": manifest["run_id"], "model_name": "lgbm"}


def finalize():
    manifest = A.load_manifest()
    if manifest["state"] != "policy_frozen":
        raise ValueError("Finalization requires a frozen policy and can run only once")
    train, test = build_dataset()
    train = train.reset_index(drop=True)
    frame, labels = A.inputs(train), train[C.TARGET]
    manifest["state"] = "finalization_started"
    A.save_manifest(manifest)
    if C.HOLDOUT_WINDOW is not None:
        training, validation = make_folds(train, holdout=True)[0]
        bundle = fit_fixed(frame.iloc[training], labels.iloc[training], manifest)
        predictions = A.predict(bundle, frame.iloc[validation])
        value = score(labels.iloc[validation], predictions)
        heldout = {"score": value, "metric": C.METRIC, "rows": len(validation),
                   "fold": fold_table(train, [(training, validation)])[0],
                   "threshold": manifest["policy"]["threshold"]}
        if C.TASK == "binary":
            from src.impact import value_curve
            heldout["locked_policy"] = value_curve(labels.iloc[validation].to_numpy(), predictions,
                                                    [manifest["policy"]["threshold"]])[0]
        np.save(C.REPORTS / "holdout_predictions.npy", predictions)
        A.record_artifact(manifest, C.REPORTS / "holdout_predictions.npy")
        print(f"HOLDOUT {C.METRIC}={value:.5f}; policy locked before evaluation")
        log_experiment("final holdout [lgbm]", value, "untouched; no selection", "evaluation only")
    else:
        heldout = {"status": "not configured; no holdout performance claim"}
    A.write_json(C.REPORTS / "holdout.json", heldout)
    A.record_artifact(manifest, C.REPORTS / "holdout.json")
    manifest["holdout"] = heldout
    A.save_manifest(manifest)
    bundle = fit_fixed(frame, labels, manifest)
    joblib.dump(bundle, C.MODELS / "lgbm.joblib")
    A.record_artifact(manifest, C.MODELS / "lgbm.joblib")
    if test is not None:
        submission = pd.DataFrame({C.ID_COL: test[C.ID_COL], "prediction": A.predict(bundle, A.inputs(test))})
        A.validate_submission(submission, test)
        submission.to_csv(C.REPORTS / "submission.csv", index=False)
        A.record_artifact(manifest, C.REPORTS / "submission.csv")
        reloaded = joblib.load(C.MODELS / "lgbm.joblib")
        np.testing.assert_allclose(submission.prediction, A.predict(reloaded, A.inputs(test)), rtol=0, atol=1e-12)
    manifest["state"] = "finalized"
    A.save_manifest(manifest)
    print(f"Finalized run {manifest['run_id']}: LightGBM artifact and submission agree")
    return manifest


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--task", choices=["tabular", "video", "transfer-pilot"], default="tabular")
    selected, remaining = selector.parse_known_args()
    if selected.task == "transfer-pilot":
        from src.video_transfer_pilot import main as pilot_main
        raise SystemExit(pilot_main(remaining))
    if selected.task == "video":
        from src.video_train import main as video_main
        return video_main(remaining)
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="dummy,linear,lgbm")
    parser.add_argument("--note", default="experiment")
    parser.add_argument("--run-dir")
    parser.add_argument("--stage", choices=["develop", "finalize"], default="develop")
    args = parser.parse_args(remaining)
    A.configure_run(args.run_dir)
    for directory in (C.REPORTS, C.MODELS, C.DATA_PROC):
        directory.mkdir(parents=True, exist_ok=True)
    if args.stage == "develop":
        develop([name.strip() for name in args.models.split(",")], args.note)
    else:
        finalize()


if __name__ == "__main__":
    main()
