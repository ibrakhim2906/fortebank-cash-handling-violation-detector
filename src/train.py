"""Cross-validate models, log results, fit the final model, write a submission.

Usage: uv run python -m src.train --models dummy,linear,lgbm,cat --note "what changed"
"""
import argparse
import datetime as dt

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src import models as M
from src.eval import HIGHER_IS_BETTER, cross_validate, make_folds, score
from src.features import build_dataset, to_model_frame


def log_experiment(change, value, notes=""):
    with open(C.ROOT / "experiments.md", "a", encoding="utf-8") as f:
        f.write(f"| {dt.datetime.now():%m-%d %H:%M} | {change} | {C.METRIC}={value:.5f} |  | {notes} |\n")


def best_logged():
    """Best score already in experiments.md, for before→after comparison."""
    path = C.ROOT / "experiments.md"
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if f"{C.METRIC}=" in line:
            try:
                vals.append(float(line.split(f"{C.METRIC}=")[1].split()[0].strip("|")))
            except ValueError:
                pass
    if not vals:
        return None
    return max(vals) if HIGHER_IS_BETTER else min(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dummy,linear,lgbm")
    ap.add_argument("--note", default="experiment")
    args = ap.parse_args()
    for d in (C.REPORTS, C.MODELS, C.DATA_PROC):
        d.mkdir(parents=True, exist_ok=True)

    train, test = build_dataset()
    train = train.reset_index(drop=True)
    X, X_test = to_model_frame(train, test)
    y = train[C.TARGET].reset_index(drop=True)
    folds = make_folds(train)
    prev_best = best_logged()

    print(f"rows={len(X)} features={X.shape[1]} folds={len(folds)} metric={C.METRIC} "
          f"target_mean={y.mean():.4f}")
    for i, (tr, va) in enumerate(folds):
        print(f"  fold {i}: train={len(tr)} valid={len(va)} valid_target_mean={y.iloc[va].mean():.4f}")

    results = {}
    for name in args.models.split(","):
        name = name.strip()
        res = cross_validate(M.FIT[name], X, y, folds)
        results[name] = res
        np.save(C.REPORTS / f"oof_{name}.npy", res["oof"])
        print(f"{name:7s} {C.METRIC}={res['score']:.5f}  mean_fold={np.mean(res['fold_scores']):.5f}  "
              f"folds={np.round(res['fold_scores'], 4).tolist()}")
        log_experiment(f"{args.note} [{name}]", res["score"], f"features={X.shape[1]}")

    if "lgbm" in results and "cat" in results:
        blend = (results["lgbm"]["oof"] + results["cat"]["oof"]) / 2
        mask = ~np.isnan(blend)
        s = score(y.values[mask], blend[mask])
        np.save(C.REPORTS / "oof_blend.npy", blend)
        print(f"blend   {C.METRIC}={s:.5f}")
        log_experiment(f"{args.note} [blend lgbm+cat]", s, f"features={X.shape[1]}")
    np.save(C.REPORTS / "oof_target.npy", y.values)

    if prev_best is not None:
        print(f"best previously logged {C.METRIC}: {prev_best:.5f}")

    if "lgbm" in results:
        n_iter = int(np.mean(M.LGB_BEST_ITERS)) if M.LGB_BEST_ITERS else 500
        model = M.fit_lgbm_full(X, y, n_iter)
        joblib.dump({"model": model, "columns": list(X.columns)}, C.MODELS / "lgbm.joblib")
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("top features (split importance):")
        print(imp.head(20).to_string())
        if X_test is not None:
            pred = M._predict(model, X_test)
            sub = pd.DataFrame({C.ID_COL: test[C.ID_COL].values if C.ID_COL in test else np.arange(len(pred)),
                                "prediction": pred})
            sub.to_csv(C.REPORTS / "submission.csv", index=False)
            print(f"wrote {C.REPORTS / 'submission.csv'} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
