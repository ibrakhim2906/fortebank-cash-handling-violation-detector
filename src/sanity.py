"""Leakage and sanity checks. Run after every feature change: uv run python -m src.sanity"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config as C
from src import models as M
from src.eval import HIGHER_IS_BETTER, cross_validate, make_folds
from src.features import build_dataset, to_model_frame


def main():
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    train, test = build_dataset()
    train = train.reset_index(drop=True)
    X, X_test = to_model_frame(train, test)
    y = train[C.TARGET].reset_index(drop=True)
    folds = make_folds(train)
    lines, problems = ["# Sanity report", ""], []

    # 1. duplicate IDs
    if C.ID_COL in train.columns:
        dup = int(train[C.ID_COL].duplicated().sum())
        lines.append(f"- Duplicate {C.ID_COL} in train: {dup}")
        if dup and not C.GROUP_COL:
            problems.append(f"{dup} duplicate IDs but GROUP_COL is not set → same entity can sit in train and validation")
        if test is not None and C.ID_COL in test.columns:
            shared = len(set(train[C.ID_COL]) & set(test[C.ID_COL]))
            lines.append(f"- IDs shared between train and test: {shared}")

    # 2. single-feature leakage scan
    if C.TASK == "binary":
        suspicious = []
        for c in X.columns:
            s = X[c]
            if isinstance(s.dtype, pd.CategoricalDtype):
                s = s.cat.codes.replace(-1, np.nan)
            if s.nunique() < 2:
                continue
            auc = roc_auc_score(y, s.fillna(s.median()))
            auc = max(auc, 1 - auc)
            if auc > 0.90:
                suspicious.append((c, auc))
        lines.append(f"- Features with single-feature AUC > 0.90: {suspicious or 'none'}")
        if suspicious:
            problems.append(f"possible leak: {suspicious}")

    # 3. shuffled-target test
    rng = np.random.default_rng(C.SEED)
    y_shuf = pd.Series(rng.permutation(y.values))
    shuf = cross_validate(M.fit_lgbm, X, y_shuf, folds)["score"]
    dummy = cross_validate(M.fit_dummy, X, y_shuf, folds)["score"]
    gain = (shuf - dummy) if HIGHER_IS_BETTER else (dummy - shuf)
    lines.append(f"- Shuffled-target LightGBM {C.METRIC}: {shuf:.4f} (dummy on shuffled: {dummy:.4f})")
    tol = 0.03 if C.METRIC in ("roc_auc", "gini", "pr_auc", "f1") else 0.02 * abs(dummy)
    if gain > tol:
        problems.append("model beats chance on a SHUFFLED target → pipeline leaks target information")

    # 4. adversarial validation
    if X_test is not None:
        Xa = pd.concat([X, X_test], ignore_index=True)
        for c in M.cat_cols(X):
            Xa[c] = Xa[c].astype("category")
        ya = np.r_[np.zeros(len(X)), np.ones(len(X_test))]
        oof = np.zeros(len(Xa))
        imp = pd.Series(0.0, index=Xa.columns)
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=C.SEED).split(Xa, ya):
            m = M.make_lgbm(n_estimators=300, learning_rate=0.05)
            m.fit(Xa.iloc[tr], ya[tr])
            oof[va] = m.predict_proba(Xa.iloc[va])[:, 1]
            imp += pd.Series(m.feature_importances_, index=Xa.columns)
        adv = roc_auc_score(ya, oof)
        top = imp.sort_values(ascending=False).head(5).index.tolist()
        lines.append(f"- Adversarial validation AUC (train vs test): {adv:.4f}; most separating features: {top}")
        if adv > 0.7:
            problems.append(f"train and test differ (adversarial AUC {adv:.3f}); check {top}")

    lines += ["", "## Problems", *(f"- ❌ {p}" for p in problems)] if problems else ["", "## Problems", "- ✅ none found"]
    report = "\n".join(lines)
    (C.REPORTS / "sanity.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
