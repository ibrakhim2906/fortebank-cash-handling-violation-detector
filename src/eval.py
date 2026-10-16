"""FROZEN validation + metric. Humans review this once, then nobody edits it.

Split logic:
- TIME_COL set   → expanding-window time split (train on earlier rows, validate on later).
                   Out-of-fold predictions exist only for the later folds; the first chunk is train-only.
- GROUP_COL set  → group K-fold: one entity never appears in both train and validation.
- otherwise      → stratified K-fold (binary) or K-fold (regression).
"""
import numpy as np
from sklearn import metrics
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold,
                                     StratifiedKFold, TimeSeriesSplit)

from src import config as C

HIGHER_IS_BETTER = C.METRIC not in ("rmse", "mae")


def make_folds(df):
    n = len(df)
    y = df[C.TARGET].values
    if C.TIME_COL:
        order = np.argsort(df[C.TIME_COL].values, kind="stable")
        tss = TimeSeriesSplit(n_splits=C.N_FOLDS)
        return [(order[tr], order[va]) for tr, va in tss.split(order)]
    if C.GROUP_COL:
        groups = df[C.GROUP_COL].values
        if C.TASK == "binary":
            spl = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
            return list(spl.split(np.zeros(n), y, groups))
        return list(GroupKFold(n_splits=C.N_FOLDS).split(np.zeros(n), y, groups))
    if C.TASK == "binary":
        spl = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
        return list(spl.split(np.zeros(n), y))
    return list(KFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED).split(np.zeros(n)))


def score(y_true, pred):
    m = C.METRIC
    if m == "roc_auc":
        return metrics.roc_auc_score(y_true, pred)
    if m == "pr_auc":
        return metrics.average_precision_score(y_true, pred)
    if m == "gini":
        return 2 * metrics.roc_auc_score(y_true, pred) - 1
    if m == "f1":
        return metrics.f1_score(y_true, (np.asarray(pred) >= 0.5).astype(int))
    if m == "rmse":
        return float(np.sqrt(metrics.mean_squared_error(y_true, pred)))
    if m == "mae":
        return metrics.mean_absolute_error(y_true, pred)
    raise ValueError(f"unknown metric {m}")


def cross_validate(fit_predict, X, y, folds):
    """fit_predict(X_tr, y_tr, X_va, y_va) -> predictions for X_va."""
    oof = np.full(len(y), np.nan)
    fold_scores = []
    for tr, va in folds:
        pred = fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
        oof[va] = pred
        fold_scores.append(score(y.iloc[va], pred))
    mask = ~np.isnan(oof)
    return {"score": score(y.values[mask], oof[mask]), "fold_scores": fold_scores, "oof": oof}
