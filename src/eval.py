"""FROZEN validation + metric. Humans review this once, then nobody edits it.

Split logic:
- TIME_COL set   → expanding windows on unique dates, optionally configured explicitly.
                   Label gaps, holdout dates and validation-group exclusion are opt-in.
- GROUP_COL set  → group K-fold: one entity never appears in both train and validation.
- otherwise      → stratified K-fold (binary) or K-fold (regression).
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold,
                                     StratifiedKFold)

from src import config as C

HIGHER_IS_BETTER = C.METRIC not in ("rmse", "mae")


def _date_window(window):
    if isinstance(window, str) or len(window) != 2:
        raise ValueError("A validation window must contain a start and exclusive end")
    start, end = (pd.Timestamp(boundary).normalize() for boundary in window)
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Window boundaries must be valid dates with start < end")
    return start, end


def make_folds(df, holdout=False):
    if C.TIME_COL:
        dates = pd.to_datetime(df[C.TIME_COL], errors="raise").dt.normalize()
        if dates.isna().any():
            raise ValueError("Validation dates must be complete")
        if not isinstance(C.LABEL_GAP_DAYS, (int, np.integer)) or C.LABEL_GAP_DAYS < 0:
            raise ValueError("LABEL_GAP_DAYS must be a non-negative integer")
        groups = None
        if C.EXCLUDE_VALIDATION_GROUPS:
            if not C.GROUP_COL:
                raise ValueError("EXCLUDE_VALIDATION_GROUPS requires GROUP_COL")
            groups = df[C.GROUP_COL]
            if groups.isna().any():
                raise ValueError("Excluded validation groups must be complete")
        reserved = pd.Series(False, index=df.index)
        if C.HOLDOUT_WINDOW is not None:
            holdout_start, holdout_end = _date_window(C.HOLDOUT_WINDOW)
            reserved = (dates >= holdout_start) & (dates < holdout_end)
        if holdout:
            if C.HOLDOUT_WINDOW is None:
                raise ValueError("holdout=True requires HOLDOUT_WINDOW")
            windows = [(holdout_start, holdout_end)]
        elif C.VALIDATION_WINDOWS is not None:
            windows = [_date_window(window) for window in C.VALIDATION_WINDOWS]
        else:
            unique_dates = dates[~reserved].sort_values().unique()
            if C.N_FOLDS < 1 or len(unique_dates) < C.N_FOLDS + 1:
                raise ValueError("Automatic validation needs at least N_FOLDS + 1 unique dates")
            chunks = np.array_split(unique_dates, C.N_FOLDS + 1)
            windows = [(pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1]) + pd.Timedelta(days=1))
                       for chunk in chunks[1:]]
        if not windows:
            raise ValueError("No validation windows configured")
        order = np.argsort(dates.to_numpy(), kind="stable")
        label_end = dates + pd.Timedelta(days=C.LABEL_GAP_DAYS)
        folds = []
        previous_end = None
        for start, end in windows:
            if previous_end is not None and start < previous_end:
                raise ValueError("Validation windows must be ordered and non-overlapping")
            valid_mask = (dates >= start) & (dates < end)
            if not holdout:
                valid_mask &= ~reserved
            train_mask = (label_end < start) & ~reserved
            if groups is not None:
                train_mask &= ~groups.isin(groups[valid_mask])
            train_indices = order[train_mask.to_numpy()[order]]
            valid_indices = order[valid_mask.to_numpy()[order]]
            if not len(train_indices) or not len(valid_indices):
                raise ValueError(f"Empty training or validation fold starting {start.date()}")
            folds.append((train_indices, valid_indices))
            previous_end = end
        return folds
    if holdout or C.HOLDOUT_WINDOW is not None:
        raise ValueError("HOLDOUT_WINDOW requires TIME_COL")
    n = len(df)
    y = df[C.TARGET].values
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
