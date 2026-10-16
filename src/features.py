"""Dataset building and feature helpers. Codex extends build_dataset() for the real case."""
import numpy as np
import pandas as pd

from src import config as C
from src.data import load_test, load_train


def agg_windows(events, id_col, time_col, value_cols, ref,
                windows=(30, 90, 180), aggs=("count", "sum", "mean", "std", "max"),
                prefix="ev", group_by=None):
    """Point-in-time aggregates of a long event table (transactions, payments, ...).

    events:   long table with id_col (e.g. client_id), time_col and value columns.
    ref:      a single cutoff date for everyone, or a DataFrame with id_col + "ref_date"
              (the decision date of each row). Only events strictly BEFORE the cutoff are
              used, so features never see the future.
    group_by: column in ref to aggregate by, when one client has several rows with
              different decision dates (e.g. "application_id"). Defaults to id_col.
    Returns one row per group with window aggregates, recency and a 30d-vs-previous trend.
    """
    key = group_by or id_col
    value_cols = list(value_cols)
    ev = events[[id_col, time_col] + value_cols].copy()
    ev[time_col] = pd.to_datetime(ev[time_col])
    if isinstance(ref, pd.DataFrame):
        ref_cols = [id_col, "ref_date"] + ([key] if key != id_col else [])
        r = ref[ref_cols].copy()
        r["ref_date"] = pd.to_datetime(r["ref_date"])
        ev = ev.merge(r, on=id_col, how="inner")
    else:
        if key != id_col:
            raise ValueError("group_by needs a per-row ref DataFrame")
        ev["ref_date"] = pd.Timestamp(ref)
    ev = ev[ev[time_col] < ev["ref_date"]]
    ev["_age_days"] = (ev["ref_date"] - ev[time_col]).dt.days

    parts = []
    for w in windows:
        g = ev[ev["_age_days"] <= w].groupby(key)[value_cols].agg(list(aggs))
        g.columns = [f"{prefix}_{col}_{agg}_{w}d" for col, agg in g.columns]
        parts.append(g)
    parts.append(ev.groupby(key)["_age_days"].min().rename(f"{prefix}_days_since_last"))
    res = pd.concat(parts, axis=1)

    if 30 in windows and 90 in windows and "sum" in aggs:
        for col in value_cols:
            recent = res[f"{prefix}_{col}_sum_30d"].fillna(0)
            prev = (res[f"{prefix}_{col}_sum_90d"].fillna(0) - recent) / 2  # per-30d average of previous 60d
            res[f"{prefix}_{col}_trend_30d_vs_prev"] = recent / prev.replace(0, np.nan)
    count_cols = [c for c in res.columns if "_count_" in c]
    res[count_cols] = res[count_cols].fillna(0)  # no events in a window = 0, not missing
    res.index.name = key
    return res.reset_index()


def build_dataset():
    """Return (train_df, test_df_or_None) with ONE ROW PER PREDICTION UNIT.

    Default: the raw train/test files as they are. For multi-table data, build features
    here with agg_windows() + data.safe_merge(), using only information available at the
    decision moment of each row.
    """
    train, test = load_train(), load_test()
    return train, test


def feature_columns(df):
    excluded = {C.ID_COL, C.TARGET, C.TIME_COL, C.GROUP_COL, *C.DROP_COLS}
    return [c for c in df.columns if c not in excluded]


def to_model_frame(train, other=None):
    """Turn dataset rows into model inputs. Categories of `other` are aligned to train."""
    cols = feature_columns(train)
    X = train[cols].copy()
    Xo = other[cols].copy() if other is not None else None
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            print(f"[features] dropping raw datetime column '{c}' — engineer it explicitly")
            X = X.drop(columns=c)
            if Xo is not None:
                Xo = Xo.drop(columns=c)
            continue
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
            if Xo is not None:
                Xo[c] = Xo[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):  # object, string or category → category
            X[c] = X[c].astype("category")
            if Xo is not None:
                Xo[c] = pd.Categorical(Xo[c], categories=X[c].cat.categories)
    return X, Xo
