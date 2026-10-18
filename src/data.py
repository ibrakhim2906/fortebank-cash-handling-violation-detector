"""Loading helpers. Never write into data/raw."""
from pathlib import Path

import pandas as pd

from src import config as C


def read_any(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def _parse_time(df):
    if C.TIME_COL and C.TIME_COL in df.columns:
        df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL])
    return df


def load_train():
    df = read_any(C.TRAIN_FILE)
    assert C.TARGET in df.columns, f"target '{C.TARGET}' not found in {C.TRAIN_FILE}"
    return _parse_time(df)


def load_test():
    if C.TEST_FILE is None or not Path(C.TEST_FILE).exists():
        return None
    return _parse_time(read_any(C.TEST_FILE))


def safe_merge(left, right, on, how="left", validate="m:1"):
    """Merge that refuses to silently duplicate or drop rows of `left`."""
    n = len(left)
    out = left.merge(right, on=on, how=how, validate=validate)
    if how == "left":
        assert len(out) == n, f"merge changed row count {n} -> {len(out)}"
    return out
