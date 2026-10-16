"""Worked example: build ONE ROW PER APPLICATION from applications + history tables.

This is the pattern to copy into src/features.py → build_dataset() when the real case has
several tables. Key ideas:
  1. The prediction unit is the application, and its decision moment is application_date.
  2. History tables are aggregated per application using ONLY events before that date
     (agg_windows with a per-row ref and group_by="application_id").
  3. Every join goes through safe_merge, so row counts can't silently change.
  4. build_dataset_naive() shows the classic mistake (aggregating ALL history) for comparison.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import read_any, safe_merge
from src.features import agg_windows

DATA = Path(__file__).resolve().parent / "data"
KEY, CLIENT, DATE = "application_id", "client_id", "application_date"


def _load():
    train = read_any(DATA / "applications_train.csv")
    test = read_any(DATA / "applications_test.csv")
    for df in (train, test):
        df[DATE] = pd.to_datetime(df[DATE])
    tx = read_any(DATA / "transactions.csv")
    pay = read_any(DATA / "payments.csv")
    return train, test, tx, pay


def _history_features(apps, tx, pay):
    ref = apps[[KEY, CLIENT, DATE]].rename(columns={DATE: "ref_date"})
    tx_f = agg_windows(tx, CLIENT, "txn_date", ["amount"], ref,
                       windows=(30, 90, 180), aggs=("count", "sum", "mean", "max"),
                       prefix="tx", group_by=KEY)
    pay_f = agg_windows(pay, CLIENT, "payment_date", ["days_past_due"], ref,
                        windows=(90, 180, 365), aggs=("mean", "max", "sum"),
                        prefix="pay", group_by=KEY)
    out = safe_merge(apps, tx_f, on=KEY, validate="1:1")
    out = safe_merge(out, pay_f, on=KEY, validate="1:1")
    # ratios: usually among the strongest bank features
    out["loan_to_income"] = out["loan_amount"] / out["income"]
    out["spend_90d_to_income"] = out["tx_amount_sum_90d"] / out["income"]
    # previous applications by the same client BEFORE this one (allowed: known at decision time)
    out = out.sort_values(DATE)
    out["n_previous_applications"] = out.groupby(CLIENT).cumcount()
    return out.sort_values(KEY).reset_index(drop=True)


def build_dataset():
    """Point-in-time correct. Returns (train_df, test_df), one row per application."""
    train, test, tx, pay = _load()
    return _history_features(train, tx, pay), _history_features(test, tx, pay)


def build_dataset_naive():
    """THE MISTAKE: aggregates each client's whole history, including after the application."""
    train, test, tx, pay = _load()

    def naive(apps):
        p = pay.groupby(CLIENT)["days_past_due"].agg(["mean", "max", "sum"]).add_prefix("pay_all_").reset_index()
        t = tx.groupby(CLIENT)["amount"].agg(["count", "sum", "mean"]).add_prefix("tx_all_").reset_index()
        out = safe_merge(apps, p, on=CLIENT)
        out = safe_merge(out, t, on=CLIENT)
        out["loan_to_income"] = out["loan_amount"] / out["income"]
        return out

    return naive(train), naive(test)
