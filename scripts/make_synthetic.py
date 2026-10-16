"""Fake credit data for tonight's dry run. Delete data/raw/* afterwards."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw"
raw.mkdir(parents=True, exist_ok=True)


def make(n, start, end, id_start):
    dates = pd.to_datetime(rng.integers(pd.Timestamp(start).value // 10**9, pd.Timestamp(end).value // 10**9, n), unit="s").normalize()
    income = rng.lognormal(12.6, 0.5, n).round(-3)
    loan = (income * rng.uniform(0.5, 6, n)).round(-3)
    late = rng.poisson(0.6, n)
    age = rng.integers(21, 65, n)
    region = rng.choice(["Almaty", "Astana", "Shymkent", "Karaganda", "Aktobe"], n)
    product = rng.choice(["cash", "pos", "card"], n, p=[0.5, 0.3, 0.2])
    util = np.clip(rng.beta(2, 3, n) + 0.05 * late, 0, 1.2)
    logit = -3 + 0.7 * late + 1.5 * util + 0.25 * np.log(loan / income) - 0.02 * (age - 40) + 0.4 * (product == "pos")
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"client_id": np.arange(id_start, id_start + n), "application_date": dates,
                         "age": age, "income": income, "loan_amount": loan, "late_payments_12m": late,
                         "utilisation": util.round(3), "region": region, "product": product, "target": target})


train = make(20000, "2025-01-01", "2026-03-31", 1)
test = make(5000, "2026-04-01", "2026-06-30", 100001).drop(columns="target")
train.to_csv(raw / "train.csv", index=False)
test.to_csv(raw / "test.csv", index=False)
print(f"wrote {raw / 'train.csv'} ({len(train)} rows, target mean {train.target.mean():.3f}) and test.csv ({len(test)} rows)")
