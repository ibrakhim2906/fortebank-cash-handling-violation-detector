"""Deterministic rehearsal inputs outside immutable raw data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C


def make_frame(count, start, end, id_start, rng):
    dates = pd.to_datetime(rng.integers(pd.Timestamp(start).value // 10**9,
                                      pd.Timestamp(end).value // 10**9, count), unit="s").normalize()
    income = rng.lognormal(12.6, 0.5, count).round(-3)
    loan = (income * rng.uniform(0.5, 6, count)).round(-3)
    late = rng.poisson(0.6, count)
    age = rng.integers(21, 65, count)
    region = rng.choice(["Almaty", "Astana", "Shymkent", "Karaganda", "Aktobe"], count)
    product = rng.choice(["cash", "pos", "card"], count, p=[0.5, 0.3, 0.2])
    utilisation = np.clip(rng.beta(2, 3, count) + 0.05 * late, 0, 1.2)
    logit = -3 + 0.7 * late + 1.5 * utilisation + 0.25 * np.log(loan / income) - 0.02 * (age - 40) + 0.4 * (product == "pos")
    target = (rng.random(count) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"client_id": np.arange(id_start, id_start + count), "application_date": dates,
                         "age": age, "income": income, "loan_amount": loan, "late_payments_12m": late,
                         "utilisation": utilisation.round(3), "region": region, "product": product, "target": target})


def generate(output, train_rows=12000, test_rows=2000):
    output = Path(output).resolve()
    if output == C.DATA_RAW.resolve() or C.DATA_RAW.resolve() in output.parents:
        raise ValueError("Synthetic output must be outside data/raw")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Synthetic output directory must be empty")
    if min(train_rows, test_rows) < 1:
        raise ValueError("Fixture row counts must be positive")
    rng = np.random.default_rng(C.SEED)
    train = make_frame(train_rows, "2025-01-01", "2026-03-01", 1, rng)
    test = make_frame(test_rows, "2026-03-01", "2026-05-01", 100001, rng).drop(columns="target")
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=12000)
    parser.add_argument("--test-rows", type=int, default=2000)
    args = parser.parse_args()
    train, test = generate(args.output, args.train_rows, args.test_rows)
    print(f"SYNTHETIC seed={C.SEED} train={train.shape} test={test.shape}; target rate={train.target.mean():.5f}")


if __name__ == "__main__":
    main()
