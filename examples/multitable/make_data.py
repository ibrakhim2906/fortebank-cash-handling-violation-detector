"""Fake bank data with the shape real cases usually have: applications + long history tables.

Writes into examples/multitable/data/ (NOT data/raw):
  applications_train.csv  application_id, client_id, application_date, age, income, loan_amount, product, region, target
  applications_test.csv   same, later dates, no target
  transactions.csv        client_id, txn_date, amount, channel        (history BEFORE and AFTER applications)
  payments.csv            client_id, payment_date, days_past_due      (history BEFORE and AFTER applications)

The trap built in: payments AFTER the application date are strongly tied to the target
(defaulters stop paying). A naive "aggregate all history" feature leaks the answer.
Some clients apply twice, so features must be computed per application, not per client.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

N_CLIENTS = 12000
HIST_START, HIST_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2026-09-30")


def rand_dates(n, start, end):
    s, e = start.value // 10**9, end.value // 10**9
    return pd.to_datetime(rng.integers(s, e, n), unit="s").normalize()


clients = pd.DataFrame({"client_id": np.arange(1, N_CLIENTS + 1)})
clients["risk"] = rng.normal(0, 1, N_CLIENTS)
clients["age"] = rng.integers(21, 65, N_CLIENTS)
clients["income"] = rng.lognormal(12.6, 0.5, N_CLIENTS).round(-3)
clients["region"] = rng.choice(["Almaty", "Astana", "Shymkent", "Karaganda", "Aktobe"], N_CLIENTS)

# applications: everyone applies once, 20% apply twice
apps = pd.concat([clients, clients.sample(frac=0.2, random_state=1)], ignore_index=True)
apps["application_date"] = rand_dates(len(apps), pd.Timestamp("2025-03-01"), pd.Timestamp("2026-04-30"))
apps["loan_amount"] = (apps["income"] * rng.uniform(0.5, 6, len(apps))).round(-3)
apps["product"] = rng.choice(["cash", "pos", "card"], len(apps), p=[0.5, 0.3, 0.2])
logit = (-2.4 + 1.1 * apps["risk"] + 0.3 * np.log(apps["loan_amount"] / apps["income"])
         - 0.015 * (apps["age"] - 40) + 0.3 * (apps["product"] == "pos"))
apps["target"] = (rng.random(len(apps)) < 1 / (1 + np.exp(-logit))).astype(int)
apps = apps.sort_values("application_date").reset_index(drop=True)
apps["application_id"] = np.arange(500001, 500001 + len(apps))

# history tables per client
risk = clients.set_index("client_id")["risk"]
n_tx = rng.poisson(60, N_CLIENTS)
tx = pd.DataFrame({"client_id": np.repeat(clients["client_id"].values, n_tx)})
tx["txn_date"] = rand_dates(len(tx), HIST_START, HIST_END)
tx["amount"] = rng.lognormal(9.5, 1.0, len(tx)).round(-1)
tx["channel"] = rng.choice(["card", "online", "cash"], len(tx), p=[0.6, 0.3, 0.1])
# risky clients cut spending in the months before they apply (a real, allowed signal)
first_app = apps.groupby("client_id")["application_date"].min()
days_before = (tx["client_id"].map(first_app) - tx["txn_date"]).dt.days
risky = tx["client_id"].map(risk) > 0.6
tx.loc[risky & days_before.between(0, 60), "amount"] *= 0.45

months = pd.date_range(HIST_START, HIST_END, freq="MS")
pay = pd.DataFrame({"client_id": np.repeat(clients["client_id"].values, len(months)),
                    "payment_date": np.tile(months.values, N_CLIENTS)})
pay["payment_date"] = pd.to_datetime(pay["payment_date"]) + pd.to_timedelta(rng.integers(0, 28, len(pay)), unit="D")
base = np.exp(-1.6 + 0.8 * pay["client_id"].map(risk).values)
pay["days_past_due"] = rng.poisson(base * 3)
# THE TRAP: after a defaulted application, payments go badly late
defaulted = apps[apps["target"] == 1].groupby("client_id")["application_date"].min()
after_default = pay["payment_date"] > pay["client_id"].map(defaulted)
pay.loc[after_default.fillna(False), "days_past_due"] += rng.poisson(25, int(after_default.fillna(False).sum()))

cut = pd.Timestamp("2026-03-01")
cols = ["application_id", "client_id", "application_date", "age", "income", "loan_amount", "product", "region"]
apps[apps["application_date"] < cut][cols + ["target"]].to_csv(OUT / "applications_train.csv", index=False)
apps[apps["application_date"] >= cut][cols].to_csv(OUT / "applications_test.csv", index=False)
tx.to_csv(OUT / "transactions.csv", index=False)
pay.to_csv(OUT / "payments.csv", index=False)
tr = apps[apps["application_date"] < cut]
print(f"applications: train {len(tr)} (target mean {tr['target'].mean():.3f}), test {(apps['application_date'] >= cut).sum()}")
print(f"transactions {len(tx):,} rows, payments {len(pay):,} rows → {OUT}")
