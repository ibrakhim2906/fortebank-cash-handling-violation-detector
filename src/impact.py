"""Turn out-of-fold predictions into a decision threshold and value in ₸ (binary tasks).

Usage: uv run python -m src.impact [--oof blend|lgbm|cat]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as C
from src import artifacts as A


def value_curve(y, p, thresholds):
    rows = []
    for t in thresholds:
        flag = p >= t
        tp = int(np.sum(flag & (y == 1)))
        fp = int(np.sum(flag & (y == 0)))
        rows.append({"threshold": float(t), "tp": tp, "fp": fp, "flag_rate": float(flag.mean()),
                     "recall": tp / max(1, int(np.sum(y == 1))),
                     "value": tp * C.VALUE_TP - fp * C.COST_FP})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="lgbm", choices=["lgbm"])
    ap.add_argument("--run-dir")
    args = ap.parse_args()
    A.configure_run(args.run_dir)
    if C.TASK != "binary":
        print("impact.py covers binary tasks; for regression report error in ₸ instead.")
        return
    manifest = A.load_manifest()
    if manifest["state"] != "developed":
        raise ValueError("Policy already frozen; use a fresh run for another policy")
    name = manifest["selected_model"]
    p = np.load(C.REPORTS / f"oof_{name}.npy")
    y = np.load(C.REPORTS / "oof_target.npy")
    mask = ~np.isnan(p)
    p, y = p[mask], y[mask]

    rows = value_curve(y, p, np.linspace(0.01, 0.99, 99))
    best = max(rows, key=lambda r: r["value"])
    per10k = best["value"] / len(y) * 10_000
    # Honest comparison: what could the bank get WITHOUT a model? (flag nobody = 0, or flag everyone)
    flag_all = int(np.sum(y == 1)) * C.VALUE_TP - int(np.sum(y == 0)) * C.COST_FP
    no_model = max(0, flag_all)
    uplift10k = (best["value"] - no_model) / len(y) * 10_000

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["threshold"] for r in rows], [r["value"] / len(y) * 10_000 / 1e6 for r in rows], color="#5b4bdb")
    ax.axvline(best["threshold"], linestyle="--", color="#999")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Value per 10,000 clients, million ₸")
    ax.set_title(f"Business value vs threshold (best at {best['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(C.REPORTS / "impact.png", dpi=150)

    headline = (f"At threshold {best['threshold']:.2f} the model flags {best['flag_rate']:.1%} of clients and "
                f"catches {best['recall']:.1%} of positives — about {uplift10k / 1e6:,.1f} million ₸ per 10,000 "
                f"clients MORE than the best policy without a model.")
    md = ["# Business impact — development-selected scenario", "",
          "Threshold selected on these development rows; this is not untouched-holdout value.", "",
          f"Predictions: out-of-fold `{name}` ({len(y)} rows); run `{manifest['run_id']}`", "",
          f"Assumptions: value of a caught positive = {C.VALUE_TP:,} ₸, cost of a false alarm = {C.COST_FP:,} ₸ "
          "(edit in src/config.py)", "", f"**{headline}**", "",
          f"- Value with model: {per10k / 1e6:,.1f} M ₸ per 10k clients",
          f"- Best policy without a model ({'flag everyone' if flag_all > 0 else 'flag nobody'}): "
          f"{no_model / len(y) * 10_000 / 1e6:,.1f} M ₸ per 10k clients",
          "- If 'flag everyone' wins without a model, the cost assumptions are probably wrong — recheck them with mentors.",
          "",
          "| threshold | flag rate | recall | value per 10k clients (₸) |", "|---|---|---|---|"]
    for r in rows[::10] + [best]:
        md.append(f"| {r['threshold']:.2f} | {r['flag_rate']:.1%} | {r['recall']:.1%} | {r['value'] / len(y) * 10_000:,.0f} |")
    (C.REPORTS / "impact.md").write_text("\n".join(md), encoding="utf-8")
    policy = {"threshold": best["threshold"], "model": name, "run_id": manifest["run_id"],
              "value_tp": C.VALUE_TP, "cost_fp": C.COST_FP, "selection": "development OOF only"}
    A.write_json(C.REPORTS / "impact.json", policy)
    manifest["policy"] = policy
    manifest["state"] = "policy_frozen"
    for filename in ("impact.json", "impact.md", "impact.png"):
        A.record_artifact(manifest, C.REPORTS / filename)
    sensitivity = []
    selected_flag = p >= best["threshold"]
    true_positives = int(np.sum(selected_flag & (y == 1)))
    false_positives = int(np.sum(selected_flag & (y == 0)))
    for multiplier in (0.5, 1.0, 1.5):
        sensitivity.append({"false_alarm_cost_multiplier": multiplier, "threshold": best["threshold"],
                            "value_per_10000": (true_positives * C.VALUE_TP - false_positives * C.COST_FP * multiplier)
                            * 10000 / len(y)})
    A.write_json(C.REPORTS / "sensitivity.json", sensitivity)
    A.record_artifact(manifest, C.REPORTS / "sensitivity.json")
    A.freeze_policy(manifest)
    A.save_manifest(manifest)
    print("\n".join(md))


if __name__ == "__main__":
    main()
