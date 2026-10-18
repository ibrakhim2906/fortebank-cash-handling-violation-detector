"""Global drivers + per-row reasons from the final LightGBM (exact TreeSHAP via pred_contrib).

Usage: uv run python -m src.explain   (after uv run python -m src.train with lgbm)
"""
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src import artifacts as A
from src.features import build_dataset

# Map raw feature names to plain words for slides and the app (fill in during the event).
HUMAN_NAMES = {"late_payments_12m": "Late payments in the past year", "utilisation": "Credit utilisation",
               "income": "Income", "loan_amount": "Requested loan amount", "age": "Age",
               "region": "Region", "product": "Loan product"}


def human(name):
    return HUMAN_NAMES.get(name, name.replace("_", " "))


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "missing" if np.isnan(v) else f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def load_bundle():
    return A.load_bundle()


def contributions(bundle, X):
    """Per-row feature contributions (log-odds for binary). Last column of LightGBM output is the bias."""
    cols = bundle["columns"]
    contrib = bundle["model"].booster_.predict(A.align_frame(X, bundle["schema"]), pred_contrib=True)
    return pd.DataFrame(np.asarray(contrib)[:, :-1], columns=cols, index=X.index)


def predict(bundle, X):
    return A.predict(bundle, X)


def reasons(bundle, X_row, k=5):
    """Top-k reasons for one row as a DataFrame (feature, value, effect)."""
    c = contributions(bundle, X_row).iloc[0]
    top = c.reindex(c.abs().sort_values(ascending=False).index).head(k)
    word = "raises" if C.TASK == "binary" else "increases"
    low = "lowers" if C.TASK == "binary" else "decreases"
    return pd.DataFrame({
        "feature": [human(f) for f in top.index],
        "value": [_fmt(X_row.iloc[0][f]) for f in top.index],
        "effect": [f"{word} score" if v > 0 else f"{low} score" for v in top.values],
        "contribution": np.round(top.values, 3),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    A.configure_run(args.run_dir)
    bundle = load_bundle()
    train, test = build_dataset()
    train = (test if test is not None else train).reset_index(drop=True)
    X = A.inputs(train)
    sample = X.sample(min(len(X), 5000), random_state=C.SEED)

    imp = contributions(bundle, sample).abs().mean().sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([human(f) for f in imp.index], imp.values, color="#5b4bdb")
    ax.set_xlabel("Mean |contribution|")
    ax.set_title("What drives the model")
    fig.tight_layout()
    fig.savefig(C.REPORTS / "feature_importance.png", dpi=150)

    p = pd.Series(predict(bundle, sample), index=sample.index).sort_values()
    threshold = A.load_manifest()["policy"]["threshold"]
    borderline = (p - threshold).abs().idxmin() if threshold is not None else p.index[len(p) // 2]
    picks = {"Highest score": p.index[-1], "Lowest score": p.index[0], "Nearest policy threshold": borderline}
    out = ["# Example explanations", ""]
    for label, idx in picks.items():
        row = sample.loc[[idx]]
        ident = train.loc[idx, C.ID_COL] if C.ID_COL in train.columns else idx
        out += [f"## {label} — {C.ID_COL}={ident}, prediction={p[idx]:.3f}", "",
                reasons(bundle, row).to_markdown(index=False), ""]
    (C.REPORTS / "explanations.md").write_text("\n".join(out), encoding="utf-8")
    manifest = A.load_manifest()
    for filename in ("explanations.md", "feature_importance.png"):
        A.record_artifact(manifest, C.REPORTS / filename)
    A.save_manifest(manifest)
    print("\n".join(out))
    print(f"saved {C.REPORTS / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
