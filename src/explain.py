"""Global drivers + per-row reasons from the final LightGBM (exact TreeSHAP via pred_contrib).

Usage: uv run python -m src.explain   (after uv run python -m src.train with lgbm)
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.features import build_dataset, to_model_frame

# Map raw feature names to plain words for slides and the app (fill in during the event).
HUMAN_NAMES = {}


def human(name):
    return HUMAN_NAMES.get(name, name)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "missing" if np.isnan(v) else f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def load_bundle():
    return joblib.load(C.MODELS / "lgbm.joblib")


def contributions(bundle, X):
    """Per-row feature contributions (log-odds for binary). Last column of LightGBM output is the bias."""
    cols = bundle["columns"]
    contrib = bundle["model"].booster_.predict(X[cols], pred_contrib=True)
    return pd.DataFrame(np.asarray(contrib)[:, :-1], columns=cols, index=X.index)


def predict(bundle, X):
    m = bundle["model"]
    cols = bundle["columns"]
    return m.predict_proba(X[cols])[:, 1] if C.TASK == "binary" else m.predict(X[cols])


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
    bundle = load_bundle()
    train, _ = build_dataset()
    train = train.reset_index(drop=True)
    X, _ = to_model_frame(train)
    sample = X.sample(min(len(X), 5000), random_state=C.SEED)

    imp = contributions(bundle, sample).abs().mean().sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([human(f) for f in imp.index], imp.values, color="#5b4bdb")
    ax.set_xlabel("Mean |contribution|")
    ax.set_title("What drives the model")
    fig.tight_layout()
    fig.savefig(C.REPORTS / "feature_importance.png", dpi=150)

    p = pd.Series(predict(bundle, sample), index=sample.index).sort_values()
    picks = {"Highest score": p.index[-1], "Lowest score": p.index[0], "Borderline (median)": p.index[len(p) // 2]}
    out = ["# Example explanations", ""]
    for label, idx in picks.items():
        row = sample.loc[[idx]]
        ident = train.loc[idx, C.ID_COL] if C.ID_COL in train.columns else idx
        out += [f"## {label} — {C.ID_COL}={ident}, prediction={p[idx]:.3f}", "",
                reasons(bundle, row).to_markdown(index=False), ""]
    (C.REPORTS / "explanations.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"saved {C.REPORTS / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
