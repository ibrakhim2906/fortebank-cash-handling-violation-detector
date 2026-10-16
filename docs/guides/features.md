# Guide: features

Features are where the score is won. In the Home Credit and AmEx default competitions, top solutions gained far more from feature engineering than from model choice.

## Step 1 — one row per prediction unit
`build_dataset()` in `src/features.py` must return a table with exactly one row per thing we predict (client, application, …). Multi-table data is aggregated into that table with `agg_windows()` and joined with `safe_merge()`.

**Worked, tested example: `examples/multitable/features_multitable.py`.** Applications + transactions + payments, clients with several applications, features computed per application with its own cutoff (`agg_windows(..., ref=..., group_by="application_id")`), ratios, and count of previous applications. Copy its pattern. `uv run python -m examples.multitable.compare` shows the result on fake data: application-only AUC 0.54 → with point-in-time history 0.73 → with naive all-history features a fake 0.94 (pure leakage, flagged by `src.sanity`).

## Step 2 — point-in-time correctness (most important rule)
Every feature uses only data from BEFORE the decision moment of that row.
- Use `agg_windows(..., ref=...)` with a per-row cutoff date when rows have different decision dates.
- Never compute features over the full history when the target window overlaps it.
- Leakage test to write for any new feature family: delete every event after the cutoff, recompute, the feature must not change.

## Bank feature patterns (try in this order)
1. **Window aggregates** over history tables (transactions, payments, balances, previous loans): count / sum / mean / std / max over 30, 90, 180, 365 days.
2. **Trends**: last 30 days vs the previous 60/90 days. Drops in activity are strong churn and default signals.
3. **Recency**: days since last transaction / payment / login / product opened.
4. **Ratios**: debt/income, payment/installment, utilisation (balance/limit), loan amount/income, spend/income.
5. **Delinquency**: count and max of days past due, share of late payments, worst status in window.
6. **Diversity and behaviour**: distinct merchant categories, share of cash withdrawals, share of night or online transactions.
7. **First / last**: first and last application, last status, time between applications.

The Home Credit 2nd-place feature set was exactly this: the same aggregates over 1/2/3-year windows applied to every history table, plus first/last records.

## Process
- Add ONE feature family per task, on its own branch. Run `uv run python -m src.train`, compare to the current best in `experiments.md`, keep only if it helps.
- Then run `uv run python -m src.sanity`.
- Don't hand-drop features by intuition; let CV decide. Do drop features that leak.
- Categorical columns: keep as pandas `category` (LightGBM/CatBoost handle them). No one-hot for trees. No target encoding outside folds.
