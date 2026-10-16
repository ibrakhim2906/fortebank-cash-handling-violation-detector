# Guide: sanity checks (run after every feature change)

Command: `uv run python -m src.sanity` → writes `reports/sanity.md`.

| Check | What it catches | Pass condition |
|---|---|---|
| Duplicate IDs | Same entity counted twice, leaking across folds | 0 duplicates, or GROUP_COL is set |
| Single-feature scan | A column that alone predicts the target (leak) | No feature with AUC > 0.90 on its own |
| Shuffled-target test | Pipeline leaking target information | Score at chance level (AUC ≈ 0.5) |
| Adversarial validation | Train and test come from different distributions | AUC < 0.7 (if test exists) |

## Also check by hand
- Row counts after every merge (`safe_merge` asserts this).
- Top 20 features: would each one exist at the decision moment? If you can't explain why a feature predicts, investigate before trusting it.
- Score suspiciously high for the problem (e.g. AUC > 0.95 for credit default) → assume leakage until proven otherwise.
- Predictions on test have a similar distribution to out-of-fold predictions.

If a check fails: STOP, report it in `docs/decisions.md`, fix before any further experiments.
