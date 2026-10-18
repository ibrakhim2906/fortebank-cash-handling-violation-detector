# Guide: modeling

## Ladder (always in this order)
1. `dummy` — the floor. Everything must beat it clearly.
2. `linear` — logistic / ridge regression. If LightGBM barely beats it, the features are weak or something is wrong.
3. `lgbm` — LightGBM with early stopping. Main workhorse.
4. `cat` — CatBoost. Often better with many categorical columns.
5. Blend — average of lgbm and cat (train.py does it automatically when both run). Ensembling usually adds very little (~0.002 AUC in Home Credit) — do it last, if at all.

Command: `uv run python -m src.train --models dummy,linear,lgbm,cat --note "<what changed>"`

## Rules
- Tabular data → gradient boosting. No neural nets unless the data is text or images.
- Tuning: only after features are frozen, max ~30–50 trials, judged by `eval.py` only. Gains are small; features matter more.
- Class imbalance: do NOT oversample/SMOTE. AUC-type metrics don't need it. For F1/recall-type metrics, tune the decision threshold instead.
- Calibration: if the business case multiplies probabilities by money (expected loss), check calibration and calibrate on out-of-fold predictions.
- The final model is retrained on all training data (train.py does this for LightGBM).

Training is now staged: default `src.train` is development-only. For binary tasks freeze the selected LightGBM threshold using `src.impact`, then explicitly run `src.train --stage finalize` to measure the configured holdout before final all-training fitting. Use the same run directory for every command. LightGBM is the supported delivered artifact; CatBoost/blend remain comparison outputs. A changed source/config/input requires a fresh run. Do not select the deployed model from an impact script's search for leftover files.

## Non-tabular or unusual problems
Read `problem_types.md` first and propose a plan before coding.
