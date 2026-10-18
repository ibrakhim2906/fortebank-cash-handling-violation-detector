# Guide: validation

Validation decides whether every later number is real or fiction. Set it once, review it, freeze it.

## Pick the split that mimics how the bank uses the model
| Situation | Split | Config |
|---|---|---|
| Rows have a date and the model predicts the future | Time-based: train on earlier, validate on later | `TIME_COL = "<date col>"` |
| One client/entity has several rows | Group K-fold by client | `GROUP_COL = "<client id>"` |
| Independent rows, no time | Stratified K-fold (binary) / K-fold | both `None` |

Never shuffle time-ordered data. For a client-disjoint evaluation, explicitly enable validation-group exclusion in time mode; setting `GROUP_COL` alone does not exclude clients in that mode. Review whether client separation matches the bank's intended population before freezing the protocol.

## Time-mode options

`make_folds(df)` remains the development-fold entry point for `src.train` and `src.sanity`. `score()` and `cross_validate()` are unchanged. Time splits use calendar dates (timestamps are normalized to midnight), return positional indices sorted by date, and never divide one calendar date between training and validation within a fold.

The new settings default to off/automatic; no case-specific dates or label horizon are enabled in `src/config.py`:

| Setting | Default | Behavior in time mode |
|---|---|---|
| `LABEL_GAP_DAYS` | `0` | Training dates must satisfy `date + gap < validation start`. Zero disables the additional maturity gap but still requires strictly earlier dates. Uses fixed calendar days, not calendar months. |
| `VALIDATION_WINDOWS` | `None` | Automatically split sorted unique development dates into `N_FOLDS + 1` approximately equal chunks. The first is training-only; each later chunk is validated using eligible earlier dates. Row counts can differ substantially between chunks. |
| `HOLDOUT_WINDOW` | `None` | Optional `(start, end)` interval, start inclusive/end exclusive. Its dates are removed from both training and validation in every development fold, including later folds if the interval is in the middle of the data. |
| `EXCLUDE_VALIDATION_GROUPS` | `False` | If enabled, remove training rows whose `GROUP_COL` appears in the current validation fold. Requires a configured, non-missing group column. Otherwise `GROUP_COL` is optional and overlap is allowed in time mode. |

Explicit `VALIDATION_WINDOWS` are a non-empty sequence of `(start, exclusive_end)` pairs, in chronological order without overlap. They determine the fold count instead of `N_FOLDS`. Reserved holdout dates are removed even if an explicit development interval overlaps them. Automatic chunking excludes holdout dates before splitting unique dates. Windows with no remaining validation rows or no eligible training rows raise `ValueError`; a large label gap can therefore require later explicit windows instead of automatic ones.

Call `make_folds(df, holdout=True)` only for final evaluation after model, calibration and threshold choices are frozen. It returns exactly the configured holdout fold, with the same gap and optional group exclusion applied to its earlier training data. It requires `HOLDOUT_WINDOW` and `TIME_COL`; no holdout is silently selected. Prefer a final chronological interval for a future-use estimate; the splitter excludes a middle holdout interval but does not prohibit development dates after it.

Missing dates, negative/non-integer gaps, insufficient unique dates for automatic folds, invalid/overlapping explicit windows and empty folds fail with an error. With `TIME_COL = None`, the existing grouped/stratified/K-fold behavior remains unchanged; gap, explicit time windows and time-mode group-exclusion settings do not change those branches. A configured holdout without a time column raises an error because its dates cannot be reserved.

Choose any nonzero label gap from the actual outcome horizon and reporting availability. A fixed 183-day illustration is not identical to six calendar months after issue, and no such assumption is enabled by default. The mock-specific configuration and read-only fold tables live only in `tests/test_eval.py`; tests never fit models. Raw mock data is optional and its two integration tests skip when the ignored CSV is absent. Run `uv run pytest` to see the printed mock fold tables when the CSV is present.

## Explicit finalization

`uv run python -m src.train` defaults to development-only. Supply `--run-dir <path>` (or set `DATATHON_RUN_DIR`) to isolate artifacts; an optional `config.json` in that directory overrides case settings at runtime without editing generic defaults. A run directory is immutable after development: changed source, inputs, settings or artifacts require a fresh run.

For binary tasks, run `uv run python -m src.impact --run-dir <path> --oof lgbm` to freeze the threshold from development OOF. Then `uv run python -m src.train --run-dir <path> --stage finalize` fits only eligible earlier training rows, scores the configured holdout with fixed iterations/policy, records `holdout.json`, and only then refits on all training labels for the retrospective submission. Repeated finalization is rejected. Without `HOLDOUT_WINDOW`, the report explicitly states that no holdout was evaluated. Regression has no binary threshold stage.

Sanity scans exclude reserved holdout labels. Reports distinguish development policy selection from untouched-holdout evaluation. No claim of expected loss, calibrated probabilities or causal savings follows merely from ROC-AUC or a synthetic TP/FP value scenario.

## Checks before freezing `eval.py`
1. Metric in `score()` is exactly the official one (same definition, same averaging).
2. Fold sizes and target rate per fold look reasonable (printed by `src.train`).
3. If a separate test set exists, run adversarial validation (`uv run python -m src.sanity`). AUC near 0.5 = train and test look alike. AUC > 0.7 = they differ; find out which features separate them and treat those as dangerous.

## During the event
- Compare experiments ONLY through `eval.py`. A score from another split is not comparable.
- After ~30 experiments the CV score itself becomes overfit. Keep the final claim honest: report the CV score of the chosen setup, not the best single lucky run.
- If there is a public leaderboard: trust CV when CV and leaderboard disagree, unless the leaderboard is large and clearly more representative.
