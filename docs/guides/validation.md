# Guide: validation

Validation decides whether every later number is real or fiction. Set it once, review it, freeze it.

## Pick the split that mimics how the bank uses the model
| Situation | Split | Config |
|---|---|---|
| Rows have a date and the model predicts the future | Time-based: train on earlier, validate on later | `TIME_COL = "<date col>"` |
| One client/entity has several rows | Group K-fold by client | `GROUP_COL = "<client id>"` |
| Independent rows, no time | Stratified K-fold (binary) / K-fold | both `None` |

Never shuffle time-ordered data. Never let the same client appear in train and validation folds.

## Checks before freezing `eval.py`
1. Metric in `score()` is exactly the official one (same definition, same averaging).
2. Fold sizes and target rate per fold look reasonable (printed by `src.train`).
3. If a separate test set exists, run adversarial validation (`uv run python -m src.sanity`). AUC near 0.5 = train and test look alike. AUC > 0.7 = they differ; find out which features separate them and treat those as dangerous.

## During the event
- Compare experiments ONLY through `eval.py`. A score from another split is not comparable.
- After ~30 experiments the CV score itself becomes overfit. Keep the final claim honest: report the CV score of the chosen setup, not the best single lucky run.
- If there is a public leaderboard: trust CV when CV and leaderboard disagree, unless the leaderboard is large and clearly more representative.
