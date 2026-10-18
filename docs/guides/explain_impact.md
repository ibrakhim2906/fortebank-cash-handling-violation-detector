# Guide: explanations and business impact

A bank jury needs two things beyond a score: **why** the model decides, and **what it is worth in ₸**.

Use one selected run throughout (`--run-dir <path>` or `DATATHON_RUN_DIR`). LightGBM is the delivered model; CatBoost/blend are comparison-only. Set agreed cost assumptions before starting development. After development and sanity checks, run impact to freeze the LightGBM policy, explicitly finalize/evaluate the holdout with `uv run python -m src.train --stage finalize`, then generate final explanations. The manifest rejects stale/mismatched artifacts and accidental repeat finalization. Value selected on development OOF is labeled as such; the locked policy's holdout results are in `holdout.json`.

## Explanations — `uv run python -m src.explain`
- Global: top drivers (`reports/feature_importance.png`). Rename features into plain words for slides.
- Individual: 3 example applications (highest risk, lowest risk, nearest the frozen threshold) with top 5 reasons each, in `reports/explanations.md`. Use test rows when available; no test labels are required.
- Write reasons the way a credit officer would say them: "flagged because 4 late payments in the last 90 days and utilisation above 90%". No raw column names on slides.
- Sanity: do the top drivers make business sense? If a driver is weird, it may be leakage or an insight — find out which.

## Business impact — `uv run python -m src.impact` (binary tasks)
- Set the money assumptions in `src/config.py`: `VALUE_TP` (value of catching a true positive, e.g. loss avoided) and `COST_FP` (cost of a false alarm, e.g. lost margin from rejecting a good client). Get rough numbers from ForteBank mentors (see `docs/jury.md`); otherwise state assumptions openly.
- The script picks the threshold that maximises value and writes `reports/impact.md` + `reports/impact.png`.
- Headline format for the slide: "At this threshold, ForteBank would catch X% of defaulters while flagging only Y% of clients — about N ₸ per 10,000 applications."
- Always show the assumptions on the slide and say the threshold is adjustable.

## Insight hunt (most important for winning)
Find 3 non-obvious, actionable findings: segments that behave differently, where the model fails, data-quality problems, surprising drivers. Each = one sentence + one number + one chart in `reports/`. Pick the strongest one as the centre of the pitch.
