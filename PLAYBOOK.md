# Playbook — the phases P1–P13 (+ PS)

Timeline: check-in 09:00 Orange Hall · kickoff 11:00 · clock starts 12:00 · pitch 13:00 Oct 18.
Team rule: **Codex writes the code. Humans own the evaluation and the story.**

How to use: the human says `run P<n>` (optionally `with: <details>`). Codex reads that prompt here and follows it exactly, then stops and reports. The prompts can also be pasted in full. Operator guide: `START_HERE.md`.

---

## PHASE 1 · Ingest the case (12:00–13:00)
Inputs: full case text in `docs/brief.md`, data files in `data/raw/`, mentor answers in `docs/jury.md` (may be added after P1).

> **P1 — Ingest (plan only, no modelling)**
> Read AGENTS.md, docs/brief.md, docs/jury.md (if filled), every guide in docs/guides/, and inspect every file in data/raw/ (shape, columns, dtypes, a few sample rows). If the case is not tabular classification/regression, use docs/guides/problem_types.md. Do not train models. Produce:
> 1. `docs/context.md` fully filled in; mark anything uncertain with ❓ and list it under "Open questions".
> 2. `CHECKLIST.md` "From the brief" section: every requirement, deliverable, evaluation criterion and format rule, quoting the brief's wording.
> 3. `reports/data_overview.md`: per table shape, grain, keys and join relationships, dtypes, missing %, cardinality, target distribution, time ranges, and a list of columns that could leak the target or would not exist at the decision moment.
> 4. `docs/plan.md`: the prediction unit, the validation split and why, the first 3 feature families to build, models (see docs/guides/modeling.md or problem_types.md), what the demo will show, and how we will measure business value. Include time estimates.
> Then stop and wait for approval.

**Humans (while Codex works):** ask organizers if AI tools and prepared code are allowed. Talk to ForteBank mentors using the questions in `docs/jury.md`. Then review Codex's files, fix them, and write the key choices into `docs/decisions.md`.

---

## PHASE 2 · Wire up and baseline (13:00–15:00)

> **P2 — Configure and build the dataset**
> Following docs/plan.md and docs/decisions.md:
> 1. Set `src/config.py`: files, ID_COL, TARGET, TIME_COL, GROUP_COL, DROP_COLS, TASK, METRIC.
> 2. Set the validation options (read docs/guides/validation.md): `LABEL_GAP_DAYS` from the target window in the brief (0 if outcomes are known immediately), `EXCLUDE_VALIDATION_GROUPS`, `HOLDOUT_WINDOW`, `VALIDATION_WINDOWS` (or None for automatic). Justify each value in one line and append it to docs/decisions.md.
> 3. Adapt `build_dataset()` in `src/features.py` so it returns one row per prediction unit for train and test. For multi-table data, follow the tested pattern in `examples/multitable/features_multitable.py` (per-row cutoff with `agg_windows(..., group_by=...)`, joins with `safe_merge`); keep it minimal for now.
> 4. Print the fold table: per fold the train/validation date ranges, row counts, target rate, and shared groups.
> 5. Run `uv run python -m src.train --models dummy,linear,lgbm --note "baseline"` and `uv run python -m src.sanity`, and report the real output.
> Do not edit `src/eval.py`. If eval.py cannot express what this case needs, explain the change and stop.

**Humans:** read `src/eval.py`, the fold table and the validation choices against `docs/context.md`. Realistic split? Exact official metric? Then commit: `git commit -am "freeze eval"`.

> **P3 — Leakage audit**
> Act as a skeptical reviewer trying to prove our baseline score is fake. Read docs/guides/sanity.md. Audit src/ and the top 20 features for leakage, information from after the decision moment, target-derived features, duplicate entities across folds, and train/test differences. Report each suspicion with evidence. Do not change code.

---

## PHASE 3 · Improve (15:00–24:00) — run 2–3 tasks in parallel, one branch each

> **P4 — Feature family (repeat per idea)**
> Read docs/guides/features.md. On a new branch `feat-<name>`, add ONE feature family: <describe it>. Respect point-in-time rules and add a test that the features don't change when events after the cutoff are deleted. Run `uv run python -m src.train --models lgbm --note "<name>"` and `uv run python -m src.sanity`. Report before → after vs the best row in experiments.md, and say keep or drop.

Good feature families to queue: window aggregates per history table · trends (30d vs previous 60d) · recency · ratios · delinquency stats · behaviour shares · first/last records.

> **P5 — Model comparison (after a few feature wins)**
> Read docs/guides/modeling.md. Run `uv run python -m src.train --models dummy,linear,lgbm,cat --note "model comparison"`. Report a table of all models and the blend. Recommend one final setup and explain why.

> **P6 — Insight hunt (the most important task for winning)**
> Read docs/guides/explain_impact.md and docs/jury.md. Find 3 non-obvious, business-relevant findings that a ForteBank manager would act on: segments behaving differently, where the model fails, data-quality issues, surprising drivers. For each: one sentence, one number computed from the data, one chart saved in reports/insights/. Rank them by how surprising and actionable they are. Write `reports/insights.md`.

Before sleeping: start long-running feature/model tasks, review results in the morning.
**Feature freeze ~00:00.** After that, depth only.

---

## PHASE 4 · Explain and money (00:00–04:00)

> **P7 — Explanations**
> Read docs/guides/explain_impact.md. Run `uv run python -m src.explain`. Rewrite the reasons in `reports/explanations.md` in plain business language, and add a mapping table from raw feature names to human names (also use it in the app).

> **P8 — Business impact**
> Using the costs we agreed in docs/decisions.md, set VALUE_TP and COST_FP in src/config.py and run `uv run python -m src.impact`. Write the headline sentence for the slide and a sensitivity table (value at ±50% costs). Report the real output.

---

## PHASE 5 · Demo and README (04:00–08:00)

> **P9 — Demo**
> Read docs/guides/demo_readme.md. Polish `app/app.py`: human feature names, the 3 demo clients preselected, the decision at the chosen threshold, the impact chart. One flow only. Make sure it runs with `uv run streamlit run app/app.py` from a fresh terminal.

> **P10 — README**
> Read docs/guides/demo_readme.md. Write README.md for judges using only numbers from experiments.md and reports/. Test every command in it.

**Humans:** record a 1-minute backup video of the demo.

---

## PHASE 6 · Pitch and freeze (08:00–12:00)
The deck structure is in `pitch/deck_outline.md`; the visual template is the "Datathon ’26 Pitch Template" deck in claude.ai artifacts. The pitch owner should start on the insight slide by ~18:00, not at the end.

> **P11 — Pitch material**
> Read pitch/deck_outline.md, docs/guides/pitch.md, docs/jury.md, CHECKLIST.md, reports/, experiments.md. Write `pitch/deck_content.md` following the outline: the 5-slide 2-minute cut and the 3 backup slides — for each slide the headline (max 8 words), the visual (file path), short on-slide text, speaker notes. Then a 2-minute script (~280 words), a 5-minute script, and a CHECKLIST.md → slide mapping. Use only real numbers from our files, and use ForteBank's own words from docs/jury.md where possible.

> **P12 — Q&A prep**
> Act as a tough ForteBank jury (risk, IT, business). Write the 12 hardest questions about our solution with short, honest answers based on our actual results and limitations.

> **P13 — Final check**
> Clone the repo into a fresh folder, run `uv sync`, then run every README command from scratch. Report anything that fails. Go through CHECKLIST.md and confirm each item, naming the file that proves it.

**Code freeze 2 hours before the deadline.** Submit, check the submission opens, stop touching code. Rehearse with a timer.

---

## Any time

> **PS — Status check**
> Without changing code: summarize the current best setup and its score (from experiments.md), what was kept and dropped, open suspicions, which CHECKLIST.md items are done or missing, and the next 3 most valuable tasks with time estimates.

---

## The 5 human checks Codex can't do for you
1. `src/eval.py` mimics real use and is frozen.
2. The top features would exist at the decision moment.
3. Every number on a slide traces to `experiments.md` or `reports/`.
4. Every `CHECKLIST.md` item is visible in the pitch.
5. The pitch fits the time limit.
