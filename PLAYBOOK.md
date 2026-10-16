# Playbook — copy-paste prompts for Codex

Timeline: check-in 09:00 Orange Hall · kickoff 11:00 · clock starts 12:00 · pitch 13:00 Oct 18.
Team rule: **Codex writes the code. Humans own the evaluation and the story.**

Every prompt below assumes Codex runs in the repo root, where it reads `AGENTS.md` automatically.

---

## TONIGHT — dry run (20 min), then sleep
```
uv sync
uv run python scripts/make_synthetic.py
uv run python -m src.train --models dummy,linear,lgbm,cat --note "dry run"
uv run python -m src.sanity
uv run python -m src.explain
uv run python -m src.impact
uv run streamlit run app/app.py
```
If all of that works, delete `data/raw/*`, clear the rows in `experiments.md`, delete `reports/*` and `models/*`, and reset `src/config.py` to the template. Also test one tiny Codex task so you know your login and limits work.

---

## PHASE 1 · Ingest the case (12:00–13:00)
Paste the full case text into `docs/brief.md`. Put the data files into `data/raw/`.

> **P1 — Ingest (plan only, no modelling)**
> Read AGENTS.md, docs/brief.md, every guide in docs/guides/, and inspect every file in data/raw/ (shape, columns, dtypes, a few sample rows). Do not train models. Produce:
> 1. `docs/context.md` fully filled in; mark anything uncertain with ❓ and list it under "Open questions".
> 2. `CHECKLIST.md` "From the brief" section: every requirement, deliverable, evaluation criterion and format rule, quoting the brief's wording.
> 3. `reports/data_overview.md`: per table shape, grain, keys and join relationships, dtypes, missing %, cardinality, target distribution, time ranges, and a list of columns that could leak the target or would not exist at the decision moment.
> 4. `docs/plan.md`: the prediction unit, the validation split and why, the first 3 feature families to build, models (see docs/guides/modeling.md or problem_types.md), what the demo will show, and how we will measure business value. Include time estimates.
> Then stop and wait for approval.

**Humans (while Codex works):** ask organizers if AI tools and prepared code are allowed. Talk to ForteBank mentors using the questions in `docs/jury.md`. Then review Codex's files, fix them, and write the key choices into `docs/decisions.md`.

---

## PHASE 2 · Wire up and baseline (13:00–15:00)

> **P2 — Configure and build the dataset**
> Following docs/plan.md and docs/decisions.md: set `src/config.py` (files, ID_COL, TARGET, TIME_COL, GROUP_COL, TASK, METRIC). Adapt `build_dataset()` in `src/features.py` so it returns one row per prediction unit for train and test. For multi-table data, follow the tested pattern in `examples/multitable/features_multitable.py` (per-row cutoff with `agg_windows(..., group_by=...)`, joins with `safe_merge`); keep it minimal for now. Do not edit `src/eval.py`. If the split in eval.py does not fit this case, explain what should change and stop. Then run `uv run python -m src.train --models dummy,linear,lgbm --note "baseline"` and `uv run python -m src.sanity`, and report the real output.

**Humans:** read `src/eval.py` line by line against `docs/context.md`. Correct split? Exact official metric? Then commit: `git commit -m "freeze eval"`.

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
The deck template ("Datathon ’26 Pitch Template" in claude.ai artifacts) already has the 8 slides and timed notes: replace every [bracket] with real numbers from P11. The pitch owner should start filling slides 2 and 3 by hour 6, not hour 20.

> **P11 — Pitch material**
> Read docs/guides/pitch.md, docs/jury.md, CHECKLIST.md, reports/, experiments.md. Draft the 6 slides: headline (max 8 words), the visual to use (file path), speaker notes. Write a 2-minute script (~280 words) and a 5-minute script. Show where each CHECKLIST.md item is covered. Use only real numbers from our files.

> **P12 — Q&A prep**
> Act as a tough ForteBank jury (risk, IT, business). Write the 12 hardest questions about our solution with short, honest answers based on our actual results and limitations.

> **P13 — Final check**
> Clone the repo into a fresh folder, run `uv sync`, then run every README command from scratch. Report anything that fails. Go through CHECKLIST.md and confirm each item, naming the file that proves it.

**Code freeze 2 hours before the deadline.** Submit, check the submission opens, stop touching code. Rehearse with a timer.

---

## The 5 human checks Codex can't do for you
1. `src/eval.py` mimics real use and is frozen.
2. The top features would exist at the decision moment.
3. Every number on a slide traces to `experiments.md` or `reports/`.
4. Every `CHECKLIST.md` item is visible in the pitch.
5. The pitch fits the time limit.
