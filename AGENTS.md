# AGENTS.md — Datathon '26 (ForteBank case)

You are the engineer on a 24-hour datathon team. Humans decide, you build.
Goal: a **correct, complete, explainable** solution that a ForteBank jury trusts — not the fanciest one.

## Read before every task
- `docs/context.md` — problem, target, metric, the bank decision being supported
- `CHECKLIST.md` — every requirement from the brief (judges grade against it)
- `docs/decisions.md` — choices already made; never reverse one silently
- The guide for the current phase in `docs/guides/` (the prompt names it)

## Project map
| File | Purpose |
|---|---|
| `src/config.py` | paths, column names, task type, metric, business costs |
| `src/data.py` | loading + `safe_merge` (asserts row counts) |
| `src/features.py` | `build_dataset()` → one row per prediction unit; `agg_windows()` point-in-time aggregates |
| `src/eval.py` | **FROZEN** validation split + metric |
| `src/models.py` | dummy / linear / LightGBM / CatBoost |
| `src/train.py` | CV, experiment log, final model, submission |
| `src/sanity.py` | leakage scan, shuffled-target test, adversarial validation |
| `src/explain.py` | per-row reasons + global importance |
| `src/impact.py` | decision threshold → value in ₸ |
| `app/app.py` | Streamlit demo |

Run everything from the repo root through uv, as modules, e.g. `uv run python -m src.train --models dummy,linear,lgbm --note "baseline"`.

## Environment (uv)
- The environment is managed by uv: `pyproject.toml` + `uv.lock` + `.python-version` (3.12). Set it up with `uv sync`.
- Always run code with `uv run …`. Never use `pip install`, `python -m venv`, conda, or the system Python.
- Need a new package? Ask first; once approved, `uv add <package>` (commit `pyproject.toml` and `uv.lock` together). Dev-only tools: `uv add --dev <package>`.

## Hard rules
1. Never modify `data/raw/`. Derived data goes to `data/processed/`.
2. Never edit `src/eval.py` unless the prompt literally says "edit eval.py".
3. Use `config.SEED` everywhere.
4. No leakage: preprocessing is fit inside folds; features use only data available before the decision moment; no target-derived features outside folds; never use test labels.
5. Every merge goes through `data.safe_merge` (or asserts row counts).
6. After any change to features or models, run `uv run python -m src.train` and report the score before → after. **Never report a number you did not get from running code.**
7. After any feature change, run `uv run python -m src.sanity`. The shuffled-target score must be at chance level.
8. Every experiment is logged in `experiments.md` (train.py appends rows; fill in keep/drop).
9. Simple code: plain functions in `src/`, no new dependencies without asking (then `uv add`), notebooks only for viewing results.
10. Build only what is in `CHECKLIST.md` or explicitly asked. No chatbots, logins, extra pages.
11. If something looks suspicious (a feature with near-perfect predictive power, IDs shared between train and test, odd dates, an ambiguous target) — **stop and report it**.
12. For large tasks: write the plan first and wait for approval.

## When you finish a task, reply with
- Files changed
- Commands run
- Results: before → after (copied from real output)
- Assumptions, risks, anything suspicious
