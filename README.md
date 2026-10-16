# Datathon '26 kit (team-only — replace with the solution README tomorrow)

A prepared repo for a 24-hour datathon where Codex writes most of the code under our direction.

## What's inside
| Path | Purpose |
|---|---|
| `AGENTS.md` | Standing rules Codex reads on every task |
| `PLAYBOOK.md` | Copy-paste prompts, phase by phase |
| `CHECKLIST.md` | Every requirement from the brief (judges grade against it) |
| `experiments.md` | Log of every result (auto-appended by train.py) |
| `docs/brief.md` | Paste the case text here at 11:00 |
| `docs/context.md` | Case context template (Codex drafts, humans confirm) |
| `docs/jury.md` | Questions for ForteBank mentors + their answers |
| `docs/decisions.md` | Decisions log so Codex stays consistent |
| `docs/guides/` | Phase guides: validation, features, modeling, sanity, explain/impact, problem types, demo/README, pitch |
| `src/` | Working starter pipeline for tabular binary classification / regression |
| `app/app.py` | Streamlit demo: client → prediction → reasons → impact |
| `scripts/make_synthetic.py` | Fake data for tonight's dry run |
| `examples/multitable/` | Tested pattern for applications + history tables, with a leakage demo |
| `tests/` | Point-in-time leakage test to copy for new feature families |

Pitch deck template: the "Datathon ’26 Pitch Template" Slides artifact in your claude.ai artifacts (8 slides with placeholders and timed speaker notes). Share it with teammates from its Share menu.

## Environment: uv
Python and every package are managed by [uv](https://docs.astral.sh/uv/). `pyproject.toml` lists the dependencies, `uv.lock` pins the exact tested versions, `.python-version` pins Python 3.12.
- Install uv once: macOS/Linux/WSL `curl -LsSf https://astral.sh/uv/install.sh | sh` · Windows PowerShell `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Create the environment: `uv sync` (downloads Python 3.12 if needed, installs the locked versions into `.venv`)
- Run anything: `uv run <command>` (no activation needed)
- Add a package (only after the team agrees): `uv add <package>`; never `pip install`

## Tonight
```
uv sync
uv run python scripts/make_synthetic.py
uv run python -m src.train --models dummy,linear,lgbm,cat --note "dry run"
uv run python -m src.sanity
uv run python -m src.explain
uv run python -m src.impact
uv run streamlit run app/app.py
```
Then clean up (see `PLAYBOOK.md`), `git init`, commit, sleep.

## Tomorrow
Follow `PLAYBOOK.md`. Ask the organizers at kickoff whether prepared code and AI tools are allowed. If prepared code is not allowed, keep only the docs and let Codex write `src/` from the guides.

## Roles
- **Driver:** runs Codex, reviews diffs, merges to main.
- **Verifier:** owns `src/eval.py` and the sanity checks, checks every number.
- **Analyst / pitch:** talks to ForteBank mentors, owns the insight, business case and deck (starts the deck by hour 6).
- **Demo:** app, README, backup video.
