# Guide: demo and README

## Demo (`app/app.py`, run with `uv run streamlit run app/app.py`)
One clear flow, nothing else:
1. Pick a client (or upload a row)
2. Show the prediction and the decision at the chosen threshold
3. Show the top 5 reasons in plain words
4. Show the business-impact chart

Rules: it must start in under 10 seconds, never crash on the demo clients, and use pre-trained artifacts from `models/` (no training at startup). Pick 3 demo clients in advance (high risk, low risk, borderline) and rehearse with exactly those. Record a 1-minute backup video.

Select a finalized run with `DATATHON_RUN_DIR=<path> uv run streamlit run app/app.py`; preserve any `DATATHON_THREADS` setting used for that run. The app loads its manifest-checked LightGBM artifact, selects low/high/nearest-threshold examples from that model, and displays the same policy and impact chart. The threshold slider explores a review scenario without modifying the frozen holdout policy. Source/input/config changes require a new run. Automated Streamlit checks do not substitute for a visible-browser check, human pitch rehearsal or backup video.

## README (judges and mentors read it)
Sections, short:
1. Problem in one sentence + the bank decision it supports
2. Approach (data → features → model → explanation → threshold)
3. Results table (from `experiments.md`): baseline vs final, CV setup
4. Key insights (3 bullets with numbers)
5. Business impact (headline + assumptions)
6. How to run — exact commands, tested from a fresh clone
7. Project structure
8. Limitations and next steps (honest, short)
