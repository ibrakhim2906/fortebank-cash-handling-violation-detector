# Datathon '26 — Game Plan

**Core rule:** Codex writes the code. The team owns the evaluation, the insight and the story.
Operator guide: `START_HERE.md` · prompts P1–P13 + PS: `PLAYBOOK.md` · deck structure: `pitch/deck_outline.md` · visual template: "Datathon ’26 Pitch Template" in claude.ai artifacts.

## Morning (Oct 17)
- `sudo apt install -y bubblewrap` (Codex sandbox) and `uv run pytest` (should pass, 2 skipped)
- Check-in **09:00, Orange Hall** — laptop, charger, extension cord
- **11:00 kickoff:** ask organizers whether prepared code and AI tools are allowed. Download the data.
- `mkdir -p data/raw` → put data in `data/raw/`, paste the full brief into `docs/brief.md`

## Timeline
| Time | Phase | Codex | Humans |
|---|---|---|---|
| 12:00–13:00 | Understand | **P1**: context, checklist, data overview, plan — no models | Ask ForteBank mentors the `docs/jury.md` questions. Approve the plan. Log decisions in `docs/decisions.md`. |
| 13:00–15:00 | Baseline | **P2**: config + dataset + baseline. **P3**: leakage audit | Set validation options (below). Read `src/eval.py`, then freeze it: `git commit -m "freeze eval"`. |
| 15:00–00:00 | Improve | **P4**: one feature family per branch, 2–3 in parallel. **P5**: model comparison. **P6: insight hunt** | Check every number against `experiments.md`. Pick the strongest insight. Deck owner starts slides 2–3 by ~18:00. |
| 00:00 | Feature freeze | — | Sleep in shifts |
| 00:00–04:00 | Explain + money | **P7**: reasons in plain words. **P8**: ₸ impact with ForteBank's costs | Check drivers make business sense |
| 04:00–08:00 | Demo + README | **P9**: demo. **P10**: README | Pick 3 demo clients (high / low / borderline). Record a backup video. |
| 08:00–12:00 | Pitch + freeze | **P11**: slide content. **P12**: Q&A prep. **P13**: fresh-clone check | Fill the deck, cut to 5 slides for a 2-minute slot, rehearse 5+ times with a timer |
| ~10:00 | Code freeze | — | Submit, check the submission opens |
| **Oct 18, 13:00** | **Pitch** | — | Best speaker presents |

## Validation options to set in P2 (eval v2, in `src/config.py`)
Decide from the real brief, write the choice in `docs/decisions.md`, then let Codex set them:
- `TIME_COL` — the decision date (application / report date). Folds always cut on date boundaries.
- `LABEL_GAP_DAYS` — length of the target window (e.g. "default within 6 months" → ~183). Training rows must have a known outcome before validation starts. 0 = off.
- `GROUP_COL` + `EXCLUDE_VALIDATION_GROUPS` — set when one client has several rows. Excluding validation clients from training is conservative; keep it off if many test rows are returning clients.
- `HOLDOUT_WINDOW` — last month(s) kept untouched until the end; check once with `make_folds(df, holdout=True)`.
- Fewer validation rows = noisier score: treat AUC differences under ~0.01 as noise.

## Roles
- **Driver:** runs Codex, reviews diffs, merges to main
- **Verifier:** owns `src/eval.py` and `src.sanity`, checks every number
- **Analyst / pitch:** ForteBank mentors, the insight, the ₸ case, the deck
- **Demo:** app, README, backup video

## What wins
1. ForteBank's own words — build the pitch on what mentors say they need
2. One insight others missed, with a number and a chart
3. Honest validation — "we caught a leak / a trap" is a strong line
4. ₸ value above the no-model baseline, with stated assumptions
5. Every brief requirement visibly ticked
6. A rehearsed 2-minute pitch

## The 5 human checks
1. `src/eval.py` mimics real use and is frozen
2. Top features exist at the decision moment
3. Every slide number traces to `experiments.md` or `reports/`
4. Every `CHECKLIST.md` item is shown in the pitch
5. The pitch fits the time limit

## If things go wrong
- **Codex usage limit:** switch reasoning High → Medium, or use a backup account
- **Score looks too good** (AUC > 0.95 on bank data): assume leakage, run `uv run python -m src.sanity`
- **Case isn't tabular:** P1 sends Codex to `docs/guides/problem_types.md` — plan first, then build
- **Codex says "sandbox failed / bubblewrap missing":** `sudo apt install -y bubblewrap`, restart the app
- **Demo breaks on stage:** play the backup video
- **A teammate needs the code:** push to a private GitHub repo; everyone pulls from main
