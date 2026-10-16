# START HERE — how to run the datathon with this repo

You give Codex three inputs. Codex builds; you approve at checkpoints.
Timeline and roles: `GAMEPLAN.md` · every prompt: `PLAYBOOK.md` · Codex rules: `AGENTS.md`.

## 1. The three inputs (11:00–12:30)
| Input | Where | When |
|---|---|---|
| The case — full text of the brief, rules, metric, deliverables, deadlines | `docs/brief.md` | At kickoff, unedited |
| The data — every file exactly as received | `data/raw/` (`mkdir -p data/raw` first) | At kickoff |
| The jury — what ForteBank mentors said, who judges, pitch length, ₸ costs | `docs/jury.md` | After talking to mentors (can come after P1) |

Then check the environment once: `uv run pytest` (passes, 2 skipped).

## 2. Talk to Codex in short commands
Open the `datathon-kit` project in Codex (agent environment: WSL). Codex reads `AGENTS.md` automatically and knows the phases in `PLAYBOOK.md`, so you only say which one to run:

| You say | Codex does | Your checkpoint before the next step |
|---|---|---|
| `run P1` | Reads brief, data, guides; writes context, checklist, data overview, plan. No models. | Read `docs/plan.md`. Fix it. Write choices in `docs/decisions.md`. |
| `approved, run P2` | Sets config + validation options, builds the dataset, prints the fold table, trains the baseline, runs sanity | **Read `src/eval.py` + the fold table.** Then `git commit -am "freeze eval"`. |
| `run P3` | Skeptical leakage audit, no code changes | Act on every real suspicion before going on |
| `run P4 with: <feature idea>` | One feature family on its own branch, score before → after, keep/drop | Merge only "keep" branches. Run 2–3 in parallel chats. |
| `run P5` | Model comparison + recommended final setup | Agree the final setup |
| `run P6` | **Insight hunt**: 3 findings with a number and a chart each | Pick the one insight the pitch is built on |
| `run P7`, `run P8` | Plain-language reasons; ₸ impact with ForteBank's costs | Do the drivers make business sense? Are the costs from `docs/jury.md`? |
| `run P9`, `run P10` | Demo app, README | Click through the demo yourself. Record a backup video. |
| `run P11`, `run P12` | Deck content in `pitch/deck_content.md` (2-min + 5-min), tough Q&A | Paste into the deck, rehearse with a timer |
| `run P13` | Fresh-clone check, checklist proof | Submit. Code freeze. |
| `run PS` | Status: best score, open items, checklist gaps | Use at every shift change |

Reasoning level: **High** for P1, P2, P3, P6. **Medium** for everything else (faster, saves limits).

## 3. Rules that keep the solution honest
- `src/eval.py` is frozen after the P2 checkpoint. Codex may only change it if you write "edit eval.py".
- Every number on a slide must exist in `experiments.md` or `reports/`.
- AUC above ~0.95 on bank data = assume leakage until `src.sanity` and P3 say otherwise.
- One Codex chat per branch. The driver merges to `main`.
- Commit after every checkpoint (`git commit -am "<what>"`), so any mistake is one `git reset` away.

## 4. What the finished repo contains
`reports/` (data overview, sanity, explanations, insights, impact, submission) · `models/` · `app/app.py` · `README.md` · `pitch/deck_content.md` · `experiments.md` · `CHECKLIST.md` all ticked.
