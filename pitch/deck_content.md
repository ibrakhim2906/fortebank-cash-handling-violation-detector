# Pitch deck content — ForteBank cash-handling detector

Live deck (Slides artifact, private until shared): https://claude.ai/artifact/LwnP6P6VqrWGmiEiahS9HU

12 slides, about 6–8 minutes. Story: problem → three worked examples → how it works → measured proof → limits and ask.
Placeholders to fill: `[Team name]` (cover and closing slides).

## Slides

| # | Slide | Message | Example / proof shown |
|---|---|---|---|
| 1 | Cover | A second pair of eyes on the cash room | Concern card: `t = 44.0 s`, cash left outside storage, 2 frames, НБРК №231 п.20 |
| 2 | Problem | Hours of footage, seconds of risk | What the reviewer gets: verdict, type, second, 1–3 frames, norm |
| 3 | Watchlist | 14 concerns, each one visible on camera | 4 groups: storage (5), access (2), transaction steps (4), workplace (3) |
| 4 | Example 1 · flagged | The cashier walks away; the banknotes stay | Timeline 31.0 s → 39.0 s → **44.0 s flag** → 45.5 s supporting frame |
| 5 | Example 2 · not flagged | It looks suspicious, but it isn’t | 4 counterexamples + storage veto: false alarms 6 → 3, catches unchanged at 12/16 |
| 6 | Example 3 · output | One API call returns the verdict and the proof | `GET /api/v1/jobs/{job_id}` JSON with the 5 key fields explained |
| 7 | How it works | Five steps on one GPU, with no cloud AI | Upload → sample 2 fps → 16 frames to Qwen3-VL-32B → confirm + veto → report; guardrails; CPU fallback |
| 8 | Built for a bank | Self-hosted, fast enough and auditable | 1 GPU · 42/42 jobs · 73 s slowest · 1,196 tests; recipe fingerprint; no shortcuts |
| 9 | Proof · step by step | Every change was measured on the same 30 clips | Correct verdicts 12 → 16 → 20 → 23 of 30 |
| 10 | Proof · full run | All 40 clips through the live API | 27/40; matrix 13 caught / 8 missed / 5 false alarms / 14 cleared; 6/22 within ±2 s |
| 11 | Limits · next steps | The gap is recognition, and here is the plan | 16 DEV events: 5 right type and time, 4 wrong time, 7 none; sharper frames, native video, independent pilot |
| 12 | Close | A reviewer’s shortlist with proof, not a verdict machine | Ask: a pilot on fresh branch footage |

Slides 4–6 use an illustrative scenario (labelled on the slide). The output fields, timing rule (evidence frame nearest the timecode, supporting frame about 1.5 s later) and description template are the service’s real format (`service/pipeline/analyze.py::build_result`).

## Where every number comes from

Nothing here was re-run for the deck; all figures are copied from results already recorded in the repo.

| Figure | Source |
|---|---|
| 14 concern types, norm references | `service/rules.json` |
| 2 fps sampling, 16 frames, 16,384 context, 60 s call, 185 s budget, 200 MB upload | `service/pipeline/frames.py`, `deploy/mission-best.env`, `docs/REVIEW_CONTEXT.md`, `service/app.py` |
| DEV30: 12/16/20/23 correct; caught 2/6/12/12; false alarms 3/4/6/3 | `service/static/evaluation.json` (first three), `experiments.md` (storage veto) |
| All 40: 27/40, TP13 FN8 FP5 TN14, 6/22 within ±2 s | `README.md`, `EXPORT_MANIFEST.json` → `actual_api_qualification` |
| 42/42 jobs, 0 failed; 72.98 s worst; repeats 64.9 s and 65.9 s | `EXPORT_MANIFEST.json` → `actual_api_qualification` |
| 1,196 tests passed | `EXPORT_MANIFEST.json` → `public_export_test_result` |
| 16/16 events within 2 s of a sampled frame; 5 / 4 / 7 type breakdown | `EXPORT_MANIFEST.json` → `recognition_diagnostic` |

## Caveats to keep saying out loud

- All 40 clips were seen during development: the numbers are development/regression scores, not an independent test or an official jury score.
- Timing is the weak spot (6/22 within ±2 s).
- Sharper frames and native video input are planned or proposed, not measured. No gain is claimed.
- The tool flags observations for a person; it never declares a legal violation or identifies anyone.
