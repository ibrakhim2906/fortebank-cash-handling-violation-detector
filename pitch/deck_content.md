# CodeLint — pitch deck content (Team Seltik)

Live deck (Slides artifact, private until shared): https://claude.ai/artifact/LwnP6P6VqrWGmiEiahS9HU

11 slides, Montserrat, minimal layout, bullet points, no icons. Focus: what CodeLint does and its advantages.

## Slides

| # | Slide | Content |
|---|---|---|
| 1 | Cover | CodeLint — automatic review of cash-handling procedures in bank video. Team Seltik |
| 2 | The task | Input: cash-room video. Output: verdict, type, timecode, explanation, evidence frames. Conditions: API, under 5 minutes |
| 3 | Our solution | Upload a video, get every violation with its second, description and original frames; 14 types, НБРК norm link, secure REST API |
| 4 | What CodeLint detects | 14 types in 4 groups: storage and security (5), access and handover (2), transaction steps (4), workplace rules (3) |
| 5 | How it works | Upload → sample 2 fps → analyze (Qwen3-VL-32B, one A40) → verify → report; no external AI service |
| 6 | Example result | Violation · cash left outside the safe · 44.0 s · frames at 44.0 s and 45.5 s · НБРК №231, п.20 |
| 7 | Accuracy | Normal work is not flagged; second check of “safe left open” gives 50% fewer false alarms (6 → 3), same 12 caught |
| 8 | Results by version | 23/30 correct, 12/16 violation videos detected, 42/42 API jobs, 73 s slowest; bars 12 → 16 → 20 → 23 |
| 9 | Results on 40 cases | 27/40 correct; 13 caught, 8 missed, 14 cleared, 5 false alarms; 6/22 within ±2 s; footer on improving via model training |
| 10 | Advantages | Private, explainable, fast, reliable, auditable, responsible |
| 11 | Close | CodeLint — clear evidence for every cash-handling violation; ready for a pilot. Team Seltik |

Slide 6 shows an example of the real output format (fields, timing rule and norm link from `service/pipeline/analyze.py::build_result`).

## Where every number comes from

All figures are copied from results already recorded in the repo; nothing was re-run for the deck.

| Figure | Source |
|---|---|
| 14 violation types, norm references | `service/rules.json` |
| 2 fps sampling, 16 frames, model and GPU | `service/pipeline/frames.py`, `deploy/mission-best.env`, `docs/REVIEW_CONTEXT.md` |
| 30-video development set: 12/16/20/23 correct; 12 of 16 caught; false alarms 6 → 3 | `service/static/evaluation.json`, `experiments.md` |
| 40 cases: 27/40, TP13 FN8 FP5 TN14, 6/22 within ±2 s | `README.md`, `EXPORT_MANIFEST.json` → `actual_api_qualification` |
| 42/42 jobs, 0 failed; 72.98 s slowest | `EXPORT_MANIFEST.json` → `actual_api_qualification` |
| 1,196 tests passed | `EXPORT_MANIFEST.json` → `public_export_test_result` |

## Q&A backup

- Slides 7–8 are measured on the 30-video development set; all 40 given videos were seen during development, so these are not an independent test score.
- Planned improvements (higher-resolution frames, native video input, more training work) are not yet measured.
