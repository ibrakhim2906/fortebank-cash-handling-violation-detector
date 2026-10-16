# Pitch deck outline (P11 fills this into `pitch/deck_content.md`)

Rules (from the organizers' workshop): don't explain the problem at length · one idea per slide · show the work, not the stack · every brief requirement visibly ticked · the pitch may be only 1.5–2 minutes.
Every number must come from `experiments.md` or `reports/`. Brackets = fill with real values.

## 2-minute cut (5 slides, ~280 spoken words)
| # | Slide | Headline (≤ 8 words) | Visual | Say (time) |
|---|---|---|---|---|
| 1 | Cover | [Solution name] — [one-line promise] | Team name | 5 s |
| 2 | **The insight** | [The finding, with its number] | Chart from `reports/insights/` | 35 s — the moment they remember |
| 3 | Why it decides | Every decision comes with its reasons | One client: risk % + top 5 reasons (or live demo) | 35 s |
| 4 | Impact in ₸ | [N] M ₸ per 10,000 [applications] above no-model | `reports/impact.png` + assumptions | 25 s |
| 5 | How ForteBank uses it | [Where it plugs in] in [N] weeks | 3 steps: where · who sees it · next step | 20 s |

## Backup slides (show only in Q&A or a 5-minute slot)
| # | Slide | Content |
|---|---|---|
| B1 | The decision | Who at ForteBank decides what, how often; today vs with our model (their words from `docs/jury.md`) |
| B2 | Why you can trust it | Validation (time-based, label gap, holdout) · leakage checks passed (shuffled AUC ≈ 0.50) · baseline → final score · fold stability |
| B3 | Every requirement covered | Table: requirement from the brief → where we show it → done |

## 5-minute version
Order: 1 → B1 → 2 → B2 → 3 (live demo) → 4 → 5 → B3 on screen during Q&A.

## Output P11 must write to `pitch/deck_content.md`
For each slide: headline, visual (file path), on-slide text (short), speaker notes. Then the 2-minute script, the 5-minute script, and a CHECKLIST.md → slide mapping.
The team pastes this into the "Datathon ’26 Pitch Template" deck (claude.ai artifacts) or any slide tool.
