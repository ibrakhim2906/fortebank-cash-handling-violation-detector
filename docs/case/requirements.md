# ForteBank case requirements and acceptance evidence

Approved selective v5 adoption, 17 October 2026. `docs/brief.md` remains the authoritative supplied case. Requirements below distinguish the organizer contract from internal targets; none is complete merely because it is written down. Current status is tracked in `CHECKLIST.md`.

| ID | Required behavior | Acceptance evidence |
|---|---|---|
| R1–R4 | Working public UI; upload MOV; processing state; detected-violation/no-detected-violation result | Actual external upload-to-result check, with errors handled |
| R5–R8 | Multiple possible events, type, source time, concise observable explanation, 1–3 actual JPEG evidence frames and applicable public RK source | Decode and source-align frames; inspect event meaning and source prerequisites separately from schema |
| R9 | Public UI/API reachable for remote jury uploads | Reachability from outside the host; machine remains available through judging |
| R20–R25 | Keyed multipart POST `/api/v1/analyze` (`file`) → `job_id`; keyed GET `/api/v1/jobs/{id}` → processing/done/failed and exact result schema | Contract/auth tests plus real public API request |
| R26–R28 | Each video completes within 300 seconds; failed/unreachable jobs count unprocessed | Upload/queue/cold startup included; repeated and overlapping jobs tested; original deadline retained through restart |
| R30–R38 | Correct video verdict, event timing/type, multiple events and robustness to new footage/angles/types | Confusion counts; one-to-one event matching; extra-event/failure counts; independent new footage where available |
| R40–R43 | Public RK sources, accurate source-to-visible-event relationship, no invented citations or invisible facts | Current version, exact quotation or marked paraphrase, exceptions and applicability documented |
| R50–R56 | Submitted processing uses local models, works offline after download, no face identification/audio; commercially permitted components; reviewer assistance | Network-denied inference, component provenance/licenses, source review and limitations |
| R57–R58 | Hackathon data restrictions; no hidden jury training/publication/third-party transfer | Data/source boundary and operational review; no hidden-test ingestion into development |
| R60–R65 | Public UI/API/key handoff, run instructions, architecture/models, sources/mappings, supported types and limitations | Completed `docs/submission.md` with actual tested values |
| R70 | Shortlisted pitch: five minutes plus two minutes Q&A | Timed rehearsal; API readiness precedes pitch polish |

Official rubric: detection 25, localization 15, classification 15, source grounding 15, robustness 10, product 10, technical solution/locality 10. No official formula converts development accuracy/AUC into jury points. Hidden set is 5+ videos, not necessarily five equal-weight items.

Internal targets: 240-second processing headroom, high recall with low false alarms, and timing diagnostics at ±1/±2 seconds. These are not organizer-confirmed tolerances or guaranteed performance. Type language, ordering and identifier length are team choices; there is no official three-event cap. Report all distinct supported events with 1–3 frames each.

No extra login, history, export or batch UI is required. Keep scope on recognition, evidence, sources and reliability. All supplied recordings were explicitly released for development review; no untouched internal holdout remains. Human annotations are team review, separate from organizer original binary/time labels.
