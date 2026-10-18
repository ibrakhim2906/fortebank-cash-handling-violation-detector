# Decisions preserved for code review

- Keep all service model inference on the one team-managed A40; no hosted cloud AI API in the service.
- Selected recipe: Qwen3-VL-32B-Instruct-AWQ,vLLM,16ordered images,16384context,60second per-call ceiling,185second total analysis/20second fallback reserve; storage veto enabled.
- Storage veto may reject only explicit complete counterexamples for storage-left-open proposals; unknown/timeout retains the initial positive. Door/other proposals retain their behavior.
- Preserve original timestamps/evidence; never infer from filenames, labels, content hashes or per-video exceptions.
- Preserve frozen evaluators; keep failed/excluded predictions in requested denominators and distinguish primary versus fallback.
- Fixed30/10split after all40were historically reviewed: no independent-test claim. Current DEV23/30,TP12/FN4/FP3,hit±2s5/16. A new actual40-video API qualification is running.
- This export is code/docs only. Private labels/footage/reviews and downloaded binaries are excluded; existing workspace/history is unchanged.
- Exported legacy helper defaults are redacted only in this snapshot, not the original working tree. Read EXPORT_MANIFEST.json for exact identity.
- Opt-in `AUDITOR_RECALL` (off|rescue|localize|on, default off) targets the recognition gap without changing the selected recipe: `rescue` adds a phase-shifted second sweep (actual source samples between primary samples) and sends its candidates plus primary `uncertain` ones to the existing independent dense focused confirmation (storage veto still applies; omissions never rescued; at most3confirmations); `localize` re-times up to3sparse-sweep proposals from the source interval between neighbouring sweep samples. Recall failures log `recall_incomplete` and never turn a primary negative into a fallback. `deploy/mission-recall.env` is a candidate only: promote it after a DEV30 run beats23/30and5/16, and report each one-factor mode separately.
