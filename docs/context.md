# Review context

This repository supports review of visually observable cash-handling procedures in the ForteBank datathon case. It identifies concerns for a human reviewer and returns original-frame evidence and source-relative timecodes. It cannot infer identity, authorization, invisible key/lock state or procedural omissions from missing/ambiguous observations alone.

## Current deployed recognition recipe

One self-managed NVIDIA A40 48GB runs Qwen3-VL-32B-Instruct-AWQ through vLLM. The service sends16ordered image blocks with actual source timestamps,16384context tokens and a60second model-call ceiling clamped to the185second analysis deadline with20seconds reserved for fallback. `AUDITOR_SPATIAL=storage_veto` is enabled. It checks only storage-left-open proposals and vetoes only an explicit, complete physical counterexample; uncertainty preserves the initial proposal. Other modes are disabled in the selected recipe. No hosted cloud AI API participates in inference.

## Actual aggregate quality

Fixed30DEV groups:23/30correct video verdicts, TP12, FN4, FP3, with5/16annotated events localized within±2seconds. This is measured development performance, not an official jury score or90–95quality. All40supplied unique recordings were historically reviewed; the fixed30/10split is development/prospective regression, not an independent untouched test. Duplicate aliases are grouped by full-content SHA. No private per-video label/error table or footage is included here. A new actual40-video public-API qualification is currently running; this export does not claim its terminal result.

## API contract

Keyed `POST /api/v1/analyze` accepts multipart field `file` and returns `job_id`. `GET /api/v1/jobs/{job_id}` with `X-API-Key` returns processing/done/failed; completed results include violation/no_violation and typed events, seconds from source start, observable descriptions and1–3original JPEG evidence frames. The official processing limit is5minutes; engineering targets are tighter. Keys, raw uploads, per-job outputs and model weights are intentionally absent.

## Review priorities

Examine recognition coverage, false positives, action direction, time localization, primary-versus-fallback accounting and temporal/spatial evidence binding. Ordered image input permits cross-frame reasoning but does not exercise native video's distinct-frame temporal patch pairs; a matched public CAP diagnostic is prepared privately, with no native32B result claimed here. Preserve source-time evidence and validation denominators. Avoid filename/hash/per-video rules at inference.

This is a clean source snapshot for code review. It includes the current working source, including previously untracked Python and UI assets, rather than a replay of the base commit. Original local Git history remains untouched. Two legacy helpers have export-only removal of case-specific defaults, documented in EXPORT_MANIFEST.json; all other copied source bytes are exact, including frozen evaluators and AGENTS.md. There is no private dataset or working numeric fallback artifact, so training/full case evaluation and an actual model service require separate authorized local setup. Some dataset-dependent tests can be skipped or require local fixtures. Do not fabricate a runnable full-model result from this export.

## Requested Claude review focus

1. Temporal coverage: which actions can fall between sparse samples, whether short windows/native representation address coverage, and whether any absence is incorrectly treated as evidence of an omitted required step.
2. Prompt/schema semantics: identify contradictions between allowed response states, uncertain visibility, physical observations, event rules and their downstream interpretation.
3. Recognition correctness: require the correct event type and source time with supporting original evidence. A correct binary verdict through an unrelated concern or a nonqualified fallback is not sufficient.
4. Invariance and leakage: forbid inference use of filenames, case labels, split IDs, hashes, video-specific branches or label-bearing overlays. Preserve groups/denominators and distinguish historically reviewed DEV/regression from independent validation.
5. Native proposal: evaluate only as an unapplied hypothesis. Do not claim trained quality or service promotion without matched-input tests and actual case validation.
