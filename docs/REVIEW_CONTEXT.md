# Review context

This repository supports review of visually observable cash-handling procedures in the ForteBank datathon case. It identifies concerns for a human reviewer and returns original-frame evidence and source-relative timecodes. It cannot infer identity, authorization, invisible key/lock state or procedural omissions from missing/ambiguous observations alone.

## Current deployed recognition recipe

One self-managed NVIDIA A40 48GB runs Qwen3-VL-32B-Instruct-AWQ through vLLM. The service sends16ordered image blocks with actual source timestamps,16384context tokens and a60second model-call ceiling clamped to the185second analysis deadline with20seconds reserved for fallback. `AUDITOR_SPATIAL=storage_veto` is enabled. It checks only storage-left-open proposals and vetoes only an explicit, complete physical counterexample; uncertainty preserves the initial proposal. Other modes are disabled in the selected recipe. No hosted cloud AI API participates in inference.

## Actual aggregate quality

The completed actual public-API run processed 40 unique recordings plus two repeats: 42/42 primary-path jobs, zero failed jobs. Operability and recognition are separate findings. Across the 40 recordings, verdict accuracy was 27/40: TP13, FN8, FP5, TN14. Only 6/22 annotated events were localized within ±2 seconds. DEV30 reproduced 23/30 correct verdicts (TP12/FN4/FP3/TN11), with 5/16 annotated events within ±2 seconds. The previously seen 10-recording regression subset scored 4/10 correct verdicts, 1/5 positive-video recall and two false positives.

Worst processing time was 72.984294733 seconds. Two additional actual repetitions of the slowest source took 64.933887418 and 65.902824726 seconds. These are measured timings, not a latency guarantee for unseen inputs.

All 40 supplied unique recordings were historically reviewed. The fixed 30/10 split supports development and regression checks, not independent untouched validation. Duplicate aliases are grouped by full-content SHA. No private per-video labels, error tables, footage or job outputs are included. These results fall short of the desired recognition standard; they are not an official jury score.

## Recognition diagnosis and experiment status

Using actual saved source timestamps, all 16 DEV anchors had a nearest coarse frame within 2 seconds, and 15 within 1 second. This oracle proximity calculation does not establish action visibility or model correctness. A relevant correct type appeared anywhere for 9/16 anchors: five had correct timing and four had wrong timing, with errors of 8.5, 12, 25.4667 and 25.5 seconds. The other seven had no relevant type: four missed positive videos and three positive verdicts based only on an unrelated type. Consequently, a correct binary verdict alone can mask a recognition failure, and timestamp quantization alone does not explain the localization gap.

Code audit confirmed that the initial omission coverage gate blocked clips longer than 8 seconds, tiled mode passed allow_omissions=False, and source-resolution requests used max_pixels=524288. These are implementation facts, not demonstrated remedies. The earlier dense 64-frame trial regressed from 20/30 to 19/30 verdict accuracy under its original comparison recipe. An exact one-factor refinement experiment is currently running; a source-resolution experiment is next. Neither has a winning result claimed here.

The native-video proposal is unapplied and remains untested on the GPU. Its initial eight-kind taxonomy excludes four DEV anchor types (no_receipt, no_id_check, keys_left_in_safe, workstation_unlocked), placing initial typed/time coverage at a maximum of 12/16 even with perfect recognition of eligible anchors. It cannot constitute a complete final solution. Any later omission extension requires separately demonstrated dense actual source coverage and visibly bounded procedure phases; missing or ambiguous evidence cannot become an absence claim.

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
