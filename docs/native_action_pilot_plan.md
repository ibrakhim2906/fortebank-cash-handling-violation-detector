# Native-video action pilot — three development review tasks

This bounded follow-up is within the human's request to review actual videos locally and improve recognition, with their subsequent full-access delegation. It changes temporal input/coverage, not the supplied labels, evaluation split, classifier gate or legal rules. The original frozen spatial comparison must finish before its model server is stopped.

## Required operational gate

Use the existing approved Qwen3-VL-2B model, pinned runtime and newly provisioned local decoder. Before bank media, `scripts.check_temporal_protocol --run` must pass all three fixed motion/order controls through genuine native `input_video`: left-to-right, reversed and stationary. Verify the completed report against exact source/model/projector/runtime/decoder fingerprints. A failed/stale report blocks this pilot. Passing does not establish bank-action or timestamp accuracy.

The pilot owns a temporary loopback-only server, with the same native profile as the diagnostic: 2 FPS, helper timestamp interval zero, image budget 256, context 4096, one slot, CPU threads eight, seed 42, offline assets and disabled logs. List actual source-frame times in the leading prompt. These are supplied timing metadata, not fabricated native per-pair timestamps.

## Frozen review recipe, before any new bank inference

- Use at most the first three eligible existing development review tasks in queue order; skip human-reviewed tasks and existing revisions of this recipe. Exclude all eight reserved groups and verify each source file's hash and allowed path before opening it. Preserve both disputed anchors and all earlier proposals.
- Sample the existing review interval chronologically at requested 2 FPS, capped at 16 frames. Require at least four actual frames and an even count; trim a final odd frame explicitly instead of fabricating/padding a source observation. Preserve actual PTS, source indices, raw-frame hashes, shapes and requested times. Do not claim the remainder of a longer interval was reviewed.
- Encode equal-size frames with long side at most 640 pixels, no upscaling, as a temporary local 2-FPS MJPEG/AVI sequence. Preserve the encoded payload hash and source mapping. Native input travels only to literal `127.0.0.1:8003`; no URL fetches, environment proxies or redirects.
- Reuse the already fixed draft/check prompts and eight action primitives. Neither official target, organizer anchor, filename nor review status enters the prompt; the model receives the observed frame sequence and actual source times only. Critique receives only sanitized prior action states. Use strict existing action schema, temperature zero, seed 42, 256 output tokens, 45 seconds per request and 150 seconds per task. Fail on budget/schema/timing errors rather than silently reducing coverage.
- Keep a present observation only when both passes support it and visibility permits it; all other action states remain unknown. Same-model agreement is consistency only. Append `proposed` revisions and private audits; never create verified labels or replace a human record, including a human revision arriving during inference. Print aggregate counters/timing only.

## Decision

Stop this pilot after at most three tasks. If all actions remain unknown, preserve the finding and stop expansion. If useful observations appear, present the locally inspectable proposals and separate observed operation from unverified semantics. No full 32-record training run, event-to-rule activation or automatic bulk expansion is authorized by this pilot recipe. A next adaptation/evaluation recipe must be written from the pilot's actual findings, without threshold tuning or reserved-data use.
