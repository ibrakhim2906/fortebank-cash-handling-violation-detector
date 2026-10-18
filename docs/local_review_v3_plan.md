# Bounded local video review and spatial follow-up

The human explicitly clarified that they want actual videos/action drafts reviewed locally and detection work continued. This reopens one bounded local-model follow-up after the rejected constant-feature experiment. No new runtime/model download or package is required. All existing models, prompts, raw deliveries and frozen evaluation remain preserved. The cloud coding assistant cannot personally inspect the clips under the case's locality requirement; local model observations and aggregate checks must not be described as independently verified truth.

## Fixed recipe before any new pilot or score

- Keep the installed pinned Qwen3-VL-2B model, current CPU server, context 4096, image cap 256, one slot, seed 42 and temperature zero. Use a separate prompt client and separate `qwen-spatial-v3` feature module; never edit the old frozen feature helpers to introduce this recipe.
- For detector inputs, keep four uniform centers at duration fractions 1/8, 3/8, 5/8 and 7/8. Decode one frame per center. From that same frame provide an overview capped at 640 pixels on the long side and the first/last square crops along the long axis, each capped at 640 without upscaling. JPEG quality 90. Crops depend only on dimensions, never labels, people or organizer anchors.
- Ask static visible questions: people/hands count buckets; hand-object contact, rectangular sheets/stacks, handheld device, open container and work surface; overall visibility. `not_visible` means not visible in this sampled view, never absence of an action during the clip. No identity, intent, legal compliance, counting correctness or unsupported motion inference. Eight categorical fields produce 28 one-hot window features and 84 max/mean/std clip features. Do not add metadata as features.
- Validate model provenance before frames, local-only requests, strict response schema, source hashes, crop coordinates and increasing actual sample times. Timestamp alignment tolerance is max(0.25 seconds, 2/FPS). Operational errors and all-unobservable clips fail explicitly; never substitute zero features. Request budget 45 seconds, total extraction budget 240 seconds.

## Operational pilot, before binary scoring

Use the first three development groups sorted by full content hash, without consulting targets or annotation anchors. Require 12/12 valid window responses, no all-unobservable clips, each extraction under 180 seconds, at least two distinct clip vectors, and at least two non-visibility fields showing different definite categories. At least 25% of the five hand/object attribute observations must be definite rather than unclear. This gate tests variation and operation, not semantic accuracy. Preserve a failed pilot and stop this recipe if it fails; do not iteratively rewrite the prompt against these clips or inspect binary scores to relax the gate.

If this pilot passes, freeze source/prompt/schema/model/crops, reuse its exact cached features, extract the remaining development groups and run only the established linear/shallow-LightGBM recipes on unchanged folds and threshold 0.5. Run actual `src.train` and `src.sanity`. Preserve the promotion gate: AUC >=0.60, balanced accuracy >0.50, recall >=10/17, sanity/runtime pass; report localization separately. No holdout decoding/evaluation, threshold inversion, hyperparameter search or new score-based prompt tuning.

## Action drafts and local second pass

Action-label review is separate from the label-independent detector. The local model may inspect three chronological frames at up to 640 pixels within a queued development review interval, first draft visible action primitives, then recheck that draft against the same frames using a separate critique prompt. No official binary target enters either prompt. Preserve the original three proposals and append new revisions rather than overwrite them.

Keep only present observations supported by both local passes; disagreements remain unknown. Same-model agreement is a consistency check, not independent verification. Save source/frame/model/recipe hashes, times and both structured outputs only in local derived audit files; append only `proposed` records to the existing review journal, skip human-reviewed tasks and never create verified labels. Print aggregate counts only. Start with three tasks; if all results remain unknown, retain that finding and stop bulk expansion. Human review remains the route to independent action truth, but the requested autonomous local review need not wait for it.

No footage, frames, action labels or clip-specific descriptions go to cloud tools. No new dependencies, external uploads, submissions, commits or pushes. Useful detection improvement is an experimental outcome to measure, not a promise.
