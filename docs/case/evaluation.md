# Video evaluation and experiment discipline

Preserve frozen `src/eval.py` and the original `src/video_eval.py` protocol provenance. All40 supplied unique recordings have been reviewed by explicit human choice; no independent internal holdout remains. Their scores are development diagnostics. A new split does not undo inspection/model-selection history.

Keep full-file SHA-256 identity and collision-checked aliases. The duplicate pairs are Х-4/Х-47 and Х-5/Х-48. Preserve original organizer binary/time labels separately from newly confirmed human-reviewed event/type notes. Both57/68 targets and multiple type assignments are retained in the versioned annotation layer; do not silently rewrite historical scores or protocol decisions.

## What to report

- Video confusion counts, recall, false alarms, precision/F1 and any existing AUC diagnostic; no conversion to jury points.
- One-to-one event matching at declared internal tolerances; matched/unmatched events, extra predictions, event precision/recall and type correctness. One prediction cannot satisfy several different events; spammed predictions must be visible in the false-event count.
- Source-time/frame validity, evidence relevance and public-source applicability separately from a schema pass.
- Every requested video in the denominator, with failures, timeouts and unprocessed counts.
- Complete API latency including upload, queue, cold start and final serialization; repeated/overlapping requests and actual hardware identity/memory.

Fit any learned preprocessing, temporal offsets or thresholds inside development folds and report excluded-recording results. Do not fit a median timing offset then present error on the same examples as an independent improvement. A hand-reviewed prompt/rule can still overfit these recordings; disclose selection history.

±1/±2 seconds and240seconds are internal diagnostics/engineering targets, not official grading tolerances. Type strings/order are a documented team policy, not an assumed exact-match jury protocol. There is no three-event cap in the supplied brief.

## Observation qualification

The first static checkpoint is an inference-only sampled-state diagnostic: same fixed prompt/views/schema on predetermined matched examples; labels are not passed to the model. It compares operational variation/consistency and attributed prior state observations, not independent human frame truth or full-video detector accuracy. It creates no learned classifier or deployment score, so do not fit a model or fabricate a shuffled-target score for it. Any later feature/model change uses the existing `src.train`/`src.sanity` video adapters, the approved frozen recipe and `experiments.md`.

Cached iteration is useful, but full decoding and real API tests are required before promotion. Version inputs, source, models, prompts, runtime/cache policy and outputs; fresh serving replay must match the extraction used to evaluate the candidate.

Transforms (flip/crop/rotation/brightness/resolution) are stress tests. Genuinely new authorized recordings from another viewpoint are a separate challenge; simple transforms do not establish new-angle robustness. Hidden jury videos remain inference-only and never enter development.
