# Local recognition follow-up

The human asked for actual local video/action review and continued recognition improvements. No supplied footage, frames, model observations or clip-specific descriptions were sent to a cloud model. Original deliveries and frozen `src/eval.py` remain unchanged. Eight reserved videos remain undecoded and unevaluated.

## Actual autonomous review

The installed local vision model rechecked three development review tasks using chronological draft and critique passes. All three completed without failures. Three further **proposed** revisions were appended, bringing the proposal journal to six records. There were zero present-action agreements and zero independently verified action labels. The predeclared guard stopped bulk expansion. Same-model consistency is not independent truth, and official binary targets do not identify action types.

## Fixed detection comparisons

Both comparisons use the same 32 development content groups, three folds, fold-fitted preprocessing, seed 42, established linear/shallow-LightGBM recipes and threshold 0.5. No fusion, threshold search or reserved-data selection is permitted. Promotion requires development AUC at least 0.60, balanced accuracy above 0.50, recall at least 10/17, and passed integrity/sanity/runtime checks. Localization is reported separately. Repeated development-set use and unknown camera/session grouping limit any generalization claim.

The spatial recipe uses four fixed instants with an overview and two dimension-based crops per instant. Its label-blind operational pilot passed: 12 valid windows, three distinct clip vectors, two varying definite fields, 91.67% definite hand/object observations and maximum extraction 89.883023 seconds. This establishes variation and operation, not semantic accuracy. Plan: `docs/local_review_v3_plan.md`.

The object recipe uses the pinned Apache-2.0 OpenCV Zoo YOLOX-S model through existing approved OpenCV. It samples at 2 FPS, retains six predefined COCO classes and constructs short anonymous IoU tracks. Two-second windows have 57 features; clip pooling yields 171. Source, model and license hashes are checked. No cash/hand class, identification, verified action labels or automatic rule mapping is supplied. Plan: `docs/object_recognition_plan.md`.

### Completed object comparison

Actual run: `data/processed/video-object-tracks/`.

| Metric | Numeric linear baseline | Object tracks + linear | Object tracks + shallow LightGBM |
|---|---:|---:|---:|
| Development ROC-AUC | 0.509804 | 0.466667 | 0.323529 |
| Balanced accuracy | 0.460784 | 0.564706 | 0.317647 |
| Recall | 10/17 | 9/17 | 4/17 |
| Precision | 0.500000 | 0.600000 | 0.307692 |
| F1 | 0.540541 | 0.562500 | 0.266667 |
| TN / FP / FN / TP | 5 / 10 / 7 / 10 | 9 / 6 / 8 / 9 | 6 / 9 / 13 / 4 |

The selected linear model fails the unchanged AUC/recall promotion gate. **Drop from live promotion.** Fewer false alarms do not compensate for the reduced recall and below-chance ranking. The live numeric model remains solely a flow-testing baseline.

Localization is mixed across the same 16 eligible positive recordings, independent of verdict: median error **5 → 2 seconds**, mean error **6.189583 → 7.4375 seconds**, and within-three-seconds coverage **31.25% → 56.25%**. The duplicate group with conflicting anchors remains excluded. These are candidate-window measurements, not evidence that every missed positive would receive an event or that action boundaries are correctly recognized. Preserve the object representation for possible later independently supervised work; do not fuse it into the live model based on these development scores.

Matched sanity passes: 20 shuffled-target trials have mean AUC **0.5156862745**, standard deviation **0.1326423850**, zero suspicious single features and no shared/decoded holdout groups. The full cache audit validates **32 records, 2,056 sampled instants, 524 windows, 5,396 accepted object predictions, 441 anonymous track segments, 32 distinct clip vectors and 163 nonconstant features**. Source hashes, sampling, track assignments and pooled features agree; no frame content or semantic truth was independently verified. The varied features therefore work operationally but do not establish useful violation detection.

### Completed spatial comparison; serving withheld

Actual run: `data/processed/video-qwen-spatial-v3/`. The fixed four-instant representation completed on all 32 permitted development groups.

| Metric | Numeric linear baseline | Spatial + linear | Spatial + shallow LightGBM |
|---|---:|---:|---:|
| Development ROC-AUC | 0.509804 | 0.605882 | 0.615686 |
| Balanced accuracy | 0.460784 | 0.694118 | 0.656863 |
| Recall | 10/17 | 10/17 | 11/17 |
| Precision | 0.500000 | 0.769231 | 0.687500 |
| F1 | 0.540541 | 0.666667 | 0.666667 |
| TN / FP / FN / TP | 5 / 10 / 7 / 10 | 12 / 3 / 7 / 10 | 10 / 5 / 6 / 11 |

The predeclared highest-pooled-AUC policy selects shallow LightGBM; do not switch to linear afterward based on its fewer false alarms. Matched LightGBM AUC improves **0.409804 → 0.615686**; selected-model comparison improves **0.509804 → 0.615686**. This clears the minimum numerical detection gate, not the quality needed for jury delivery. Fold AUCs vary substantially (0.816667, 0.75, 0.44); only 32 development groups and repeated development use make these estimates uncertain.

Localization worsens on the same 16 eligible positive groups: median error **5 → 9.733333 seconds**, mean **6.189583 → 11.635417 seconds**, and within-three-seconds coverage **31.25% → 6.25%**. Four static instants do not supply continuous action coverage. Action classification, multiple events and rule grounding remain unimplemented by this representation.

Sanity completed: 20 shuffled-target trials average **0.4004901961**, standard deviation **0.0744475359**, with no suspicious single features or shared/decoded holdout groups. This narrowly passes the existing absolute-0.1 tolerance around chance; it is a weak diagnostic, not strong reassurance, and is not rerun to obtain a nicer number. The complete audit validates **32 caches, 128 instants, 384 views, 16 distinct clip vectors and 50 nonconstant features**, with maximum recorded extraction **107.999271 seconds**. These are integrity/variation checks, not semantic verification.

Direct inference on the predetermined longest development clip took **24.344331 seconds**, with four local vision requests and zero attempted external Python connections. A subsequent fresh replay took **21.909363 seconds** but differed from the frozen training cache in **five clip-feature entries and six window-feature entries**. The exact-replay assertion failed. The active process command matches the pinned launcher, expected alias and one-slot configuration.

The bounded follow-up diagnostic completed two unchanged extractions. All four source indices, source-frame hashes, actual times and twelve JPEG-view hashes match the training cache. Both new request bodies match each other and the frozen serialization probe; historical request hashes were not recorded, so historical byte equality cannot be established directly. The new feature vectors match each other, but both differ from the original cache in the same five/six entries and **both flip the saved classifier's thresholded verdict relative to the cached descriptors**. Candidate time is unchanged. This is material training-to-serving drift, not pooling roundoff or an observed decoding mismatch. Private records and aggregate report: `data/processed/video-qwen-spatial-v3/replay-diagnostic-1/`. The failed original check is preserved.

The pinned runtime documents prompt-cache numerical nondeterminism as a possible mechanism; causality is not established here. Do not silently turn caching off in the frozen client: a corrected extraction policy would need a new version, fresh descriptors and validation. **Withhold live promotion.** Preserve the measured development gain and failed replay; leave the numeric service available for flow testing.

The temporary candidate API passes engineering checks: UI HTTP 200, missing/incorrect keys 401, two development uploads complete in **79.545473** and **81.188368 seconds**, valid evidence JPEG/schema, identical bytes renamed return the identical complete result in **23.364396 seconds**, and a corrupt file fails safely in **3.558562 seconds**. These checks do not override the separate verdict drift or establish accuracy. Report: `data/processed/video-qwen-spatial-v3/api_check.json`. The temporary API is stopped after testing.

The native-video recovery route is separate from this frozen trial. Pinned local FFmpeg/ffprobe are installed under `tools/local-video-decoder/`, with hashes, sources and LGPL licenses preserved. The first actual native-video gate produces three valid structured responses in **14.865009 seconds** total, but only the forward control is correct: reverse and stationary controls fail. There is no operational failure. **Gate failed; no native bank-video pilot was started.** Report and exact helper source snapshot are preserved under `data/processed/temporal-protocol-native/`.

The separately predeclared cache-policy diagnostic completes six requests in **24.005464 seconds** with explicit `cache_prompt: false`. Every response is `left_to_right`; only **2/6** controls are correct. All six report `timings.cache_n = 0`. This policy does not restore the native motion controls. That finding does not resolve the separate spatial cache-history/drift hypothesis or establish an exclusive cause. No cache-policy repair, V4 training or native bank pilot follows. Report/source snapshots: `data/processed/local-cache-behavior/`.

A separate color diagnostic completes **4/4 correct** responses in **15.454847 seconds**: red/blue native four-frame videos and matching single JPEGs each return the expected color. Input hashes and usage/cache counters are preserved in `data/processed/native-visual-input-v1/visual_input.json`. This verifies color-dependent visual input through both paths; it does not establish that the model distinguishes frame order, motion or cashier actions.

An exact reproduction of the pinned runtime's FFmpeg buffer-decoder command passes all **three synthetic cases**: each exits successfully, emits exactly four frames and preserves the known forward, reverse or stationary order. Seekback warnings occur despite the correct decoded byte counts/order. Report: `data/processed/native-decoder-reproduction/decoder_reproduction.json`. This narrows the diagnosis without proving how the full model represents or understands those frames.

The two fixed native color-order controls, RRBB and BBRR, complete **2/2 correct** first/last-color responses in **10.037758 seconds**. Both report `cache_n = 0` and 180 prompt tokens. Report: `data/processed/native-frame-order-v1/frame_order.json`. Visual color input and coarse first/last order therefore work; this does not establish square-motion, cashier-action or precise event-timing accuracy. **The failed motion gate is preserved and the native bank-video pilot remains blocked.** No bank or reserved footage was used in these checks. Plan: `docs/inference_cache_repair_plan.md`; broader recovery/pilot plans: `docs/recognition_recovery_plan.md`, `docs/native_action_pilot_plan.md`; decoder provenance: `docs/local_video_decoder.md`. Spatial promotion remains withheld for the separate measured training-to-serving drift.

## Evidence and outstanding semantics

A separate pure presentation helper can select up to three supported source times from a stable non-person track **inside the already selected positive window**. It cannot change the verdict or window, and is not activated merely because its synthetic tests pass. It describes uncertain object appearance only; it cannot establish use, ownership, employment, storage, intent, omitted counting or a legal violation. Plan: `docs/object_evidence_plan.md`.

The current official public clauses are documented in `docs/action_rule_candidates.md`, but action-to-rule associations remain disabled. Independent local action annotations are still needed to measure action-specific accuracy. Multiple-event behavior, grounded event types/rules, public hosting and full jury delivery remain incomplete.

Review of the presentation helper identified two requirements before any activation: match selected evidence to original source indices/raw-frame hashes rather than approximate timestamp seeks, and preserve the original candidate time separately from the first evidence image. The helper now rejects low-confidence predictions and track gaps/jumps inconsistent with the fixed tracker. It remains inactive after the failed detection trial.

## Verification so far

The final full-suite run passed **753 tests**, with **two optional-data skips** and **12 existing LightGBM deprecation warnings**, in **33.38 seconds**, including the new frame-order and spatial cache-policy diagnostic tests. The prepared spatial cache/history experiment itself has **not run**: the human requested a full strategic reassessment and external-data evaluation. The proposed next direction is documented in `docs/recognition_recovery_plan.md`. No model/action accuracy follows from software tests. No new package was installed for this follow-up. No commits, pushes, external media uploads, submissions or public deployment occurred.

Reassessment verification also confirms both original raw delivery hashes and frozen `src/eval.py` remain unchanged. The original local Qwen service is restored; Linux API/review/model health endpoints return HTTP 200. No new feature/model training was run during this strategic assessment, so the recorded before/after model scores above are unchanged.

Windows loopback API and review pages on ports 8001 and 8004 also return HTTP 200. The reassessment incorporates separate architecture/validation and dataset-license reviews. Review corrections explicitly fix the pilot class denominator, require full-length runtime stress coverage and prevent streaming state from crossing windows/recordings/folds.

Follow-up files: added `scripts/check_native_frame_order.py`, `tests/test_native_frame_order.py`, `scripts/check_spatial_cache_policy.py`, `tests/test_spatial_cache_policy.py` and `docs/spatial_cache_diagnostic_plan.md`; updated `docs/recognition_recovery_plan.md`, `docs/context.md`, `docs/decisions.md`, `docs/inference_cache_repair_plan.md`, `README.md`, `GAMEPLAN.md` and this report. The earlier color-probe source was executed unchanged. The generic pitch timing in `GAMEPLAN.md` now matches the case's five-minute pitch plus two-minute Q&A.

Additional commands actually run:

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m scripts.check_native_visual_input --run --run-dir data/processed/native-visual-input-v1
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m scripts.check_native_frame_order --run --run-dir data/processed/native-frame-order-v1
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline pytest -q
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m scripts.serve_action_model
git diff --check
sha256sum src/eval.py
```

Read-only Python/PowerShell commands verified raw-file hashes, RAM/GPU exposure, public dataset class metadata and local endpoint status. No bank pixels or clip-specific semantic output was emitted. Research did not download model weights or training images/videos.

Actual commands:

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m src.train --task video --encoder object-tracks --run-dir data/processed/video-object-tracks --baseline-run-dir data/processed/video-baseline-comparison --models linear,lgbm --note 'fixed local YOLOX object timelines and anonymous tracks'
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m src.sanity --task video --run-dir data/processed/video-object-tracks
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m scripts.audit_object_observations --run-dir data/processed/video-object-tracks
```
