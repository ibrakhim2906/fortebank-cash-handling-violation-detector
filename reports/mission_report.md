# MISSION report — selected service and honest validation snapshot

Latest snapshot: **18 October 2026, 06:42 Asia/Qyzylorda**. Public checkpoint3 has completed all40 clean uploads and two distinct slowest-source repeats; the offline primary/source/recipe qualifier succeeds. All40 clean jobs use the selected primary recipe with no failed jobs. Maximum clean time is72.984294733s; the two qualified repeats take64.933887418s and65.902824726s. The original06:30 partial deadline snapshot is preserved below. Selective storage veto remains deployed and tagged; proposal_refine/storage_veto is now a pending controlled comparison, not a retained improvement.

**The recognition targets are not met.** Actual public-API DEV recognition reproduces **23/30 correct verdicts,12/16 positive videos detected,3 false alarms and5/16 annotated events within±2seconds**. The updated measured engineering proxy is **55.8854166667/100**, versus45.8854166667 before full API runtime qualification: **the entire+10 is Product/runtime evidence, with no recognition gain**. The previously seen reserve scores4/10correct,recall1/5,FP2; all40score27/40correct,recall13/21,FP5 and6/22events within2seconds. The reserve remains previously seen prospective regression, not an independent test. Dense-input, storage-crop, desk-rescan, broad factual-ledger and concise action-window recipes fail promotion. All280 robustness/self-check jobs remain deferred/zero credit, and strict licence qualification remains false/Tech0. Passing operational gates does not establish excellent recognition.

## Case and service contract

The ForteBank case concerns incorrect cash-operation conduct and omissions that can be established visually. The prediction unit is an uploaded video. An object transfer is an observable action, not automatically a violation. The service must not invent cash contents, ownership, authorization, identity checks, tiny key states or computer-session lock state from ambiguous appearances.

The required keyed interface is POST /api/v1/analyze with multipart field file, followed by GET /api/v1/jobs/{job_id} using X-API-Key. Job states are processing, done, failed. Completed output has violation/no_violation and typed source-time events, observable explanations and 1–3 original JPEG evidence frames, with qualified normative source/point information in descriptions. The official processing limit is five minutes; MISSION's target is 200 seconds. No audio or face identification is used. API keys remain outside reports.

Sources: [official brief](/home/ibrak/datathon-kit/docs/brief.md), [API specification](/home/ibrak/datathon-kit/docs/case/api_spec.md), [checklist](/home/ibrak/datathon-kit/CHECKLIST.md). The selected public endpoint is [the existing A40 service](https://p7ayn21cpy6voe-8000.proxy.runpod.net). Its verified worker freeze and all 40 completed clean checkpoint3 traces and both independently uploaded repeat traces match selected recipe `d4c64e727fab0890048f332a03e90e1fae61343e3ddb39afce1bbad864140bfc`.

## Data, labels and honest validation

The 42 supplied files contain 40 unique full-content SHA groups: 21 positive and 19 negative. Duplicate aliases stay together. Team-reviewed event/type notes are distinct from the original organizer binary/time sheet; disagreements remain disclosed. All 40 videos were historically reviewed at the human's request. The frozen 10-video reserve is therefore **previously seen prospective regression**, not an independent test. No reserve per-video errors were used for these iterations.

Development contains 30 groups: 16 positive, 14 negative, 16 annotated event anchors. Camera-excluded reporting uses the three actual team-supplied setups, with validation sizes 26, 1, 3. No supervised timing calibration is fitted. Repeated development-driven selection can overfit despite group exclusion. Non-crossfit numeric fallbacks receive no honest primary-model credit and remain in the requested denominator. Exact-request cache replay is allowed for unchanged predictions, but its walltime never qualifies API latency.

Type accuracy is conditional on events localized within 2 seconds. Typed coverage below divides correct localized type matches by all 16 annotated events. Do not describe 5/5 conditional matches as 100% event recognition. All metrics use the unchanged frozen evaluator; hashes remain 0ef1963df44a04495ecf95d8c40c82c606ac7987ac410202094c5e671483a762 for src/eval.py and 8effdc878f842accee556e349d5a768ae2b7ea16a79b6d058cdc3a000e61b658 for scripts/eval_video.py.

Evidence: [split/folds](/home/ibrak/datathon-kit/data/processed/mission-validation-v1.json), [protocol disclosure](/home/ibrak/datathon-kit/reports/validation_protocol.md), [team annotations](/home/ibrak/datathon-kit/data/processed/team-annotations-v5/v1/normalized.json).

## Complete development comparisons

Every row uses all 30 DEV videos, including failed/excluded primary predictions. These are conservative engineering proxies, not official jury aggregation or forecasts.

| Recipe | Correct verdicts | Positive videos found | False alarms | Primary completed/requested | Events±1s | Events±2s | Correct type/localized | Typed coverage/all events | Proxy/100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original 8B strict | 12/30 | 2/16 | 3 | 28/30 | 1/16 | 1/16 | 1/1 | 1/16 | 33.4375000000 |
| Source-resolution 8B | 16/30 | 6/16 | 4 | 30/30 | 3/16 | 3/16 | 2/3 | 2/16 | 33.6458333333 |
| 32B proposal baseline | 20/30 | 12/16 | 6 | 30/30 | 4/16 | 5/16 | 5/5 | 5/16 | 43.3854166667 |
| 64-frame 32B dense input | 19/30 | 11/16 | 6 | 30/30 | 4/16 | 4/16 | 3/4 | 3/16 | 38.3333333333 |
| Grounded storage/door crops | 20/30 | 10/16 | 3 | 28/30 | 3/16 | 5/16 | 4/5 | 4/16 | 39.9166666667 |
| Grounded desk/staff rescan | 13/30 | 13/16 | 8 | 21/30 | 4/16 | 5/16 | 5/5 | 5/16 | 37.5520833333 |
| Factual per-frame state ledger | 19/30 | 7/16 | 0 | 28/30 | 3/16 | 4/16 | 4/4 | 4/16 | 41.6145833333 |
| Selective storage counterexample veto | **23/30** | **12/16** | **3** | **30/30** | 4/16 | 5/16 | 5/5 | 5/16 | **45.8854166667** |
| Storage veto plus concise short action windows | 18/30 | 14/16 | 10 | 30/30 | 4/16 | 5/16 | 5/5 | 5/16 | 41.7187500000 |

- Original 8B strict: [honest report](/home/ibrak/datathon-kit/data/processed/mission-v1-dev-honest.json), [proxy](/home/ibrak/datathon-kit/data/processed/mission-v1-rubric.json).
- Source-resolution 8B: [honest report](/home/ibrak/datathon-kit/data/processed/mission-source-dev-30-honest.json), [proxy](/home/ibrak/datathon-kit/data/processed/mission-source-dev-30-rubric.json).
- 32B proposal baseline: [honest report](/home/ibrak/datathon-kit/data/processed/mission-32b-proposal-dev30-honest.json), [proxy](/home/ibrak/datathon-kit/data/processed/mission-32b-proposal-dev30-rubric.json).
- 64-frame 32B dense input: [honest report](/home/ibrak/datathon-kit/data/processed/mission-wide90-dev30-honest.json), [proxy](/home/ibrak/datathon-kit/data/processed/mission-wide90-dev30-rubric.json).
- Grounded storage/door crops: [honest report](/home/ibrak/datathon-kit/data/processed/mission-spatial-open-dev30-honest.json), [proxy](/home/ibrak/datathon-kit/data/processed/mission-spatial-open-dev30-rubric.json).
- Grounded desk/staff rescan: [honest report](/home/ibrak/datathon-kit/data/processed/mission-desk-dev30-honest.json), [proxy](/home/ibrak/datathon-kit/data/processed/mission-desk-dev30-rubric.json).
- Factual state ledger: [honest report](/home/ibrak/datathon-kit/data/processed/mission-ledger-dev30-honest.json), [proxy](/home/ibrak/datathon-kit/data/processed/mission-ledger-dev30-rubric.json).
- Selective storage counterexample veto: [honest report](/home/ibrak/datathon-kit/data/processed/mission-storage-veto-dev30-honest.json), [proxy](/home/ibrak/datathon-kit/data/processed/mission-storage-veto-dev30-rubric.json), [frozen recipe](/home/ibrak/datathon-kit/data/processed/mission-storage-veto-dev30-freeze.json).
- Concise short action windows: [honest report](/home/ibrak/datathon-kit/data/processed/mission-action-concise-dev30-honest.json), [proxy](/home/ibrak/datathon-kit/data/processed/mission-action-concise-dev30-rubric.json).

The selected DEV configuration's proxy components are detection 19.1666666667, localization 4.21875, conditional classification 15, qualified legal-source share 7.5, robustness 0, product 0, tech 0. Legal half-credit for empty rule bindings is a property of the requested proxy; it does not prove that a legal breach occurred. The completed network-denial proof alone cannot earn Tech points while strict licence qualification is unsatisfied; partial API completion cannot earn full-set Product points.

## What the new experiments establish

**Dense input: drop.** Same 32B model with 64 frames, 32k context and 90-second call budget completes 30/30 primary predictions, 21 new calls and 9 exact-request replays, with 0 budget flags. Recall falls 12/16 → 11/16, false alarms stay 6, localization falls 5/16 → 4/16 and proxy falls 43.3854166667 → 38.3333333333. Its maximum 65.175433388-second cached budget estimate is not actual public-API latency. All 30 outputs needed one call; the later multi-call replay-clock repair does not explain away the regression.

**Dynamic high-resolution storage/door crops: drop.** Raw service accuracy improves to 22/30, TP11/16, FP3, but two numeric fallbacks cannot receive primary-model credit. Honest results are 20/30, TP10/FN6, FP3, 2 failed/excluded, 5/16 localized events, 4/5 conditional type matches and 4/16 typed coverage. False alarms fall 6 → 3 while recall falls 12/16 → 10/16 and proxy falls 43.3854166667 → 39.9166666667. The trial uses 30 coarse replays and 21 new calls, 0 budget flags. Crops use source-detail JPEGs with original context; public evidence stays unchanged. Reducing false alarms by rejecting true concerns is not an accepted improvement.

**Grounded desk/staff rescan: drop.** Raw output has 22/30 correct verdicts, TP16/16 and FP8, but 9 results use the numeric fallback. Honest results are 13/30 correct, TP13/FN3, FP8, 9 failed/excluded, 5/16 localized events, 5/5 conditional type matches and 5/16 typed coverage. Proxy falls 43.3854166667 → 37.5520833333. There are 21 primary results, 19 new calls, 30 coarse replays and 0 budget flags. Raw 16/16 cannot be reported as 100% primary-model recall. Extending this recipe's timeout alone would not address its increased false alarms.

**Correctness repairs retained without an accuracy claim.** Candidate-order budget exhaustion now continues to later already-supported proposals. Evaluation-only replay advances admission clocks by cached model durations and respects bounded call limits; network clocks stay real. A restored 16-frame comparison preserves 30/30 baseline outputs with 30 exact coarse replays and 0 new calls. These repairs do not raise the measured recognition score.

**Factual state ledger: drop.** A complete 30-video export yields 28 primary results and two numeric fallbacks. Raw accuracy is 21/30; honest accuracy is 19/30, TP7/FN9, FP0, hit ±1s 3/16, hit ±2s 4/16, conditional type matches 4/4 and typed coverage 4/16. The proxy is 41.6145833333, below the retained baseline's 43.3854166667. Eliminating false alarms by dropping true positives (12/16 → 7/16) fails promotion. The ledger cannot find actions absent from the initial sweep or solve unseen taxonomy by itself. Selective reuse as a storage-only counterexample veto succeeds; adding short action windows subsequently increases false alarms and is rejected.

## Selective storage veto deployed; action-window comparison rejected

Code `9765270` keeps the same resident 32B model, 16-frame/16k context, 60-second per-call limit and proposal baseline. The first trial uses `AUDITOR_SPATIAL=storage_veto`: factual ledger verification is applied **only to safe-left-open proposals**. Explicitly observed normal closure can veto that proposal; unresolved detail or timeout preserves the original coarse positive. Door proposals and other candidate behavior remain unchanged. This avoids the earlier ledger policy's broad recall loss from treating uncertainty as a rejection.

**Selective storage veto: keep and deploy.** All 30 DEV primary predictions complete with 0 failed/excluded results: 23/30 correct, TP12/FN4, FP3, 4/16 hit ±1s, 5/16 hit ±2s, 5/5 conditional type matches and 5/16 typed coverage. Relative to baseline, correct verdicts improve 20 → 23, false alarms 6 → 3, recall stays 12/16 and proxy improves 43.3854166667 → 45.8854166667. This uses 39 exact-request replays, 0 fresh calls and 0 budget flags; cached timing does not qualify public latency. Historical DEV tag `good-0549` records source `9765270`, recipe `4c4cc7ab92e5f29b597b9c9cafc9678caf57d25ffce6043479d8f61c7e652def`.

**Concise short action windows: drop.** Completed `action-concise-dev30` has 30 primary outputs, 30 fresh calls plus 39 exact replays, and zero failures/budget flags. TP increases 12/16 → 14/16, but false alarms increase 3 → 10, accuracy falls 23/30 → 18/30, and proxy falls 45.8854166667 → 41.71875. Localization and correctly typed coverage stay 5/16. Uncertain overall answers sometimes contain per-candidate clear/prerequisite claims, admitting normal transfers or work items as concerns. Additional positive verdicts therefore do not establish correctly recognized violations. The earlier interrupted action-window attempt is preserved as partial and has no complete score.

**Actual selected serving identity.** Source `746c5cb` retains the `7cfc154` service pipeline; later changes add checkpoint tooling only. The unused optional action-prompt source changes the recipe hash even with that module disabled. Actual selected recipe is `d4c64e727fab0890048f332a03e90e1fae61343e3ddb39afce1bbad864140bfc`: 32B AWQ, 16 images, 16384 context, 60-second call limit, proposal policy, standard resolution, storage veto on, desk rescan off, action windows off, frame-only perception. Worker environment and source hashes are verified in the [actual freeze](/home/ibrak/datathon-kit/data/processed/runpod-setup-v1/mission-checkpoint-2-storage-selected/actual-freeze.json). Full public API scoring of this identity is pending; do not relabel the earlier cached DEV timing as its API latency.

## Why the existing numeric fallback is not a trained-quality result

The existing final numeric artifact was fit on 32 legacy groups. Of the nine current desk-rescan fallback videos, **six are in its final training manifest and three are outside**. Their selected raw correctness is not independent full-pipeline evidence. It would also be inaccurate to claim all nine were memorized. The primary-qualified evaluator correctly excludes all nine because the deployed fallback is not cross-fitted under the current protocol.

A bounded supervised alternative was investigated using only 222 existing gray/color/edge/motion features. 24 current DEV caches passed exact current source/configuration provenance and finite-vector checks; 6 are missing. No model or extraction ran because the frozen camera split exposes a more fundamental limitation:

| Frozen DEV fold | Training positive | Training clean | Validation positive | Validation clean |
|---|---:|---:|---:|---:|
| leave-setup-a |4|**0**|12|14|
| leave-setup-b |15|14|1|0|
| leave-setup-c |13|14|3|0|

**All 14 clean DEV recordings belong to setup A.** Holding A out leaves four positive training recordings and no clean example. A learned binary head cannot establish a two-class decision boundary in that fold. No constant surrogate, fake fold, reserve-label import or artificial score was substituted. Additional descriptors or a stronger encoder alone do not resolve this validation confound. Clean and violating examples across setups are needed for a meaningful learned-head camera-generalization claim.

This does not invalidate grouped aggregation of the fixed pretrained observer, which is not fitted to the case-fold training labels. Its repeated-development-selection/historical-review limitations still apply. [Exact audit](/home/ibrak/datathon-kit/data/processed/mission-supervised-dev30-v1/audit.json) and [explanation](/home/ibrak/datathon-kit/data/processed/mission-supervised-dev30-v1/feasibility.md). No current supervised OOF result exists.

## Historical baseline development failures

The prior baseline misses four positive videos and falsely flags six clean videos. This is a historical comparison, not the current selective-veto error list:

| DEV video alias | Error |
|---|---|
| Х-40 | Missed violation |
| X-1 | Missed violation |
| Х-25 | False alarm |
| Х-12 | False alarm |
| X-2 | Missed violation |
| Х-29 | False alarm |
| Х-27 | False alarm |
| Х-39 | False alarm |
| Х-21 | False alarm |
| Х-34 | Missed violation |

The false-alarm cluster is five safe-left-open claims and one door-left-open claim. Omissions and personal items remain important missed action groups; tiny keys/unlocked-session concerns are not established by video-level true positives for other cues. The per-type counts below overlap when a single anchor accepts multiple labels, so their totals must not replace the global 16-event denominator.

| Annotated type | Labeled occurrences | Localized±2s | Correct type |
|---|---:|---:|---:|
| `unauthorized_access` | 3 | 2 | 0 |
| `hand_to_hand_transfer` | 2 | 2 | 2 |
| `door_left_open` | 2 | 1 | 1 |
| `cash_left_out` | 4 | 1 | 1 |
| `food_drink` | 1 | 1 | 1 |
| `personal_items` | 1 | 0 | 0 |
| `no_receipt` | 1 | 0 | 0 |
| `no_id_check` | 1 | 0 | 0 |
| `safe_left_open` | 1 | 0 | 0 |
| `keys_left_in_safe` | 1 | 0 | 0 |
| `workstation_unlocked` | 1 | 0 | 0 |
| `counting_without_client` | 1 | 0 | 0 |

No-count-check and personal-activity have zero DEV support; absence of DEV examples does not establish that those types work. The selective-veto contender retains the same per-type localization/typed counts, but removes three baseline false alarms. Root must bind the eventual current per-video failure list to its final selected report; do not mix error identities across recipes.

## Actual public API checkpoint and robustness

Checkpoint1 completed 44 actual public API jobs: 10 reserve, 4 predeclared DEV, 28 transformed DEV and 2 longest-video repeats. There were 0 operational failed jobs. Of 42 jobs with analysis traces, 41 used the primary model and 1 transformed case used numeric fallback. The two runtime repeats retained statuses/times but no selected-primary recipe trace; they do not establish two qualified primary runtime passes.

- Ten previously seen reserve videos:accuracy 4/10, recall 1/5, FP2, hit ±2s 1/6, type 1/1. No reserve per-video failure analysis was used for tuning.
- Four predeclared DEV videos:accuracy 3/4, recall 1/2, FP0, hit ±2s 0/2.
- All 14clean uploads:accuracy 7/14, recall 2/7, FP2, hit ±2s 1/8; clean maximum 51.142505405s.
- **All 44 observed jobs:**maximum 76.512474010s, on an augmented input. Do not substitute the clean-subset maximum for the overall maximum.
- Longest-video repeats: 47.210525177s and 44.749218023s, with the provenance limitation above.
- 28 transforms of 4 DEV sources:raw accuracy 20/28 = .7142857143; primary-qualified 19/28 = .6785714286, versus clean 3/4 = .75. Re-encode signatures match 4/4 DEV pairs, not all 40 sources.

This historical bounded scope receives zero full-set Product/Robustness points. All 280 transformed predictions and the full renamed/re-encoded self-check remain deferred, with zero robustness credit. The checkpoint runner persists clean results and selected-primary slowest-clean-source repeats before any long robustness work; this does not retroactively repair missing old traces or prove transformed worst-case latency.

Evidence: [summary](/home/ibrak/datathon-kit/data/processed/mission-checkpoint-1/summary.json), [runtime/primary audit](/home/ibrak/datathon-kit/data/processed/mission-checkpoint-1/runtime-and-primary-audit.json), [scope](/home/ibrak/datathon-kit/data/processed/mission-checkpoint-1/scope.json).

**Current checkpoint3 is terminal and qualified.** The first selected checkpoint2 client stopped before its evaluator submitted a video; its counter/scope entry remains preserved as an aborted-before-inference attempt. Replacement `mission-checkpoint-3-storage` completes through the public URL with no model-reply cache. The offline qualifier verifies40unique source groups,40primary clean jobs and42unique actual job IDs including the two repeats, no missing trace records, and exact selected recipe/frozen-evaluator provenance. Raw and primary-qualified aggregate recognition agree; no fallback or failed clean job is silently dropped.

| Actual selected public API scope | Correct verdicts | Recall | False alarms | Events±1s | Events±2s | Conditional type | Completed / requested | Worst clean time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV30, camera-excluded reporting |23/30|12/16|3|4/16|5/16|5/5|30/30|72.984294733s|
| Previously seen reserve, aggregate only |4/10|1/5|2|0/6|1/6|1/1|10/10|69.170502054s|
| All40, aggregate only |27/40|13/21|5|4/22|6/22|6/6|40/40|72.984294733s|

Actual public-API runtime qualification is **true**. The slowest completed clean source is selected by its observed72.984294733s runtime, then uploaded twice as new jobs: **64.933887418s** and **65.902824726s**. Both complete with verified primary recipe and satisfy≤200seconds. These are actual API durations, not replayed benchmark walltimes. The primary recipe remains `d4c64e727fab0890048f332a03e90e1fae61343e3ddb39afce1bbad864140bfc`. Recognition reports deliberately carry `latency_qualified=false`; the separate actual-API runtime artifact supplies this operational qualification.

The measured proxy now totals **55.8854166667**, using unchanged DEV recognition and actual qualified operational gates:

| Proxy component | Measured points |
|---|---:|
| Detection |19.1666666667/25|
| Localization |4.21875/15|
| Classification, conditional on a localized match |15/15|
| Legal, current qualified rule coverage |7.5/15|
| Robustness |0/10|
| Product |10/10|
| Tech |0/10|

The+10 over45.8854166667 is entirely Product evidence; detection, timecodes, type coverage and legal behavior do not improve in this checkpoint. All280transforms and the full renamed/re-encoded self-check are explicitly deferred; zero robustness credit remains. Strict licence qualification is false, so Tech stays0 despite the actual tested offline-processing proof. This is the requested engineering proxy, not a jury score or a90–95forecast.

Evidence: [DEV honest aggregate](/home/ibrak/datathon-kit/data/processed/mission-checkpoint-3-storage-qualified-v1/dev-honest.json), [reserve aggregate only](/home/ibrak/datathon-kit/data/processed/mission-checkpoint-3-storage-qualified-v1/reserve-aggregate.json), [all40 aggregate only](/home/ibrak/datathon-kit/data/processed/mission-checkpoint-3-storage-qualified-v1/all40-aggregate.json), [actual API runtime](/home/ibrak/datathon-kit/data/processed/mission-checkpoint-3-storage-qualified-v1/actual-api-runtime.json), [proxy components](/home/ibrak/datathon-kit/data/processed/mission-checkpoint-3-storage-qualified-v1/rubric.json), [qualification provenance](/home/ibrak/datathon-kit/data/processed/mission-checkpoint-3-storage-qualified-v1/provenance.json). No reserve per-video errors, predictions, aliases or trace copies are used in this report update.

**Historical06:30 deadline snapshot.** At06:30:15 there were28/40complete clean traces, reserve10/DEV18, all primary/audit-complete/same recipe,28unique job IDs andzero trace/transporterrors. Completed worst69.170502054s,DEVworst67.762737024s. No repeats or terminal API-stage/runtime/summary files existed at that sample. This accurately records partial status at the original deadline; the terminal qualification above supersedes it without retroactively claiming deadline completion.

## Model, external resources and licenses

The 32B comparison uses QuantTrio/Qwen3-VL-32B-Instruct-AWQ, revision c2d44b1376c73fa120fdb4e216c2ef61c1945498, AWQ-Marlin/float16 on the one existing NVIDIA A40. The quantized and official parent weight cards are verified Apache-2.0. Official 8B rollback weights are retained and Apache-2.0. vLLM 0.11.0 and Transformers 4.57.1 are locally installed. No hosted cloud AI API is part of service inference. The model is loopback-only; only the existing keyed service is public. [Pinned 32B manifest](/home/ibrak/datathon-kit/deploy/qwen3-vl-32b-awq.json).

The installed license inventory/notice table is now complete: 238 distribution records, 316 actual notices, with a portable hashed archive. This resolves the earlier draft's stale table-incomplete statement. However, all_redistribution_notices_verified=false, strict_license_allowlist_satisfied=false and tech_gate_qualified=false in the audit. Python PSF, NVIDIA runtime terms and approved FFmpeg LGPL are preserved accurately rather than relabeled Apache/MIT. Inventory completion alone does not earn Tech credit. [License audit](/home/ibrak/datathon-kit/data/processed/mission-v6-bootstrap/license-audit-summary.json), [notice summary](/home/ibrak/datathon-kit/data/processed/mission-v6-bootstrap/license-notice-completion-summary.json), [submission table](/home/ibrak/datathon-kit/docs/submission.md).

Public action datasets have been acquired separately with recorded URLs, UTC timestamps, hashes and licenses. MEVA supplies two clean source videos with six example action annotations; its 14 visualized exemplars burn ground-truth answers into pixels and are reference-only. Actions such as opening a door or transferring an object are not automatically bank violations. No model training or bank-score improvement from these resources is asserted here. [MEVA resource record](/home/ibrak/datathon-kit/data/processed/mission-external-meva-v1/README.md).

A completed public CAP primitive probe uses **12 publisher-labeled short action tasks from seven source videos/seven actor groups**. The same self-hosted 32B model gets **9/12 correct**, with 0 failed requests and 0 unknown replies, in 12 actual model calls. Maximum task time is **10.950502618 seconds**, mean 8.408196755 seconds. Inputs contain eight chronological original frame indices, resized to a 624-pixel edge/JPEG70 with no label overlays; publisher truth stays outside model requests. This is an external primitive diagnostic, not bank-case violation accuracy, training completion, full-video detection or a jury grade.

| CAP primitive | Correct / requested |
|---|---:|
| Opens cabinet | 1/2 |
| Closes cabinet | 1/2 |
| Drink | 2/2 |
| Eat | 1/2 |
| Remove backpack | 2/2 |
| Remove outerwear | 1/1 |
| Opens facility door | 1/1 |

Observed mistakes include close-cabinet classified as open-cabinet, eat as drink, and one open-cabinet action as other. The small positive-action selection does not establish no-action false-alarm rates or bank-procedure reasoning. It documents ordinary-action limitations; the subsequent bank short-window trial is rejected separately for increased false alarms. [Aggregate diagnostic](/home/ibrak/datathon-kit/data/processed/mission-external-primitive-probe-v2/result-32b-v1/aggregate-diagnostic.json), [terminal model summary](/home/ibrak/datathon-kit/data/processed/mission-external-primitive-probe-v2/result-32b-v1/summary.json), [recipe/provenance](/home/ibrak/datathon-kit/data/processed/mission-external-primitive-probe-v2/result-32b-v1/recipe.json).

## Current target audit and final handoff fields

| Requirement | Completed evidence | Status |
|---|---|---|
| Honest proxy ≥90 | Current qualified proxy55.8854166667; recognition unchanged |Failed|
| Recall 1.0 | Selected DEV result 12/16 = .75 |Failed|
| At most 1 false-alarm video | Selected DEV result 3 |Failed|
| At least 90% events within ±2s | Selected DEV result 5/16 = .3125 |Failed|
| Type accuracy ≥.85 | Selected DEV result 5/5 conditional; coverage 5/16 |Conditional target passes; recognition incomplete|
| Selected-primary worst-case ≤200s twice | Clean worst72.984294733s; two verified primary repeats64.933887418/65.902824726s |Pass on actual public API|
| No outbound processing on selected recipe | Actual primary GPU proof passes in 27.481672797s |Pass for the tested processing path|
| Strict licence qualification | Complete inventory/notices; strict allowlist and redistribution qualification not passed |Not qualified; zero Tech credit|
| Full40 clean API |40/40primary,0failed;42uniquejobs with repeats |Pass;Product10|
|280 robustness/self-check |Explicitly deferred/incomplete |Zero Robustness credit|
| Selected deployment/publication | Selected worker/freeze verified; clean public main and mission-best pushed |Published best partial result; recognition targets remain unmet|

Latest completed full source test run for pipeline `7cfc154`: **1197 passed, 2 skipped, in 55.33s**. Actual CPU-off DEV API fallback completed **2.469121377 seconds**. Checkpoint API-only tooling subsequently passes 19 focused tests before commit `746c5cb`; the service pipeline is unchanged. These establish software/operability only. The legacy tabular train/sanity commands fail missing data/raw/train.csv; no tabular score is invented, and the old 32-group classifier must not be retrained as a substitute for the current DEV split.

**Actual selected-GPU network-denial proof passes.** The [terminal artifact](/home/ibrak/datathon-kit/data/processed/mission-selected-gpu-offline-proof.json) records `passed=true`, primary engine `mission-v1`, `audit_complete=true`, one actual model request and **27.481672797 seconds**, on the exact selected `d4c64e…` recipe. Model-tree guards and the processing worker deny new INET/INET6/PACKET sockets. This establishes the tested processing path without outbound network access; it does not establish recognition accuracy, cold-start timing or complete licence compliance. The separate terminal public-API artifact now qualifies the two required primary repeats.

Only Pod p7ayn21cpy6voe is authorized: GPU rate $0.59/hour, $15 total cash cap including storage, auto top-up off. The root agent reports a live provider read at 01:25 UTC confirming one RUNNING A40, cost $0.59/hour, created 17 October 15:12:17 UTC. At the deadline, about 10.3 hours × the estimated $0.601/hour GPU-plus-storage rate is approximately $6.18 spent, leaving approximately $8.82 of the human-reported $15. These are elapsed-time estimates, not a wallet/billing balance read; taxes or additional charges must remain inside the cap. No new Pod was created by these experiments.

**Finalization addendum, 06:20–06:21.** Authorized code-only publication to the [GitHub repository](https://github.com/ibrakhim2906/fortebank-cash-handling-violation-detector) succeeds through native WSL Git. Authenticated remote readback confirms public `main` and peeled `mission-best` at `a4cf69e276391ca21e170134eccbc130567f60c7`. Private `mission-best` points to environment-only freeze commit `2f02b3f`; the different Git identities reflect deliberately separate clean publication history. The selected serving recipe remains exactly `d4c64e727fab0890048f332a03e90e1fae61343e3ddb39afce1bbad864140bfc`, equal to the deployed worker and the public export's recomputed fingerprint. `deploy/mission-best.env` records the selected configuration; its private/public SHA-256 is `794ee2eb0e6f26b4c76534fb140d7a9a7dfbaecd2975d7f30c5bdea0244bc781`. Publication causes no Pod restart or recognition change. **The tag records the best measured partial result, not completed MISSION recognition targets.**

The clean [WSL checkout](/home/ibrak/fortebank-cash-handling-violation-detector) preserves code, configuration and an export manifest while excluding bank media, labels, private evidence, credentials, weights and oversized historical decoder binaries. Two export-only diagnostic-default redactions are recorded explicitly. Public-source tests pass1196, skip6 in45.55s: four skips concern unavailable private fixtures and two pre-exist. Existing working history, unrelated preparation and frozen evaluators remain preserved. Deadline arrival, publication and this report do not establish target completion. The historical deadline operational counts supersede the earlier06:18:48 sample. Checkpoint3 was partial at06:30:15 and is now terminal/primary-qualified at this06:42 update; the distinction is retained explicitly.

### Root must fill before final handoff

- Checkpoint3 full40+2 primary/runtime qualification is complete. Keep reserve reporting aggregate-only and requalify the eventual retained changed recipe before promotion.
- Preserve both deadline-partial and subsequent terminal API evidence; current selected offline proof, source identity and runtime qualification are recorded above.
- Keep deferred robustness/re-encode checks at zero credit until actually completed on the chosen recipe.
- Current measured proxy55.8854166667 includes Product10,Robustness0,Tech0. Refresh only from actual changed-recipe evidence.
- Continue the pending audit-driven recognition comparisons; no additional winner is retained yet. Publication/tag identity is recorded above.

Historical recipe identities for comparison, not unverified current deployment claims:

- `mission-v1-dev`: `1f95e816f2e04e23452d2c6403518855b26eea7dd5a82ea1278df8b2f7a99672`.
- `mission-32b-proposal-dev30`: `6eff8a1aadb7fac9aa9dfd17c9d1a65271e4a0b425d9c150cdec1a25352caef5`.
- `mission-wide90-dev30`: `d7fbb8594be74967874100840179de97b98c2f27eabf92c7d4f888828fb3a78f`.
- `mission-spatial-open-dev30`: `464ad9823508c3f8e153b808d8fdeb79259e62a93d85f09c5476b04f57a28221`.
- `mission-desk-dev30`: `65094c26aaaf9b90d2b4b4e58facf9d6b8b7a53485b3fd6e5266e68684d7b2e3`.

- `mission-ledger-dev30`: `b8f78a76c111b969a8b74ff3ffebd2f35e55d0fdcb70f9ab5f4180b639b94e62`.

- `mission-storage-veto-dev30`: `4c4cc7ab92e5f29b597b9c9cafc9678caf57d25ffce6043479d8f61c7e652def`.


## V7 continuation after this snapshot

**Actual DEV event-error decomposition.** Among16annotated anchors,9have a matching reported type somewhere:5within2seconds and4at the wrong time. The remaining7have no matching type (4no predicted events,3only unrelated types despite a positive verdict). The four wrong-time distances are8.5,12,25.4667and25.5seconds. Matching type elsewhere does not prove it is the same physical event. Refinement can address at most these four existing type/time mismatches; the seven missing/wrong-type anchors require better event recognition. This is a DEV-only diagnostic, not an independent score or timestamp calibration.

The actual coarse-sample oracle finds an overview frame within2seconds of **all16/16**DEVanchors,15within1second,maximum nearest error2seconds. Thus timestamp quantization alone cannot explain actual5/16localization. The oracle assumes perfect event identification and says nothing about visibility or short-action coverage. Duration metadata also bounds the audit: all40recordings exceed8seconds (11.634–142.534s,median21.95s),but only5/40exceed60seconds;DEVhas4/30above60seconds. Two DEVanchors concern omission types, whose current coverage guards remain required. [Event decomposition](/home/ibrak/datathon-kit/data/processed/mission-event-error-decomposition-v1.json), [coarse-time oracle](/home/ibrak/datathon-kit/data/processed/mission-coarse-time-oracle-v1.json), [duration context](/home/ibrak/datathon-kit/data/processed/mission-audit-duration-context-v1.json).

**Native CPU geometry is measured; native GPU recognition is not.** The [actual installed-processor probe](/home/ibrak/datathon-kit/data/processed/mission-native-cpu-grid-v1.json) passes all four CPU cases at4,40,159and160frames under vLLM0.11.0/Transformers4.57.1, using local cached model metadata, disabled CUDA visibility and two CPU threads. Processed frame areas are147456/131072pixels; visual-token counts are288/2560/10240/11520, with a maximum toy-control prompt of12230tokens before the full action catalogue. Explicit total-video-size overrides preserve the intended detail despite launcher pixel aliases. This probe makes **zero GPU model calls** and performs no model restart, bank-video request or service change. Its own fields retain `full_http_route_verified=false` and `matched_controls_qualified=false`: HTTP video decoding, pixel/time fidelity, GPU memory/runtime and recognition remain unqualified. CPU geometry alone is not recognition progress.

The first short-window attempt remains a preserved partial diagnostic. Source7cfc154 reduces absent-action verbosity; one auxiliary record improves98.647→31.759s. The complete concise comparison nevertheless increases false alarms to10 and is rejected. The selected public service therefore keeps action windows off and storage veto on.

Stopping the first trial also stopped its child API supervisor. The model remained healthy, the API was restored, and corrected controller55829 subsequently completes all30 results. This process-control mistake is preserved in progress.md; it does not alter saved recognition results.

The human instructs continuation after the original06:30 deadline. Checkpoint3 is now terminal and separately qualified. The independent source audit sets the next order: first test existing **proposal_refine** with standard resolution/storage veto, then test **32B source resolution** with proposal/storage veto. Both are config-only comparisons on the same30DEV and resident32B16/16384/60 model; neither requires a model restart. Root has launched proposal_refine/storage_veto/standard as controller65298 with unchanged source, public worker and resident model. Its complete comparison is pending; no recognition gain is claimed. Compare typed localized coverage, hit@2s, positive misses and false alarms with failures retained in the denominator. Preserve the omission-coverage guards: sparse global frames and the current omission-disabled tiled path cannot prove that required actions never occurred.

Only after those comparisons should the prepared native32B video/image-list control run be considered, with four matched public CAP clips and exact decoder-pixel verification before inference. **No native GPU run has executed at this deadline snapshot.** It remains an untested hypothesis. Preserve the current selected worker and complete its real API qualification before any controlled model-configuration change; run the full280 robustness/self-check only on the eventual selected winner. The deployed/tagged23/30 development contender is the best completed partial result; the current recognition targets remain unmet.
