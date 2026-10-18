# Recognition recovery plan — reassessed against the full ForteBank case

Updated 17 October 2026 after the human requested a second evaluation of the whole approach and then approved the first experiment. MoViNet-A0-Stream float16 TFLite v2 and the explicitly approved LiteRT runtime are installed locally; synthetic runtime/repeatability checks pass. Semantic training remains gated on independently reviewed action labels. No external training corpus has been acquired. The previously prepared spatial cache/history experiment is deferred, not executed. Earlier recovery reasoning is retained below for provenance. Current experiment evidence: `reports/movinet_experiment.md`.

## Decision from the reassessment

Latest approved diagnostic: the direct pretrained probe now completes on eight development training recordings, with repeatable local MOV inference but weak fixed target-class rankings and no independent action accuracy. Reusing existing classes has been tested before further training; it has not established a usable cashier-action detector. Do not expand or promote the unchanged generic classifier on the strength of runtime alone. Preserve the separate supervised transfer pilot. Next recognition work must distinguish missing action vocabulary, insufficient visible detail and actual classification failures using independently checked examples; public workflow/rule context must define what is being detected. Evidence: `reports/pretrained_probe.md`.

Keep the working local upload/API/evidence infrastructure. Change the recognition priority to independently reviewed action intervals, continuous temporal coverage, one compact pretrained video encoder and a small supervised classifier. Treat public rules as separately reviewed applicability conditions. The current VLM is not a reliable primary verdict or action-label engine; repairing its numerical drift alone would not supply the missing action semantics.

External data is explicitly allowed and can help. Use pretrained temporal knowledge and small, relevant, licensed additions to teach visual actions. Preserve the distinction between a visible action and a possible bank-rule violation. A bigger generic corpus does not supply missing cashier-role, transaction-stage, ownership or regulatory labels.

## What changes when the ambition is 95–100

The human asks for the best chance of a near-perfect jury result. This is an ambition, not a forecast: the exact score formula is unspecified and our current measurements do not support that expectation. A 95-point result permits only five lost points across all seven categories. A working model or a successful one-action pilot alone cannot justify it.

Local execution is mandatory; pretrained weights are a practical starting point rather than a requirement that guarantees quality. With only 32 development recordings and no verified action labels, training a capable video backbone from scratch is not a credible primary route. The proposed adaptation is a locally running pretrained temporal encoder with a small supervised classifier, conditional on evidence that its representation preserves the needed action detail. Pretraining learns general visual patterns from an external corpus; adaptation learns reviewed cashier actions from development data; new jury videos are inference-only. MoViNet-A0-Stream and LiteRT are installed, but the completed direct-class probe has not established cashier-action accuracy. A larger training run is not automatically the right response to weak unlabelled predictions.

Strengthen the plan in these specific ways:

1. **Make action and rule coverage the first deliverable.** After the small feasibility pilot, independently review all 32 development recordings locally for observable event types, intervals, relevant context, hard negatives and uncertain regions. Build a coverage table showing which action families have verified examples, applicable public rules and independent evaluation support. A binary violation label does not provide those annotations. Ask mentors to clarify ambiguous rule applicability and evaluation timing, without requesting hidden-test content. Do not fabricate a complete taxonomy from model proposals.
2. **Evaluate the whole event output.** Before scoring, freeze matching conventions and record misses, false alarms, type correctness, event-time errors, evidence relevance and rule applicability. Explicitly test multiple events and reviewed omissions. Report per-family results and support counts alongside the video confusion matrix; a good average must not conceal an unsupported family. The supplied positive timestamp is not an exhaustive event list.
3. **Obtain independent viewpoint evidence.** Prioritize team-owned staged action/counterexample pairs filmed from another angle if the team can supply them. Separate filming sessions into training and untouched challenge sets before use. Keep all derivatives of a recording together. These clips supplement the bank data; they do not establish accuracy on the hidden bank test. Image augmentation alone does not demonstrate robustness to a different camera.
4. **Improve the cause of a measured failure.** Start with the compact temporal encoder and verified labels. Add hand/object detail only if the pilot identifies missed small objects as the bottleneck. Consider a larger model or limited backbone adaptation only if suitable local compute, sufficient labels, commercial rights and the five-minute runtime are established. Avoid spending the remaining time on an unconstrained model search.
5. **Finish delivery in parallel with recognition.** Prepare the public hosting configuration, exact API contract, offline startup, evidence display, failure handling, component licenses and five-minute end-to-end checks early. Public deployment and key handoff remain separate concrete delivery actions. Reserve the final hours for a frozen-policy evaluation, a complete demonstration and the five-minute pitch; do not leave reachability testing until the deadline.

Keep the existing feasibility and promotion gates unchanged: they are minimum safeguards, not excellence targets. Define any stricter event-level acceptance criteria before evaluating them. Preserve the eight reserved recordings for one final frozen-policy check; repeated development comparisons and an eight-recording holdout cannot establish a precise generalization rate or jury grade. Never turn unsupported actions into confident violations to increase coverage.

The highest-value human contribution is local verification of action/rule meaning and, if possible, a second-camera recording session. Engineering, local extraction, training, measurement and integration can remain primarily autonomous. No new runtime, training, external corpus or deployment is authorized merely by this plan revision.

## Datathon and case context

- Event: Datathon '26, 17–18 October, Nazarbayev University, Astana, teams of 2–4. The public site says the 24-hour clock starts at 12:00 on 17 October and the final pitch starts at 13:00 on 18 October. The case FAQ specifies a 12:00 deadline; use 18 October 12:00 local time as the working delivery deadline. Exact submission channel and any later organizer change still need confirmation. [Official event schedule](https://datathon.com.kz/).
- Client: ForteBank. A reviewer uploads a previously unseen silent cashier video and receives possible visible violations with events, times, evidence and public Kazakhstan rule references. This supports human review, not a definitive legal or disciplinary finding.
- The complete internal cashier regulations and violation taxonomy are not supplied. FAQ 32 says dataset violations do not depend on undisclosed internal rules without a public normative basis. FAQ 9 allows unseen violation types in the test set.
- Final evaluation includes a different camera angle. A scene-identity or fixed-pixel-location shortcut is particularly risky.
- Finalists give a five-minute pitch plus two-minute Q&A; slides are optional. Case-specific FAQ takes priority over generic two-minute-pitch suggestions in the preparation kit.
- Inputs: human-supplied `open.zip` and `Кейс_ForteBank.docx`, preserved in `data/raw/fortebank-20261017/`. The brief and organizer FAQ are transcribed in `docs/brief.md`. `docs/jury.md` has no recorded additional mentor answers; do not invent them.

## Supplied data and what its labels mean

| Item | Verified state |
|---|---|
| Delivered files | 42 MOV recordings plus `open_labels.xlsx` |
| Spreadsheet labels | 22 positive / 20 negative rows; binary target and positive timestamp, no action type |
| Unique recordings | 40 content groups: 21 positive / 19 negative |
| Development | 32 groups: 17 positive / 15 negative; three frozen folds |
| Reserved evaluation | 8 groups: 4 positive / 4 negative; still undecoded and unevaluated |
| Video headers | H.264; 464 × 848 track dimensions; approximately 11.63–142.53 seconds and 29.76–30 FPS; no audio |
| Organizer scene description | Staged footage, one overhead camera and one cash desk; cashier, customers and collectors |
| Annotation problem | One duplicate recording has two different positive anchors, 57 and 68 seconds; preserve both and exclude it from timing evaluation until resolved |
| Additional action supervision | Six machine proposals, zero independently verified action labels |

Organizer timestamps are sparse anchors, not complete event intervals or proof that other intervals are negative. Frame multiplication and augmentation do not turn 40 recordings into thousands of independent examples. Group all windows, crops, duplicates and augmentations from one recording together. Exact-content grouping does not establish actor/session independence.

## Required product and scoring

The user uploads MOV without manually running code. The interface shows progress and a binary result. Each positive event requires a type, timecode, observable description, 1–3 original JPEG evidence frames and a relevant public RK rule reference. Multiple events must be supported. Put rule/source references in the required description/UI without changing the jury JSON schema.

The keyed asynchronous API is `POST /api/v1/analyze` with multipart field `file`, returning `job_id`, then `GET /api/v1/jobs/{job_id}` returning `processing`, `done` or `failed` and the final result. Use `X-API-Key`, reject wrong/missing keys, preserve `no_violation` / `violation`, and return an empty event list for a negative verdict. Complete every accepted evaluation video within five minutes. Deliver public UI/API URLs, jury key, run instructions, model/architecture description, source-to-rule mapping, supported event catalogue and limitations.

| Rubric | Points | What our next work must demonstrate |
|---|---:|---|
| Detection | 25 | Measured false alarms and misses on independent recordings |
| Localization | 15 | Correct original frame/time inside the reviewed event |
| Event classification | 15 | Correct observable action/type, supported by independent labels |
| Rule grounding | 15 | Applicable public rule, correct interpretation and visible evidence |
| Robustness | 10 | Viewpoint change, occlusion, hard negatives and unseen-video behavior |
| Product readiness | 10 | Reachable UI/API, reliable upload/status/result flow |
| Technical solution/locality | 10 | Reproducible offline inference and a practical architecture |

Detection, localization, classification and rules account for 70 points. More infrastructure tests cannot substitute for those capabilities. No official jury score can be inferred from our development ROC-AUC; the rubric's exact automated scoring formula is not supplied.

All inference stays on the serving computer and works offline after installation. No cloud AI calls, bank footage, frames or descriptions may be sent externally. No face identification or audio analysis. No claims about invisible bookkeeping, intent, ownership or checks performed offscreen. Hidden jury uploads are inference-only. No ready-made finished solution. FAQ 24 requires commercially usable open-source code/pretrained models, and FAQ 34 reserves commercial use for ForteBank. New package installation still follows AGENTS.md approval and uv requirements.

## Does progress make sense?

The first local service, duplicate-aware split, preserved raw data, original-frame evidence and refusal to promote failed experiments were useful. The allocation now needs correction: repeated generic representations and extensive protocol debugging have not produced a validated action recognizer.

| Measured development result | Numeric baseline | Best spatial candidate |
|---|---:|---:|
| ROC-AUC | 0.509804 | 0.615686 |
| True positives / positive recordings | 10 / 17 | 11 / 17 |
| False positives / negative recordings | 10 / 15 | 5 / 15 |
| Median candidate timestamp error, 16 eligible positives | 5.000 s | 9.733333 s |
| Live status | Flow-testing only | Withheld: fresh extraction flips a saved verdict |

The spatial result is an improvement in a weak development proxy, not demonstrated excellent detection. Its fold AUCs are 0.816667, 0.75 and 0.44. Repeated reuse of the same 32 groups further weakens model-selection confidence. Object tracks and earlier appearance/action-descriptor trials failed their promotion gates. See `reports/recognition_followup.md` for all measured comparisons and limitations.

What has been trained so far is primarily small classifiers on numeric/pretrained observations; no successful cashier-action-specific video-model training has been established. Four static instants discard event order and can miss short actions. The six-class object tracker does not directly detect hands or cash. The local VLM's unknown action drafts do not create training truth.

The current diagnostics show that color-dependent native video and first/last color order work (4/4 color controls and 2/2 order controls), while the original square-motion gate still fails. This does not establish cashier-action accuracy or prove all temporal input is broken. Preserve those findings; no more open-ended VLM prompt or generic-feature search is the primary plan.

## Hardware and model choice

Fresh inspection reports about 18 GiB WSL RAM and an AMD Radeon integrated graphics adapter on Windows. Neither `/dev/dxg` nor `/dev/nvidia0` is exposed to WSL. No GPU training/inference acceleration is verified. The installed quantized Qwen/llama.cpp stack is an inference stack, not a validated fine-tuning setup.

First candidate to investigate: **MoViNet-A0-Stream**, a small temporal video network intended for efficient streaming inference. Start with frozen pretrained features, or documented logits if a feature signature is unavailable, and a regularized local action/window classifier. Do not fine-tune a large backbone or VLM in the critical path. The official streaming recipe uses 172 × 172 inputs at 5 FPS; the implementation publishes a small TFLite export. Those facts do not establish latency or quality on this machine. The official README warns against choosing the Base implementation for fast CPU inference simply because its checkpoint is small. [Official implementation and CPU guidance](https://raw.githubusercontent.com/tensorflow/models/master/official/projects/movinet/README.md), [exact A0 streaming model documentation](https://raw.githubusercontent.com/tensorflow/tfhub.dev/master/assets/docs/tensorflow/models/movinet/a0/stream/kinetics-600/classification/3.md).

The first experiment now pins the exact **A0-Stream float16 TFLite version 2** artifact, its checksum and **ai-edge-litert==2.3.0**. Official Google/Kaggle model metadata explicitly assigns Apache 2.0 to that exact variation/version; evidence and license texts are preserved in `licenses/movinet/`. The human approved runtime installation. No Base or int8 substitution is used. See `docs/movinet_experiment.md` and [Google model metadata](https://www.kaggle.com/api/v1/models/list?search=movinet). This closes the earlier exact-weight-license and runtime-installation questions.

General-action pretraining may still miss small cashier hand/object movements. A real, independently reviewed local action pilot is the acceptance test. The model name is a hypothesis, not a promised fix.

## Proposed recognition flow

1. **Decode the full recording with exact source timestamps.** Preserve chronological coverage and aspect ratio. Use overlapping short temporal windows instead of four isolated instants. Initial proposal: eight-second windows at five sampled frames/second with a two-second stride, subject to the first bounded CPU benchmark; freeze the final recipe before scoring. Reset streaming state to zero for every independently classified window/view and use the identical policy in extraction and serving. No state may cross recordings or folds.
2. **Encode motion and local detail.** Use one frozen temporal encoder. Keep scene context; add a deterministic detail crop only if the reviewed examples show tiny hands/objects are the actual failure and the crop can be selected without labels or camera-specific coordinates. Reject a crop strategy that loses the event.
3. **Learn observable actions from reviewed intervals.** A small regularized head predicts supported actions/background from those features. External examples retain their own action meanings. Do not convert handoff, restacking or phone presence directly into a violation label. Fit preprocessing and learned heads inside the existing folds.
4. **Build separate event intervals.** Apply one predeclared temporal aggregation policy, retain multiple non-overlapping events and refine the supporting original frame around the candidate interval. Keep event time independent from arbitrary evidence-frame order. Do not report sub-second precision unsupported by sampling/evidence.
5. **Apply reviewed public-rule conditions.** Emit an event/rule association only when the required observable predicates and context are supported. Use fixed source-grounded explanation templates. The VLM can remain an optional private proposal aid; it does not create verified labels or control the primary verdict.
6. **Serve through the existing API/UI.** Reuse the working keyed job/evidence flow. Complete public hosting, offline startup, five-minute worst-case checks, run documentation and jury-key delivery after the chosen policy is frozen.

Rule work runs alongside action labeling. The current NBK №231 paragraph 12 and paragraph 19(2) catalog is source-verified but not activated: personal-property storage needs ownership/boundary context; omitted counting needs a sufficiently complete cash-acceptance interval. A visible handoff alone satisfies neither. Start with a few event families for which both a visible predicate and a defensible public rule can be established; do not invent a catalogue from model guesses. Explicitly disclose unsupported/unseen event types.

## External data: allowed, with a specific role

FAQ 7 explicitly welcomes own footage, synthetic data and open datasets (`docs/brief.md`, lines 205–207). External-data permission is not inferred from the separate pretrained-model allowance.

| Source | Useful role | Decision for this deadline |
|---|---|---|
| Public pretrained temporal weights | Transfer motion/action knowledge without downloading a full corpus | First priority after exact artifact/runtime/license checks |
| Open Images, a small verified subset | Hand/phone/container localization if the semantic pilot exposes that gap | Conditional auxiliary data; verified boxable classes include Human hand, Mobile phone, Drawer, Handbag and Envelope. Verify each image license and preserve attribution; it supplies objects, not cashier events. |
| Team-owned staged desk videos from another angle | Matched action/counterexample pairs and camera-change coverage | High-value optional addition: roughly 20–40 short clips with documented action intervals. All takes from one recording session stay grouped. Use an independently held filming session to test the angle change. |
| Something-Something V2 | Relevant generic object manipulation | Exclude under its current research agreement unless separate rights are obtained; agreement restricts derived results and combination with third-party data. |
| EPIC-KITCHENS | Hand-object action segments | Exclude from commercial-ready training under standard CC BY-NC terms; separate commercial permission is offered. |
| HaGRIDv2 | Hand localization/static gestures | Not a cashier-action corpus. Actual terms are a custom attribution/share-alike adaptation, not plain CC BY-SA. Full 119.4 GB 512px release is disproportionate; no download is planned. |
| BankNote-Net | Currency-image representation | Low priority: released embeddings support currency recognition, not temporal handoffs, counting correctness or bank-rule labels. |

Sources: [Open Images rights and per-image verification](https://github.com/openimages/dataset/blob/main/READMEV2.md), [official boxable classes](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv), [Something-Something terms](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf), [EPIC-KITCHENS terms](https://epic-kitchens.github.io/2025), [HaGRID dataset](https://github.com/hukenovs/hagrid), [actual HaGRID license](https://github.com/hukenovs/hagrid/blob/080e18917376ec935e453cd0e599c23478c7e98f/license/en_us.pdf), [BankNote-Net dataset/model description](https://github.com/microsoft/banknote-net).

External acquisition must have source URLs, license, hashes, labels and source/session grouping. Keep external data and derived labels separate from untouched official raw deliveries. Never use external videos labelled theft/anomaly as automatic positive labels for this cashier case. Add data only to address an observed failure, then measure transfer to the fixed bank development folds.

## Smallest decisive next experiment

**Before another 32-record extraction**, choose one genuinely observed, reviewable action and its difficult counterexamples. A possible primitive is person-to-person object handoff versus single-person relocation/contact; this is an example task, not a claim that it is present or unlawful in any supplied clip.

- Independently review source intervals locally. Start with eight training examples and exactly eight blind pilot examples, each group containing four action-positive and four hard-negative examples from distinct development recordings. Use the existing folds to separate training and pilot sources; do not touch the eight reserved recordings. If this support is unavailable, stop as insufficient support for this fixed pilot rather than changing its denominator, fabricating examples or claiming the action is trainable.
- Keep model proposals as proposals. A local human verifies action meaning, event interval, counterexample and uncertainty; the cloud coding agent receives aggregate status only. A repeated answer from the same model is not independent verification.
- Run a bounded local runtime/signature/repeatability test first. Reject the selected artifact if complete longest-development-video processing cannot fit the five-minute API budget with headroom. This is an early rejection check; the longest development recording is shorter than the case's 145-second upper range. Before claiming compliance, also run a full-length 145-second synthetic pipeline stress test, cover dense event/evidence output and measure total API time including queueing/serialization under the documented concurrency policy. Measure real CPU time; published device timings are not estimates for this laptop.
- Freeze the one encoder, input/window policy, regularized head, threshold and score rule before the blind pilot. Initial gross-failure gate: at least 3/4 reviewed positives and 3/4 hard negatives correctly recognized, relevant original evidence inside the reviewed event, and consistent repeated verdicts. Eight examples are only a feasibility gate, not a reliable accuracy estimate.
- If it passes, complete development action labels and one grouped three-fold experiment. Report video confusion counts, precision/recall/F1, per-action metrics, event coverage/time error, runtime and failure cases. Run the required training/sanity commands, preserve all results and log keep/drop. Compare with both recorded numeric and spatial results; do not tune against the reserved set.
- If the semantic pilot fails, stop that recognizer. Identify whether labels, resolution, action visibility or encoder transfer failed before spending more data/compute. Do not return to an open-ended model sweep.

The existing minimum development promotion gate (AUC ≥ 0.60, balanced accuracy > 0.50, recall ≥ 10/17, integrity/sanity/runtime pass) remains a floor, not an excellence claim. Do not silently alter it. Freeze any proposed additional event-quality criteria before their evaluation. Reserve the eight untouched recordings for one final frozen-policy check; after that, do not tune and reuse them as a fresh test. A small reserved set still cannot prove broad generalization.

## Work order and human input

The operational proposal below was prepared at approximately 15:49 on 17 October (Asia/Qyzylorda, UTC+05:00), with about 20 hours to the working noon deadline on 18 October. Times are planning limits, not measured task durations or promises. Implementation depends on the reviewed pilot labels and approval for any new runtime; a delayed start reduces the available experiment budget rather than consuming the final delivery reserve.

| Order / target | Work | Concrete checkpoint |
|---|---|---|
| First 2–3 working hours after implementation starts | In parallel, prepare the local label queue and independently verify the fixed one-action pilot; resolve the exact temporal artifact, license and runtime, then benchmark the approved stack. | Repeatable local inference, credible five-minute runtime budget and a passing semantic pilot. If labels are unavailable, do not train on guessed action truth. |
| After the pilot, during the evening | Expand verified action/interval/counterexample coverage across the 32 development recordings; run the fixed grouped comparison and investigate specific failures. | Actual before/after misses, false alarms, event types, timing and evidence quality. Stop expansion if the pilot fails; diagnose the cause before proposing a replacement. |
| In parallel with recognition | Verify action-to-rule conditions, integrate multiple events and original evidence, prepare the public serving configuration and deployment checks. | Every supported event has a visible justification and an applicable public source; exact API/UI and delivery configuration are ready for an end-to-end rehearsal. |
| If a teammate can supply the footage | Record and independently label second-angle action/counterexample clips, separating filming sessions before training or evaluation. | Viewpoint-challenge results reported separately; no hidden-test use or claimed robustness from augmentation alone. |
| By 08:00 on 18 October, target | Freeze the model, thresholds, extraction policy, event logic and rule mappings; run the one reserved evaluation and full-length runtime checks. | Preserved final results, failure cases and no subsequent tuning against the reserved set. A material post-freeze correction invalidates the earlier final-test claim; do not present a repeat as independent evidence. |
| 08:00–11:00 on 18 October, target | Complete authorized deployment, verify external API/UI reachability, package startup/licenses/limitations, rehearse the five-minute pitch and prepare the backup demonstration. | Working jury deliverables ready by 11:00, with one hour reserved for submission problems before the working 12:00 deadline. Confirm the actual organizer submission route. |

The first milestone is a measured action recognizer on independently labelled clips, with correct event evidence. The final milestone is the complete reliable service. Neither is satisfied by downloading a model. A pilot failure triggers an honest scope reduction and explicit limitations, not a claim that the existing flow-testing baseline is accurate. The datathon clock does not make an unvalidated model excellent.

Codex can own source research, subset manifests, local extraction/training, evaluation, event/evidence implementation, tests and documentation. The irreplaceable input is a small local human review of actual action examples and rule context; the existing review page can supply it without sending footage/descriptions to chat. Additional team-owned footage or access to a separate GPU computer could help, but the primary plan assumes this existing CPU machine. Public hosting and any new runtime/package approval remain concrete delivery decisions, not permissions implied by dataset research.

## Earlier recovery plan — retained as historical reasoning

The material below predates the second assessment. Its frozen experiments and safety boundaries remain historical evidence; it does not authorize executing the failed native pilot or override the proposed prioritization above.

## Outcome and current evidence

The human wants an impressive, working ForteBank video reviewer, built mainly autonomously under their guidance. The final model and all video processing stay local. Public pretrained models/code/documentation are permitted; task adaptation uses only the supplied training data. New uploaded evaluation videos are inference inputs. No automatic learning from jury uploads is planned.

The software flow works, but useful recognition is unestablished. The numeric live baseline has development AUC 0.509804, recall 10/17 and 10/15 false alarms. The object comparison fails promotion (AUC 0.466667, recall 9/17), despite improving median candidate timestamp error 5 to 2 seconds; its mean error worsens. The original Qwen action descriptors were constant. Three autonomous local draft/critique reviews returned no agreed present actions. None of this supplies independently verified action labels. The already-running spatial comparison remains frozen and must finish scoring and sanity checks; no additional generic-feature search is proposed.

## 1. Prove the temporal input path before another bank-data batch

Public primary sources establish a distinction between independent images and video input: Qwen's video processor groups frames in pairs and represents timestamps; the pinned llama.cpp b11707 implementation marks video frames as mergeable, unlike ordinary images. This establishes an untested input path, not proof of a bug or the cause of the failed experiments. The current spatial recipe intentionally tests still-image observations and must not be changed midway.

The pinned runtime documents `input_video` and uses local FFmpeg/ffprobe for decoding. Neither executable was found in WSL PATH or targeted existing Windows/bundled-runtime locations. Prepare a pinned, commercially usable standalone decoder proposal and synthetic checks first. Installation requires human approval under AGENTS.md rule 9. Do not replace the runtime/model or download larger weights speculatively.

Before bank footage, use four equal-size 256-pixel synthetic frames at times 0, 0.5, 1 and 1.5 seconds: a red square moving left-to-right, the reversed sequence and a stationary control. Freeze native video at 2 FPS with helper timestamp interval zero; list frame times once in the leading text. Source inspection shows helper timestamp text can interrupt adjacent-frame pairing, so precise timestamp alignment is a separate subsequent test; this is not a measured recognition defect. Require all three valid, correct responses, fixed seed, bounded context and requests, local-only processing and preserved source/provisioning provenance. If only time-labelled independent images are exercised, call that protocol by its actual name. A passed synthetic check proves basic motion/order operation only, not event localization or bank-action accuracy.

Deliverable: a reproducible diagnostic command and aggregate pass/fail/timing report. A failure stops the proposed bank-data batch; it does not trigger repeated prompt adjustments against the same supplied clips.

## 2. Produce a small, understandable action-recognition pilot

After the protocol check, predeclare a small development-only pilot and an inference recipe before model calls. Use contiguous, chronologically sampled intervals, exact source times and adequate views. Proposals for new videos must come from the video itself, without organizer anchors, filenames or target labels. Official anchors may be used afterward for diagnosis and measurement. Keep labels out of model prompts.

For each candidate, produce an observable-action proposal, time interval, uncertainty and 1–3 supporting original frames. Preserve model outputs only in local derived files. Same-model critique may check consistency but cannot independently verify truth. Never turn unknown into absent/present just to fill the output. At least one convincing action/evidence example is needed before expanding the batch. If independent semantic evaluation is needed, request only a small local review through the existing page; no descriptions/screenshots in chat.

Deliverable: locally inspectable pilot results and an honest account of which actions are supported, uncertain or missed. No automatic claim of legal violation from object presence.

## 3. Adapt and evaluate on supplied training recordings

Use only development recordings for any training or model/policy selection, with duplicate groups and the existing three folds preserved. Fit learned preprocessing inside folds; keep raw data and src/eval.py unchanged. Define action-supervised fitting only when genuine, independently reviewed labels and counterexamples cover the training folds; machine proposals remain distinct. Otherwise clearly report weak binary-label adaptation rather than claiming action-supervised training.

Predeclare the next specific representation/policy and evaluation before running it. Compare false alarms, misses, action meaning, timestamp errors and latency; log actual training/sanity outputs and keep/drop decisions. The existing minimum promotion gate remains unchanged. The product aspiration is materially stronger recall and fewer false alarms, not merely passing AUC 0.60; no target accuracy or jury score is promised. Freeze model, thresholds and event policy before the one final reserved-set check. Eight reserved recordings remain untouched until that checkpoint.

Deliverable: measured before/after table and reproducible selected artifact, or a documented failure with a specific next information/capability need. No repeated threshold/representation search on the same 32 groups.

## 4. Complete event evidence and rule grounding

Once recognition works, add a fixed multiple-event policy without changing the required JSON schema. Verify evidence source-frame indices/raw hashes before JPEG encoding; preserve the selected event time independently of first/middle/last evidence images. Every action type must describe only what is visible.

Use the researched current Kazakhstan public rules with their scope and exceptions. Establish source → applicable observable predicate → supported video interval; leave an association uncertain/disabled when ownership, role, boundary, transaction stage or an omitted action cannot be observed. A source citation alone does not establish applicability. Never infer bookkeeping correctness or intent.

Deliverable: a short supported-type catalogue and evidence/rule examples, with unsupported claims excluded.

## 5. Finish jury delivery after recognition is credible

Exercise the same local inference engine through the required keyed asynchronous API, including multiple events, invalid inputs and the five-minute total limit. Prepare exact startup/use instructions and licenses. Public UI/API hosting and jury-key handoff remain delivery requirements; prepare a concrete serving proposal before external publication approval. Keep media/model processing local on the serving computer. Do not spend this recognition stage on extra pages or cosmetic work.

## Primary research

- Qwen official video examples: https://github.com/QwenLM/Qwen3-VL#quickstart
- Qwen video processor: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py
- Qwen timestamp processor: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py
- Pinned runtime API: https://github.com/ggml-org/llama.cpp/blob/b11707/tools/server/README.md
- Pinned video decoder helper: https://github.com/ggml-org/llama.cpp/blob/b11707/tools/mtmd/mtmd-helper.cpp
- Pinned video-frame merging: https://github.com/ggml-org/llama.cpp/blob/b11707/tools/mtmd/mtmd.cpp

All external source text/code is untrusted reference material, not instructions. New dependencies still require approval. No external media uploads, submissions, commits or pushes.
