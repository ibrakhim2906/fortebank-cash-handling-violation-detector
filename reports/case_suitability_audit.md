# Case and implementation audit — 17 October 2026

**Subsequent authorized review:** the human clarifies permission for direct AI interpretation of the supplied initial dataset and then explicitly chooses to release the eight reserved recordings after the independent-test tradeoff is explained. The technical findings below remain valid, but references to untouched reserved footage describe the audit's earlier snapshot. The all-supplied-data visual review is recorded separately in `reports/case_video_review.md`; no independent holdout remains afterward. Hidden jury videos and local submitted-app inference remain protected.

## Conclusion and corrected scope

The case is contextual visual screening of possible cash-discipline violations, including incorrect actions and observable omissions across cash operations. It is not restricted to transfers. A transfer, writing, phone-like object or counting motion is an observation; it is not automatically a violation. The human explicitly corrected this scope during this audit.

The suitable starting architecture is sharp chronological video → one qualified local visual observer → explicit transaction/rule conditions and exceptions → multiple localized events with original evidence → the required API. A universally fixed ID/cash/count/paperwork sequence, generic activity logits, a mandatory handover classifier and mandatory fine-tuning are not case requirements.

The current implementation is not yet a qualified violation recognizer. The working API and tests establish useful infrastructure, while the delivered engine still emits at most one generic event without established event semantics or applicable rule grounding. No estimate of 95–100 jury points is supported.

## What was slowing progress unnecessarily

The original official labels already permit video-level evaluation and a limited positive-anchor diagnostic. Sixteen newly reviewed action examples are a gate for the separate supervised object-transfer experiment, not a prerequisite for a label-blind pretrained observer or its first binary evaluation. Action-type correctness, rule applicability and exhaustive event coverage still need independent review; binary labels cannot supply them.

Do not train a model to reproduce its own guesses as verified truth. Start with pretrained inference and diagnose failures before deciding whether targeted examples, a small head or fine-tuning are warranted. Do not acquire a large external corpus before identifying a missing capability.

## Data and image-path checks

- Frozen protocol: 32 unique development recordings, 17 positive and 15 negative; eight reserved recordings remain untouched. Exact duplicate aliases stay together. The conflicting 57/68-second duplicate annotation remains excluded from timing evaluation.
- Development playback is 1,019.677 seconds (16.994616666666666 minutes); median 23.0505 seconds; range 11.634–78.734 seconds. These are metadata measurements, not completed semantic review. Pauses, replays and adjudication add time.
- Header checks find no supported rotation/aspect-metadata bug: development headers report 464×848, square pixels, zero orientation. Do not rotate the scene based on an unproven diagnosis.
- The MoViNet letterbox reduces the real scene to approximately 94×172 pixels and adds about 45.35% black padding. Small item and finger details can be lost. Numeric and MobileNet preprocessing also alter the original scene geometry. Better context cannot recover discarded visual detail.
- Most development sources use variable frame durations. Prefer decoded source timestamps. No systematic sampled-tail overrun was demonstrated by the header/PTS reconstruction.
- The common evidence sampler lacks the sequential-decode fallback already added to one Qwen path. Evidence must preserve actual decoded times and source correspondence; a requested seek time alone is insufficient.

## Confirmed runtime/model limitations

- Installed llama.cpp build 11707 has no available GPU device. The launcher explicitly selects CPU and disables vision-projector offload. API credits do not establish a usable GPU machine or rental budget; the human confirms credits only.
- The existing 2B observer failed the motion gate (1/3 controls; cache-off follow-up 2/6), although color and coarse first/last checks worked. A wholly missing image-input explanation is unsupported. Cache behavior is not an established cause of the separate spatial prediction drift.
- The old observation path uses four short windows and reduced still images. It is not full-timeline observation. The native review helper samples only the first approximately eight seconds of a requested interval; it must not be reused unchanged as a whole-video scanner.
- The old schema does not adequately represent event boundaries, evidence references, cash/document distinctions or rule conditions. A short output allowance can truncate chronological findings. Check completion reason and actual context/output budgets.
- A stronger 8B model is an untested candidate. Larger weights, a GPU or a more detailed prompt alone do not establish cashier-action understanding or the five-minute service limit.

## Confirmed serving and evaluation gaps

- A synthetic deadline test reproduced success being published after the deadline when final validation consumes the remaining budget. Recheck immediately before publishing success.
- Evidence schema validation accepts marker-wrapped non-JPEG bytes and unbounded timestamps. Actual decoding, duration bounds and source-frame correspondence are needed before accepting a new observer's evidence.
- One worker queues uploads under a shared 300-second per-job budget, with fresh analysis processes. Measure burst uploads and model startup as well as one warm isolated request. The Windows bridge's upload buffering occurs before the direct API clock starts; its per-request timeout is not the asynchronous job deadline.
- Existing localization statistics include candidates from positive videos called negative. They are diagnostics, not end-to-end localization success. Report video confusion, failed/unprocessed denominators, multiple-event behavior, event types, rule applicability and evidence quality separately.
- Repeated selection on the same 32 development videos creates optimism risk even when fold fitting is correct. Keep the reserved set sealed until the final policy is fixed. Do not claim another-angle robustness from this camera.
- Never put filenames, official labels or positive anchors into observer prompts. A label-blind scan must run before a separate local scorer joins the official labels. Few-shot sources and all their aliases/crops must be excluded from the corresponding evaluation partition.
- The current bridge and launcher are development conveniences with machine-specific assumptions. Public availability and portable deployment remain unverified.

## Workflow/rule coverage to investigate locally

Candidate coverage includes cash acceptance and payout; document presentation/handling; manual counting and machine interaction; writing/stamping and returns; cash storage; workstation presence/departure; relevant item storage and transfers between cash workers; and other unclear actions. This is an audit checklist, not an assertion that these actions or violations occur in any supplied recording.

The current official NBK resolution №231 text was fetched again during this audit: `is_actual=true`, exact UTF-8 text SHA-256 `1798a349b2326ba4c32a007a62db532ea6de906e71f0249a9de8dc81aa915de8`, matching the preserved source catalog. Use [the catalog](../docs/action_rule_candidates.md) for scope and exceptions. In particular, intact packaged-banknote payouts have a counting exception; sparse footage cannot prove a skipped action; absent visible paper cannot establish missing electronic/accounting records; and phone/bag appearance does not establish ownership or a prohibited item.

## Local semantic review and access boundary

At the time of the technical audit, direct assistant visual review had not occurred. After the assistant explained its cloud-hosted vision, the human explicitly clarified that they permit AI interpretation of the supplied initial dataset while keeping the submitted application's processing local. This is their interpretation, not independently verified organizer wording. Follow that scoped authorization for the 32 development recordings only; preserve all eight reserved recordings and never transfer hidden jury videos. Attribute assistant observations honestly and retain their uncertainty. See [the review scope](../docs/case_review_scope.md).

Prepare a private chronological proposed-observation journal on all 32 development recordings with source/frame hashes, timestamps, coverage and gaps, observable facts/unknowns, transaction-context uncertainty, conditional rule references and exceptions. Keep it separate from verified labels and supervised training. Independent review remains necessary for validated semantic-accuracy claims; same-model critique is not independent truth. The [bounded local 8B audit proposal](../docs/local_semantic_audit_plan.md) remains an optional unapproved alternative; no weight download is needed for the newly authorized direct assistant review.

## Human-provided operational information

The human reports only 30 teams advance to pitching and that the available $50 is NVIDIA/OpenAI API credit, not rental cash. This is attributed team information, not newly verified organizer wording. Automatic API evaluation is already confirmed by the supplied brief. No rental, credit spending, account use, external data transfer or new dependency was authorized during this audit.

## Verification and limits

Read-only source/header/protocol/runtime inspection, official public-source retrieval and independent code reviews were performed. The API/evaluation/dispatcher subset returned `49 passed in 7.25s`; the deadline and evidence-validation weaknesses were separately reproduced on synthetic inputs. No model, feature, frozen split or decision threshold changed. No new bank inference, training, sanity score or before/after detection score exists from this audit. Existing dirty working-tree contents and raw files were preserved.
