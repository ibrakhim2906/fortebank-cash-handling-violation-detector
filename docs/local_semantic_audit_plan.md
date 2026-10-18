# Proposed local semantic audit — whole cash-operation workflow

Status: prepared for the human's approval under AGENTS.md rule 12. No new weights have been downloaded, no bank videos have been newly processed and no model has been promoted by this plan. The human's request concerns all visibly incorrect cash-operation actions and omissions, not only transfers.

**Later scope update:** this is now an optional local-model proposal, not the active review blocker. The human subsequently authorizes direct assistant review of the supplied dataset and explicitly releases all formerly reserved examples for that review; see `docs/case_review_scope.md` and the completed `reports/case_video_review.md`. The reserved set described below reflects the earlier proposal and is no longer independent. No 8B download was necessary for the completed visual review.

## Purpose and scope

Produce a private chronological catalogue of what a qualified local model observes across all 32 development recordings, which procedural concerns each observation could support, what context/exceptions remain unresolved, and which recordings fail processing. Preserve all eight reserved recordings, duplicate grouping and original annotations. The cloud assistant will not see frames, per-video descriptions or per-video predictions.

The catalogue contains proposed observations, never self-verified gold labels. It does not require sixteen transfer labels. It does not train or fine-tune a model. Official binary labels and eligible positive anchors will be used only by a separate scorer after label-blind inference outputs are fixed.

## Exact bounded model candidate

Reuse the already installed standalone CPU llama.cpp build 11707 (`1268edac5`); the pinned source supports Qwen3-VL. No new Python dependency or runtime is proposed. An actual 8B load has not been tested. The existing 2B server/manifest and service remain unchanged; use a separate local-only auditor process/manifest and stop that auditor if memory pressure threatens the existing service.

Official model: [Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/5bcaa9cc11a22c7ddeade75f0c84b51e43b3f7ab), Apache 2.0, revision `5bcaa9cc11a22c7ddeade75f0c84b51e43b3f7ab`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | 5,027,784,800 | `67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2` |
| `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | 1,159,029,824 | `ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38` |

Total public weight download: 6,186,814,624 bytes, approximately 6.19 GB / 5.76 GiB. Use official revision-pinned resolve URLs, preserve source/license/hash provenance and refuse overwrite of different files. The inspected laptop has 18 GiB RAM; weight-file size is not peak runtime memory. CPU speed, context fit and visual adequacy remain unmeasured. No GPU rental, hosted inference, API credit spending or external upload is included.

## Before a full development run

1. Verify asset/license hashes and local-only endpoint; record actual model/runtime/template identity and limits. Download public weights before inference, then use local file paths with network access disabled for the auditor where available. Do not change the frozen observer or server used by earlier experiments.
2. Freeze the frame sampling, aspect-preserving detail policy, prompt/schema, output budget and timeout before testing. Preserve source timestamps and exact frame indices/hashes. Requests contain only selected visual inputs/times, never labels, anchors, filenames or previous proposed answers. Distinguish sampled coverage from continuous observation; absence claims require sufficient coverage.
3. Run fixed synthetic color/order/motion controls including reversal and stationary negatives. Validate completion, stable source-time mapping and fresh-process repeatability. If controls fail, stop rather than expanding the failed recipe to all bank videos. Report the failure and one bounded explanation; do not silently keep changing prompts.
4. Run a small deterministic label-blind development pilot only if synthetic qualification passes. Check resource use, decoding/JSON/evidence integrity, unresolved cases and full chronological coverage. This checks operation, not semantic action accuracy; model self-agreement must not be used as proof. Protect any separately blind transfer-pilot pool from training access.
5. Freeze the passing recipe, then process all 32 development videos with explicit time/memory limits, durable checkpoints and a failure denominator. A slow offline audit does not demonstrate the required five-minute serving limit. No automatic deployment or supervised fitting follows the audit.

## Private outputs

Use `data/processed/semantic-audit-v1/` with directory permissions 0700 and files 0600. Preserve a manifest, immutable source/model/recipe fingerprints and an append-only `observations.jsonl` plus aggregate summary. Per interval, record actual frame/time references, crop mapping, sampled coverage/gaps, literal proposed actions, uncertain objects/roles/transaction stage, possible procedural concern, source candidate, prerequisites and exceptions. Include other/unclear actions rather than forcing a closed taxonomy.

Review cash/document presentation and movement, acceptance versus payout, counting and machine use, writing/stamping/returns, storage transitions, unfinished-service departures and other relevant conduct. A normal transfer is not a violation. Omission, prohibited-item, amount, identity, ownership and bookkeeping claims require evidence the camera may not supply.

Keep all per-video semantics and evidence local. Reuse the existing private review page when displaying results; do not publish that review interface. Human confirmations, if obtained, go in a separate append-only review journal. Report aggregate completion/failure counts and official-label verdict agreement to chat, with no invented action-type accuracy. Preserve uncertain outputs as uncertain; do not silently equate an unreadable recording with `no_violation`.

## Acceptance and next decision

Success means a complete, provenance-preserving proposed audit and measured limitations, not automatic semantic correctness or a promised jury grade. After it runs, decide from actual evidence whether pretrained observation suffices, sharper views are needed, or a specific compact recognizer needs independently reviewed examples. Qualify every rule mapping before serving it as a supported violation type. Keep the eight reserved videos for one final frozen-policy evaluation.
