# Recognition handoff — 18 October 2026, 08:30 Asia/Qyzylorda

Start here before continuing recognition work. The user has about 2.5 hours remaining as of 08:26 local and wants measured recognition gains. The existing frontend and upload API work. This snapshot preserves current source, preparation work, reports and an isolated candidate; it does not promote an unmeasured recognizer. The current request is to commit and publish this work for another agent.

## Selected service and measured baseline

- Existing Runpod A40: `p7ayn21cpy6voe`; one model sequence at a time. Service source was left unchanged during the latest experiments. Do not launch simultaneous GPU evaluations.
- Model: Qwen3-VL-32B-Instruct-AWQ, vLLM 0.11.0 / Transformers 4.57.1, loopback model port 8002; public application port 8000.
- Selected recipe SHA-256: `d4c64e727fab0890048f332a03e90e1fae61343e3ddb39afce1bbad864140bfc`. `deploy/mission-best.env` records the selected settings: 16 overview images, proposal confirmation, storage veto, standard resolution, original workflow, frames-only perception. Desk/action experiments are off.
- Actual public API, all 40 supplied recordings: **27/40 correct verdicts**, TP 13 / FN 8 / FP 5 / TN 14; **6/22 annotated events within 2 seconds**, with correct types on those six. Forty primary-model clean jobs; worst clean upload 72.984 seconds. Two slowest-source repeats completed in 64.934 and 65.903 seconds.
- Fixed DEV30 comparison: **23/30 correct**, TP 12 / FN 4 / FP 3 / TN 11; **5/16 correctly typed events within 2 seconds**. These are the improvement denominators.
- Every supplied recording has been reviewed. The reserve is previously seen; there is no independent unseen-video evaluation. Never present conditional type accuracy 6/6 as complete classification coverage or claim 90–100 jury points.

Private evidence, available in the existing workspace/Pod but deliberately absent from Git: `data/processed/mission-checkpoint-3-storage-qualified-v1/`, `data/processed/mission-storage-veto-dev30-qualified.json`, and `data/processed/recognition-recovery-20261018/`. Do not upload bank footage, frame caches, keys, labels or request traces to make this repository self-contained.

## What was established

1. The exact-input audit reconstructs all 30 selected coarse request hashes, including JPEGs, prompt, schema and source timestamps. Inputs are upright, nonblank 720×394 images; timestamp rounding error is at most 0.0005 seconds. There is no demonstrated blank-image, rotation or request-mapping bug.
2. All 16 DEV event anchors have an overview sample within 2 seconds. Four nearest same-type timing misses are early by 8.5, 12, 25.4667 and 25.5 seconds; these pairs are not verified physical-event matches. A universal timestamp shift is unjustified.
3. Sparse overview coverage cannot establish omitted actions or offscreen absence. Three omission routes are correctly ineligible for these clips. Preserve these guards.
4. On three **public, nonbank** cabinet probes, a focused image region improves primitive opening/closing/static labels from 1/3 to 3/3 with the same model. This supports testing dynamic workspace grounding, not a bank-accuracy claim or fixed camera crop.
5. Expanded primary-context prompts timed out: v2 stopped after four timeouts; v3 stopped after two. Streaming showed a long inventory of negative findings instead of an empty observation list. Both owned controllers were stopped; no live model/service process was killed. Drop that primary-observer design.
6. An old experimental desk path can admit a normal/negative visible fact as an alert. That path is disabled in the selected recipe. It does not explain all selected errors and has not been changed in the live service.

## Completed comparisons to avoid repeating

| Fixed DEV30 recipe | Correct verdicts | TP / FN / FP | Typed events within 2s | Decision |
|---|---:|---|---:|---|
| Selected | 23/30 | 12 / 4 / 3 | 5/16 | Retain |
| Proposal refinement only | 23/30 | 12 / 4 / 3 | 3/16 | Drop |
| Source resolution only | 20/30 | 11 / 5 / 3, two degraded jobs | 5/16 | Drop |
| Broader storage context | 13/30 | 6 / 10 / 7 | 2/16 | Drop |
| Grouped predicate sweeps | 17/30 honest qualified verdicts | 14 / 2 / 10, two degraded jobs | 5/16 | Drop |

Grouped raw verdict agreement was 19/30, but two degraded jobs remain failures in honest scoring. Native-video opening/closing controls failed, so no bank native-video benchmark was authorized by that gate. Native maintenance finished and the original model/API were restored. Do not reopen these failures by silently relaxing the gates.

## Ready but unmeasured: workspace focus v1

Code-only package: `tools/recognition-recovery/workspace-focus-v1/`. Its README explains isolated application and tests. The patch is **unapplied to the serving source** and has **no bank recognition result yet**.

It preserves the original primary observer and adds a grounded detail pass, even when the primary already emits an event. Three overview images ground a workspace region dynamically; 13 cropped views and three original whole views retain source times. Supported detail types are cash left out, storage left open, personal items, food/drink and counting after client departure. Explicit positive predicates, before/onset/evidence ordering, original-view departure evidence and the existing storage veto remain required. Auxiliary failure is recorded as unresolved and cannot invent a clean result. Original full images remain API evidence.

The private candidate has **9 focused tests passed** and its isolated full-source copy has **1200 passed, 2 skipped**. Patch SHA-256: `6210b587f99e53a3616da702dc30c747f365807d43145f58220e4bcbb8c2a86f`. No GPU DEV30 run was launched for this candidate at handoff.

Next concrete work:

1. Verify the GPU is idle and no benchmark owns it. Reuse the existing isolated controller at `data/processed/recognition-recovery-20261018/phase-context-v3/run_isolated.py` and `controller_base.py`; adapt its flag/patch/helper paths to workspace focus and freeze the new hashes. Do not reuse the aborted context experiment's terminal state or results.
2. In a fresh isolated candidate, require OFF-mode exact selected DEV30 result/status/engine parity with zero fresh model calls. Then run ON-mode fixed DEV30 once, sequentially. Preserve every timeout, auxiliary abstention and false alarm. Do not inspect reserve errors to tune inference.
3. Compare actual typed coverage, TP, FP, failures and time, not only conditional classification or proxy points. If it improves the accepted tradeoff, freeze it and run actual public API qualification, all 40 clean videos plus the two slowest-source repeats. Reserve time for this final step. Otherwise retain the selected service and report the gap honestly.

Credentials, SSH gateway helpers and deployment process details remain only in ignored setup directories. The exact local API-key path is documented in `docs/jury_handoff.md`; never print or commit the value. Consult `reports/progress.md`, `experiments.md`, `docs/plan.md`, `docs/submission.md` and `docs/release_rule_matrix.md` for provenance, resource limits and remaining release gates. Do not edit either frozen evaluator (`src/eval.py`, `scripts/eval_video.py`), raw data, or silently activate legal rules.
