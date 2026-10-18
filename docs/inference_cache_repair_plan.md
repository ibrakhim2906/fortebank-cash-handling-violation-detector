# Bounded inference-cache diagnosis

The frozen spatial trial improves development AUC, but two rereads of identical source frames/JPEG views produce descriptors that flip the saved classifier's verdict relative to its training cache. The unchanged native-video synthetic gate then returns valid responses but passes only the forward-motion control; reversed and stationary controls fail. Bank-video expansion is blocked. Neither finding establishes the cause.

Current status: cache-off fails the fixed native-motion controls; it does not resolve the separate spatial cache-history/drift hypothesis. Color input, coarse first/last color order and the exact FFmpeg decoder path pass their separate diagnostics. Square-motion and cashier-action understanding remain unestablished. The native bank-video pilot is unrun and blocked; eight reserved videos remain untouched.

The pinned llama.cpp server documentation says prompt caching defaults to enabled and can change numerical results through prefix/batch reuse. Test this specific runtime hypothesis before changing models, questions, labels or thresholds.

## Controlled synthetic test

Preserve the failed gate report and exact source snapshot. In a separate script/output directory, use the same verified local model/runtime/decoder, temporary native-video server, fixed synthetic clips, prompt, schema, seed 42, temperature zero and budgets. Run forward, reverse and stationary controls twice. The sole request-policy change is explicit `cache_prompt: false`. Require all six correct responses and agreement between repeats. Record any returned `timings.cache_n` separately; missing cache metadata cannot prove that prefix reuse was disabled. No supplied footage or labels are read. No prompt adjustments, larger-model downloads or repeated tests for a favorable result.

This is a mechanism diagnostic, not permission to bypass the failed bank-media gate. A pass supports a cache-policy repair; it does not establish semantic accuracy or exclusive causality. A failure stops this proposed repair route and is reported honestly.

## Only if the controlled test passes

Write the explicit no-prefix-cache policy into both the synthetic gate and native pilot request recipes, with version/source fingerprints and tests. Preserve the original failed report/source. Run the revised gate into a fresh directory and require all three controls; the pilot must verify that exact policy/provenance before any bank-media call. Then run only the already planned immutable three-task local pilot. All action outputs remain unverified proposals; no bulk expansion or legal mapping follows automatically.

The scored spatial V3 extractor and caches remain frozen and withheld. A spatial cache-policy correction would be a new extraction version requiring fresh development descriptors, fixed CV and sanity checks. Do not silently reuse or overwrite V3 features, retune thresholds, fuse representations or touch the eight reserved videos. Decide that bounded repair from actual diagnostics and pilot results, not a wider score search.

## Actual outcome and next input-path check

The original native motion gate passes **1/3** controls in **14.865009 seconds**, with no operational error. The cache-off test then completes all six requests without operational errors in **24.005464 seconds**. All six report `timings.cache_n = 0`, but every response is `left_to_right`: only the two forward controls are correct (**2/6**). Repeats agree, but the test fails. **Do not apply the proposed cache-policy repair, start V4, or reopen the bank pilot from this result.** Explicit cache-off does not restore these controls; an exclusive cause is not established. Reports: `data/processed/temporal-protocol-native/temporal_protocol.json`, `data/processed/local-cache-behavior/cache_behavior.json`.

The separate input-path check completes **4/4 correct** responses in **15.454847 seconds**: both red/blue native videos and matching still JPEGs return the expected color. Prompt, color schema, seed, temperature, cache-off setting and budgets are fixed. Input hashes and usage/cache counters are preserved in `data/processed/native-visual-input-v1/visual_input.json`. Color-dependent input delivery is verified; temporal order, motion and action understanding are not.

Exact reproduction of the pinned runtime's FFmpeg buffer-decoder command also passes all three synthetic cases: exactly four decoded frames and the expected known order, with successful exits despite seekback warnings. Report: `data/processed/native-decoder-reproduction/decoder_reproduction.json`. Correct decoder output does not prove that the full model uses or understands every frame.

The fixed native RRBB/BBRR color-order controls complete **2/2 correct** first/last-color responses in **10.037758 seconds**, both reporting `cache_n = 0` and 180 prompt tokens. Report: `data/processed/native-frame-order-v1/frame_order.json`. Coarse first/last order is demonstrated, so a claim that the entire temporal input path is broken would be unsupported. This does not repair the failed square-motion gate or authorize a bank-video batch. All diagnostics above use zero bank or holdout recordings. Keep the separate spatial score/drift findings and failed reports unchanged; do not change model, prompts or thresholds against supplied clips to obtain a favorable result.
