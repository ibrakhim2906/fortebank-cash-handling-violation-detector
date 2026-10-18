# Spatial inference consistency diagnostic

The frozen spatial V3 classifier improves development AUC to 0.615686, but identical source images produce a different descriptor vector on later extraction and flip the saved verdict. Promotion remains withheld. The separate failed native-motion cache test does not resolve this still-image finding.

Within the human's approved autonomous recognition work, run one bounded, unscored diagnostic on the already selected longest development recording. Preserve the frozen V3 code, caches, classifier, prompts, sampling, schema, seed and threshold. Do not read any reserved recording or fit a model.

Compare explicit `cache_prompt: true` and `false`. For each policy, use the original verified local server profile and extract all four fixed windows in three conditions: a fresh process; after one fixed synthetic three-image blue conditioning request; and after restarting the process. This gives exactly 24 recording requests and two synthetic requests, sequentially. The live numeric API and review UI do not call this model; no other inference job may use port 8003 during the diagnostic.

Record actual request hashes, normalized hashes excluding only the cache flag, source/JPEG identities, per-call cache counters and timings. Store model observations privately in a fresh derived directory; emit only aggregate identity, feature and verdict-difference counts. The explicit cache flag is a diagnostic override, not an unchanged V3 feature recipe. Original evidence and failed checks stay intact.

If caching enabled drifts while disabled remains stable, that supports request-history sensitivity under these tested conditions. If both remain stable, the historical cause remains unresolved. If disabled also drifts, that repair is insufficient. Missing cache counters weaken any mechanism conclusion. None of these outcomes independently establishes semantic accuracy or authorizes promotion. Any actual serving-policy repair needs a separately versioned extraction, fresh development training and sanity checks before a deployment decision.

The pinned runtime documents cache-related numerical variation and compares image hashes/token counts during prefix matching. Do not claim that it reused the wrong image.

Sources: [runtime documentation](https://github.com/ggml-org/llama.cpp/blob/b11707/tools/server/README.md), [media-prefix comparison](https://github.com/ggml-org/llama.cpp/blob/b11707/tools/server/server-common.cpp#L697-L740), [request processing](https://github.com/ggml-org/llama.cpp/blob/b11707/tools/server/server-context.cpp#L3387-L3454).
