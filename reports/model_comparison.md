# Auditor comparison — MISSION development evidence

No model is promoted as a90-point solution. All supplied recordings were previously inspected. Model selection uses development feedback and remains exposed to overfitting; the regression reserve cannot establish independence.

Frozen comparison: existingA40,seed42,same16-image sweep and focused confirmations,source-time images,same schema and catalogue. Full model/runtime and exactrequest hashes prevent cross-model cache reuse. Evaluation-only blind perception replay saves CPU; latency and Product require actual end-to-end publicAPI replays.

| Recipe | Scope | Accuracy | Recall | False-alarm videos | Event hit±2s | Type accuracy among hits | Measured API worst |
|---|---|---:|---:|---:|---:|---:|---:|
|Qwen3-VL-8B BF16,v1|30development|.400|2/16|3|1/16|1/1|107.918s|
|Same8B exactrequest replay|12fixed diagnostic development clips|5/12|2/6|3|1/6|1/1|not measured by replay|
|Qwen2.5-VL-7B BF16|Stopped before case footage:2/4controls pass|not measured|not measured|not measured|not measured|not measured|not measured|
|Qwen3-VL-32B AWQ|12same clips|pending|pending|pending|pending|pending|pending|

The initial30API requests yielded28results and2transient polling errors. Failures remain in the denominator. All28traces used the primary8B; a numeric fallback does not qualify as honest pretrained recognition. Classification1/1is conditional on one localized event and must not be described as100%type coverage. The12exactreplay outputs equal the priorAPI results,19cachehits,0newmodelcalls; this validates the cache,not detection quality.

Full source manifests and licences are in deploy/. Qwen2.5-VL-7B and QuantTrioQwen3-VL-32B-AWQ cards declareApache2.0,including the quantized32Bparent. No hosted AI API, external labeling service, filenames, source hashes or gold labels enter the service inference path.
