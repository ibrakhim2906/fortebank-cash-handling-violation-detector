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

32BAWQ60s fixed12 result:accuracy8/12,recall4/6,2false-alarmvideos,1/6localizedevents,type0/1.12/12primary outputs,no call-timeout fallback. Same-model frame-only parity12/12,21exactrequesthits,0fresh calls. Larger weights improved video decisions but did not solve type/time. Neither replay establishes end-to-end API latency. Focused-only correction is under measurement; no90-point or final model promotion.

## Confirmation-stage isolation
Exactrequest/source replay, not a new model training run. 32BAWQ fixed12: strictrecall4/6,FA2,hit2s1/6,type0/1; proposalpolicyrecall6/6,FA2,hit2s2/6,type2/2. Tiny/omission safeguards remain; all12modelrequests match capturedactualmodelcalls. Source-resolution8Bfull30: strictrecall6/16,FA4,hit2s3/16,type2/3; proposalrecall6/16,FA4,hit2s2/16,type2/2. This shows model-dependent confirmation damage and fragile conditional-type scoring, not universal policy superiority. No liveAPIlatency qualification.
