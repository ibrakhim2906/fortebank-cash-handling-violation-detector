# ForteBank video service — working submission draft

Status: MISSION in progress. No qualified recognition score or final jury handoff yet.

Current development configuration (18 October local): pinned Qwen3-VL-32B-Instruct-AWQ on the existing A40, vLLM0.11.0/Transformers4.57.1, loopback8002, standard16-image sweep, frame-only preprocessing, proposal policy. The local64-image/32k-context candidate has passed software/coverage checks but is **not yet deployed or quality-qualified**. Default code remains16images. Source/runtime identities are preserved with each benchmark; the public checkpoint is frozen until completion.

Measured results, not a jury forecast:

| Scope | Video accuracy | Positive videos found | False alarms | Events within2s | Correct types among localized events |
|---|---:|---:|---:|---:|---:|
|30previously-seenDEV, fixed32Bproposal recipe |20/30|12/16|6|5/16|5/5|
|10previously-seen regression-reserve, public API checkpoint1 |4/10|1/5|2|1/6|1/1|

The type fractions are conditional on localization: correct typed event coverage is5/16onDEV, not100% recognition. DEV produced19events,14unmatched to the supplied anchors. Five false-alarm events were safe-open predictions and one was door-open. The engineering proxy is43.3854with missing gates at zero; it is not an official score. This candidate fails the release targets and has not earned a quality-promotion tag.

All14clean public checkpoint requests (10reserve+4DEV) completed through the primary model without failed jobs; worst51.1425seconds including upload/polling. This is a partial API sample; the two longest-video repeats and full40-video API qualification are still outstanding. FourDEV API calls reproduce the exact benchmark request hashes and answers. Augmentation checks are in progress. The reserve is not independent and no per-video reserve errors were inspected for tuning.

Actual GPU network-denied proof remains incomplete: two model processes pass the kernel guard, while the EngineCore audit rejected its same-machine TCPStore connection. A tested correction recognizes only exact namespace-owned interface addresses, not arbitrary private networks; it requires the next controlled model restart. No offline/Tech credit is claimed from this failed attempt.

Public development URL: https://p7ayn21cpy6voe-8000.proxy.runpod.net . API uses X-API-Key, delivered separately; never committed here. Upload MOV/MP4, poll the returned job. Prototype availability does not establish accuracy. The existing A40 Pod is the only approved host; $0.59/hour, $15 total including storage, auto top-up off. Preserve serving time for jury testing.

The service separates durable upload/jobs, source-time video decoding, pretrained observations, explicit physical predicates/rules, a self-hosted vision auditor and source-aligned JPEG evidence. API fields remain the supplied contract. CPU fallback works locally but its numeric anomaly model is weak and lacks action-specific grounding. No face identification/audio or hosted AI inference is used by the service.

## Component provenance

| Component | License | Pin and current status |
|---|---|---|
| Qwen3-VL-32B-Instruct-AWQ | Apache-2.0 | [Quantized model card](https://huggingface.co/QuantTrio/Qwen3-VL-32B-Instruct-AWQ), revisionc2d44b1376c73fa120fdb4e216c2ef61c1945498; official parent Qwen/Qwen3-VL-32B-Instruct also Apache-2.0. Built-in architecture, AWQ-Marlin/float16; no downloaded executable model code. Hash-verified20.53GBassets in deploy/qwen3-vl-32b-awq.json. Current development candidate; results above remain insufficient. |
| Qwen3-VL-8B-Instruct | Apache-2.0 | Official [model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),revision0c351dd01ed87e9c1b53cbc748cba10e6187ff3b. Card checked17October2026. Full LFS hashes and small-file Git blobs in deploy/qwen3-vl-8b.json. BF16 installed and hash-verified;4/4syntheticimagecontrols pass. Full30developmentbaseline:recall2/16,3falsealarms; insufficient recognition. |
| vLLM | Apache-2.0 | Version0.11.0,isolated Pod environment; [official support](https://docs.vllm.ai/en/v0.11.0/models/supported_models.html) includes Qwen3VL. Loopback only,no prompt embeddings/remote models/user model configs. Not a project dependency. Installed and version-verified;CPUfallback/APIoperate. GPUkernel-blockedproof remains pending. |
| Transformers | Apache-2.0 | Pinned4.57.1 in the separate vLLM environment. |
| YOLOX-S / OpenCV Zoo | Apache-2.0 | Source47534e27c9851bb1128ccc0102f1145e27f23f98,weights SHA c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063; existing verified local installation. |
| OpenCV | Apache-2.0 | Existing approved headless runtime in uv.lock. |
| FastAPI / Starlette / Pydantic / uvicorn | MIT / BSD-3-Clause / MIT / BSD-3-Clause | Existing locked project packages. Full installed-version/license audit remains pending. |
| NumPy / SciPy / pandas / scikit-learn | BSD-3-Clause | Existing locked project packages. |
| Python / PyTorch / NVIDIA CUDA system runtime | PSF / BSD-style / NVIDIA runtime terms | Record actual notices. GPU driver/runtime is not an Apache model and must not be mislabeled; transient dependency audit pending. |
| Existing FFmpeg8.1.3 LGPL build, OpenH264 | LGPL-3.0 / BSD-2-Clause | Previously approved standalone decoder; bundled notices preserved. Used for local robustness generation. Do not claim all libraries are MIT/Apache. No new GPL model is introduced. |

Earlier inactiveGGUFweights were hash-verified and reclaimed for space;licences/manifests/exactredownloadURLs and historicalfailed/unqualifieddiagnostics are preserved. The official8BBF16weights remain the rollback model. A license-table-complete scoring gate has not passed merely because this draft exists.

## Validation and limits

See reports/validation_protocol.md. All40 supplied unique recordings were previously inspected; the new10-video reserve is regression-only,not independent. Only three team-specified setups exist; minor groups are too small for confident per-type conclusions. Team human-reviewed types remain distinct from organizer-original weak labels. Unknown/offscreen actions never establish absence. A lit laptop screen does not prove an unlocked session. Generic object detections do not establish cash, personal ownership or legal authorization.

Legal source verification and visual applicability are separate. Current source research in docs/action_rule_candidates.md and docs/case/legal.md prevails over the supplied seed. No invented bank policy, categorical key-removal/food ban, electronic-ID failure or authenticity conclusion is allowed. Final supported types, per-type results, source mappings, errors, offline test and measured latency will be filled from actual MISSION evidence.

Model comparisons performed: Qwen/Qwen2.5-VL-7B-Instruct,revisioncc594898137f460bfe9f0759e9844b3ce807cfb5,16.60GB; QuantTrio/Qwen3-VL-32B-Instruct-AWQ,revisionc2d44b1376c73fa120fdb4e216c2ef61c1945498,20.53GB. Both exact cards declare Apache-2.0; the quantized parent Qwen/Qwen3-VL-32B-Instruct also declares Apache-2.0. Metadata/LFS hashes in deploy/. The quant card explicitly requires vLLM≥0.11.0 and config is4-bitAWQ/group128/GEMM,visual module unquantized. Only built-in architectures and safetensors are accepted; no downloaded Python/pickle. 7Bfailed2/4syntheticcontrols and neverranbankvideo;inactiveweightsreclaimedwithverifiedmanifest.32BAWQfixed12diagnostic:recall4/6vs8B2/6,FA2vs3,hit2s1/6,type0/1;notpromoted asanoverallrubricimprovement. See reports/model_comparison.md. Native8Bcolor/timecontrols pass,butmotioncontrols fail;thatservicebranchwasremoved. No strongrecognizer isqualified.

Prepared comparison: QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ,revisiona5ea10732e2c9330957864ece3ec66d806a4b00d,officialparentQwen/Qwen3-VL-30B-A3B-Instruct revision9c4b90e1e4ba969fd3b5378b57d966d725f1b86c. BothpinnedcardsApache-2.0. Hashverifiedweights;4/4syntheticcontrols. Fixed12strictpilotrecall1/6,FA1,hit2s0/6;dropped. Manifestdeploy/qwen3-vl-30b-a3b-awq.json preservesredownloadprovenance;weightsreclaimedonlyaftermodelstops. Installed-environmentlicenseinventory238distributions,allwithmetadata/notices,recordedprivately;not yet a verified license-complete gate.
