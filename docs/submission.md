# ForteBank video service — working submission draft

Status: measured recognition remains below release targets; checkpoint3 API/runtime qualification is complete. Final jury handoff remains open.

## Current licensing and submission audit — 18 October 2026, 04:38 UTC+05:00

The timestamp above identifies the preserved licence inventory audit. The operational update below supersedes earlier checkpoint1, pending 64-image comparison and failed offline-attempt statements retained as history. No recognition promotion follows from documentation or inventory completion.

### Current operational evidence — 18 October 2026

- Selected checkpoint3 recipe: `d4c64e727fab0890048f332a03e90e1fae61343e3ddb39afce1bbad864140bfc`, the existing A40/32B AWQ service with 16 images, 16384 context, 60-second model-call ceiling, proposal policy, selective storage veto and frame-only processing. See `deploy/mission-best.env` and `data/processed/mission-checkpoint-3-storage-qualified-v1/provenance.json`.
- Checkpoint3 completed **40/40 clean uploads through the primary recipe, zero failed jobs, and two distinct qualified slowest-source repeats**: 42 unique actual job IDs. Clean worst time was **72.984294733 seconds**; repeats were **64.933887418** and **65.902824726 seconds**. Recognition on all 40 previously reviewed recordings was **27/40 correct verdicts, 13/21 positives detected, five false alarms and 6/22 events within ±2 seconds**. This is not independent-test performance. Exact aggregates and timing: `data/processed/mission-checkpoint-3-storage-qualified-v1/{all40-aggregate,actual-api-runtime}.json`.
- Actual selected-GPU network-denied processing **passed**, with engine `mission-v1`, `audit_complete=true`, one primary request and **27.481672797 seconds**, on the same selected recipe. Evidence: `data/processed/mission-selected-gpu-offline-proof.json`. This establishes the tested processing path, not recognition quality or licence compliance.
- Code publication is recorded at [the team repository](https://github.com/ibrakhim2906/fortebank-cash-handling-violation-detector). Local `origin` is configured; the separate publication checkout records `main` and `origin/main` at `c9619f7917d20895958a370a0fcab82c1cf20684`, with an update-by-push reflog entry at 06:44:43 Asia/Qyzylorda. Its `mission-best` tag peels to selected-source commit `a4cf69e276391ca21e170134eccbc130567f60c7`. These are saved publication/local Git facts; this correction made no fresh remote request.
- **Licence table verified against the organisers' rule (18 October 2026).** The 07:24 local recheck verifies all 238 distribution records, 316 notices and 319 archive file members with zero integrity failures (`data/processed/recognition-recovery-20261018/licenses/{audit.md,verification.json}`). Under the licence policy below, `scripts/verify_license_policy.py` classifies every record with zero failures; `license_table_verified=true` in `data/processed/license-policy-verification.json`. With the existing network-denied proof, Tech is 10: checkpoint3 proxy 55.885 → **65.885** (`data/processed/checkpoint3-rubric-license-policy.json`). Not a recognition gain or a jury score. The full 280-transform robustness qualification remains outstanding (Robustness 0).

### Licence policy

The rule is the organisers' requirement (R50–R56, `docs/case/requirements.md`): open-source code and models whose licences permit commercial use, each recorded. The team's earlier Apache/MIT/BSD-only wording for every library is replaced by this rule (human decision, 18 October 2026).

- **Model weights:** Apache-2.0, MIT or BSD only. Selected: `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` @ `c2d44b1376c73fa120fdb4e216c2ef61c1945498`, Apache-2.0 on the manifest and the pinned model card.
- **Supporting runtime:** permissive licences (MIT, BSD, Apache-2.0, ISC, Unlicense, CC0, Zlib), plus PSF, MPL-2.0 and LGPL terms, and NVIDIA CUDA runtime terms for `nvidia-*` distributions only.
- **Fails:** GPL/AGPL/SSPL, non-commercial terms, proprietary terms outside the NVIDIA CUDA runtime, or any distribution whose licence can't be identified.
- **Scope:** the delivered service, meaning the API service environment, the model-server environment, CPython, the FFmpeg decoder and the model weights. The host GPU driver and the `runpod/base` image are platform components provided by Runpod and are outside this table.

| Family | Count | Components outside plain MIT/BSD/Apache |
|---|---:|---|
| Permissive | 212 | Includes ISC (dnspython, shellingham, prometheus-fastapi-instrumentator), Unlicense (email-validator), MIT-CMU (pillow), 0BSD/CC0/Zlib terms in NumPy |
| PSF | 6 | CPython 3.12.3, typing_extensions 4.16.0 (both envs), aiohappyeyeballs 2.7.1, matplotlib (PSF-based), regex 2026.9.29 (Apache-2.0 AND CNRI-Python) |
| MPL / LGPL | 8 | certifi 2026.7.22 (MPL-2.0, both envs), tqdm 4.70.1 (MPL-2.0 AND MIT, both envs), frozendict (LGPL-3.0), soxr (LGPL-2.1-or-later), pycountry (LGPL-2.1-only), FFmpeg/ffprobe decoder (LGPL-3.0-or-later). OpenCV's bundled FFmpeg is LGPL-2.1, recorded in its third-party notice |
| NVIDIA CUDA runtime | 14 | cuBLAS, CUDA runtime, CUPTI, NVRTC, cuFFT, cuFile, cuRAND, cuSOLVER, cuSPARSE, cuSPARSELt, cuDNN, nvJitLink under NVIDIA SDK terms; NVTX (Apache-2.0) and NCCL (BSD-3-Clause) ship with NVIDIA classifiers |

Three records without usable metadata are resolved from their preserved notice text: xformers 0.0.32.post1 (BSD-3-Clause), openai-harmony 0.0.8 (Apache-2.0) and setuptools 79.0.1 (MIT). The per-row result for all 238 records is in `data/processed/license-policy-verification.json`. LGPL components are used unmodified as shared libraries or a separate executable. Their notices are retained in the archive, and source is available from the upstream projects pinned in the manifests.

### Scope, evidence and compatibility status

- Exact Pod inventory: **238 installed Python distributions**: 81 in the service environment and 157 in the isolated model environment; **203 distinct name/version pairs**. Inventory SHA-256: `4fd35becf22234c56eac1de859f88e54b697e876ec9839ab1f6afbd32d2e96f2`. The private evidence is `data/processed/mission-v6-bootstrap/pod-installed-licenses.json`; it records each distribution's declaration, version, project URLs and 303 notice-file path/hash entries. This is an installed-package inventory, not a claim that every package participates in video inference.
- Additional notices were read directly from the Pod without restarting the service or GPU model: setuptools 79.0.1 is MIT; openai-harmony 0.0.8 is Apache-2.0; the CUDA sparse-LT notice is present outside `.dist-info` at `/opt/vllm/lib/python3.12/site-packages/nvidia/cusparselt/LICENSE.txt`. Exact texts/hashes are preserved in `data/processed/mission-v6-bootstrap/pod-missing-license-notices.json`.
- Local preparation environment inventory is separate: 109 distributions in `data/processed/mission-license-audit-local.json`. Do not substitute it for the Pod inventory or infer Pod versions from local packages.
- *(Superseded by the licence policy above, 18 October 2026.)* **The literal MISSION Apache/MIT/BSD-only allowlist is not satisfied by the existing complete runtime.** Existing PSF/Python, LGPL, MPL, ISC, CNRI, Unlicense, Zlib/CC0 and NVIDIA SDK terms are recorded below. The already approved GPU/decoder setup does not silently relabel these licenses or establish a blanket exception. All compared model-weight cards are recorded as Apache-2.0; no model with an unverified/restricted weight license is promoted.
- All ten metadata-only distribution records now have preserved exact-version upstream component notices: catboost, openpyxl, streamlit, et_xmlfile, tqdm (both environments), flatbuffers, tokenizers, sentencepiece and dnspython. Three installed CatBoost-widget third-party notices were also recovered outside `.dist-info`. This resolves the package-notice presence gaps; it does not establish completed redistribution compliance.
- The portable notice archive is `data/processed/mission-license-notices-20261018.tar.gz` (796210 bytes; SHA-256 `eff01ecf954d56957edf5975a4e2bb55e4992d446fd5a253adb5fc7ff9662dd8`). Its **316 actual notice files** comprise 303 hash-verified installed notices, one relocated NVIDIA SDK notice, three widget notices, and nine upstream licenses. The enclosed `manifest.json` records every source path/URL/hash and `installed-distributions.json` retains the exact Pod metadata. No media, weights, keys or inference payloads are included.
- Preserve the entire installed notice directories when packaging a runtime. Main roots are `/workspace/datathon-kit/.venv/lib/python3.12/site-packages/` (service) and `/opt/vllm/lib/python3.12/site-packages/` (model); append each inventory `license_files[].path`. GPU driver/base-image notices are outside Python metadata and also require retention. The archive does not supply corresponding source/relinking material or establish that every system component and redistribution obligation has been audited.
- Existing scoped approvals are explicit in `docs/decisions.md`: the 17 October FFmpeg decision names the pinned **LGPL-v3** decoder and approved project-local installation; the Runpod decision authorizes the pinned GPU runtime/model setup, and MISSION explicitly authorizes the vLLM/8B setup. These establish the authority to install/use the recorded components. They are not a blanket reclassification of transitive licenses or an explicit waiver of the later literal Apache/MIT/BSD-only library wording. No broader approval is invented.

### Supplemental notice sources

These files were retrieved as license documents only, without installing or updating a package. Exact bytes, retrieval metadata and hashes are in the archive manifest. Version-tag URLs are recorded as source provenance; the saved content hashes preserve the texts retrieved.

| Package/version | Exact-version upstream notice |
|---|---|
| catboost 1.2.10 | [LICENSE](https://raw.githubusercontent.com/catboost/catboost/v1.2.10/LICENSE) |
| openpyxl 3.1.5 | [LICENCE.rst](https://foss.heptapod.net/openpyxl/openpyxl/-/raw/3.1.5/LICENCE.rst) |
| streamlit 1.66.0 | [LICENSE](https://raw.githubusercontent.com/streamlit/streamlit/1.66.0/LICENSE) |
| et_xmlfile 2.0.0 | [LICENCE.rst](https://foss.heptapod.net/openpyxl/et_xmlfile/-/raw/2.0.0/LICENCE.rst) |
| tqdm 4.70.1 (both environments) | [LICENCE](https://raw.githubusercontent.com/tqdm/tqdm/v4.70.1/LICENCE) |
| flatbuffers 25.12.19 | [LICENSE](https://raw.githubusercontent.com/google/flatbuffers/v25.12.19/LICENSE) |
| tokenizers 0.22.2 | [LICENSE](https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/LICENSE) |
| sentencepiece 0.2.2 | [LICENSE](https://raw.githubusercontent.com/google/sentencepiece/v0.2.2/LICENSE) |
| dnspython 2.9.0 | [LICENSE](https://raw.githubusercontent.com/rthalley/dnspython/v2.9.0/LICENSE) |

### Material notices and exceptions

| Component | Observed license / notice | Scope and remaining qualification |
|---|---|---|
| Qwen3-VL 32B AWQ, 8B BF16, tested 7B and 30B-A3B AWQ | Apache-2.0 on the pinned model cards/manifests already listed | Exact revisions and hashes are in `deploy/`; failed candidates stay failed. Parent/quantized card attribution is distinct from dependency licenses. |
| Python; typing_extensions 4.16.0; model aiohappyeyeballs 2.7.1; matplotlib 3.11.2 | PSF/Python-family declarations | Existing runtime/preparation components; retain their actual notices. Matplotlib is installed, not a demonstrated primary-inference dependency. |
| certifi 2026.7.22; tqdm 4.70.1 | MPL-2.0; MPL-2.0 AND MIT, respectively | Installed in both environments. Do not call the combined tqdm declaration MIT-only. |
| frozendict 2.4.7; soxr 1.1.0; pycountry 26.2.16 | LGPL v3; LGPL-2.1-or-later; LGPL-2.1-only | Model-environment dependencies, not additional trained models. Inclusion does not demonstrate that the audio path is used; application analysis remains visual only. |
| OpenCV headless 5.0.0.93 | OpenCV Apache-2.0 plus bundled third-party terms | Both wheels preserve `opencv_python_headless-5.0.0.93.dist-info/LICENSE-3RD-PARTY.txt`; its FFmpeg section explicitly supplies LGPL-2.1 terms. The wrapper license does not replace bundled-code notices. |
| Standalone local FFmpeg/ffprobe | LGPL-3.0-or-later, approved pinned build | `tools/local-video-decoder/manifest.json` pins build/URL/hashes; `licenses/COPYING.LGPLv3` and `licenses/COPYING.GPLv3` are retained within that tool folder. A retained GPL text does not alone prove this LGPL build was built with GPL components. |
| NVIDIA CUDA runtime/libraries | Twelve installed distributions declare NVIDIA proprietary/SDK terms | CUDA runtime, cuFile, CUPTI, cuFFT, cuSOLVER, cuSPARSE, cuSPARSELt, cuBLAS, cuDNN, cuRAND, NVRTC, nvJitLink. NVTX and NCCL instead declare Apache/BSD. Hardware/runtime terms are not an Apache model license. |
| Pillow 12.3.0 / 13.0.0 | MIT-CMU, with bundled notices | Different versions in S/M; preserve the complete wheel notice, including bundled codec/library sections. |
| NumPy 2.5.4 / 2.2.6; SciPy 1.18.1 | NumPy service expression: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; model NumPy/SciPy declare BSD with bundled notices | Preserve complete compiled-wheel notices; a short package classifier does not enumerate every bundled third-party license. |
| regex 2026.9.29; email-validator 2.3.0; ISC packages | Apache-2.0 AND CNRI-Python; Unlicense; ISC | Existing model dependencies. ISC packages include prometheus-fastapi-instrumentator, shellingham and dnspython. |
| vLLM 0.11.0; Transformers 4.57.1; torch 2.8.0; Triton 3.4.0 | Apache-2.0; Apache-2.0; BSD-3-Clause; MIT | Isolated installed versions are verified by the exact Pod export. Cloud/API client packages being installed does not establish their use; submitted inference routes to the local loopback model endpoint. |

### Full installed Python distribution table

`S` = service environment; `M` = model environment. A row shared by both environments is combined only when name, version and declaration match. Notice counts refer to the original `.dist-info` inventory and are evidence locations, not a compliance grade. Generic declarations such as “BSD License” retain the metadata wording rather than inventing a precise SPDX variant.

| Package | Version | Environment | Declared license / inspected notice | Notice-file entries |
|---|---|---|---|---|
| agent-detector | 2.0.0 | M | MIT | M:1 |
| ai-edge-litert | 2.3.0 | S | Apache Software License | S:1 |
| aiohappyeyeballs | 2.7.1 | M | PSF-2.0 | M:1 |
| aiohttp | 3.14.5 | M | Apache-2.0 AND MIT | M:2 |
| aiosignal | 1.4.0 | M | Apache 2.0 | M:1 |
| alembic | 1.20.0 | S | MIT | S:1 |
| altair | 6.3.0 | S | BSD License | S:1 |
| annotated-doc | 0.0.5 | S/M | MIT | S:1; M:1 |
| annotated-types | 0.8.0 | S/M | MIT | S:1; M:1 |
| anyio | 4.15.1 | S/M | MIT | S:1; M:1 |
| astor | 0.8.1 | M | BSD-3-Clause | M:1 |
| attrs | 26.1.0 | S/M | MIT | S:1; M:1 |
| backports.strenum | 1.3.1 | S | MIT | S:1 |
| blake3 | 1.0.11 | M | CC0-1.0 OR Apache-2.0 | M:1 |
| cachetools | 7.2.3 | M | MIT | M:1 |
| catboost | 1.2.10 | S | Apache License, Version 2.0 | S:0 |
| cbor2 | 6.1.5 | M | MIT | M:1 |
| certifi | 2026.7.22 | S/M | MPL-2.0 | S:1; M:1 |
| cffi | 2.1.1 | M | MIT-0 | M:1 |
| charset-normalizer | 3.5.2 | S/M | MIT | S:1; M:1 |
| click | 8.5.0 | S/M | BSD-3-Clause | S:1; M:1 |
| cloudpickle | 3.1.2 | S/M | BSD-3-Clause | S:1; M:1 |
| colorlog | 6.12.0 | S | MIT License | S:1 |
| compressed-tensors | 0.11.0 | M | Apache 2.0 | M:1 |
| contourpy | 1.4.0 | S | BSD-3-Clause | S:1 |
| cuda-pathfinder | 1.8.3 | M | Apache-2.0 | M:1 |
| cupy-cuda12x | 14.2.0 | M | MIT | M:2 |
| cycler | 0.12.1 | S | BSD License | S:1 |
| depyf | 0.19.0 | M | MIT | M:1 |
| detect-installer | 0.2.1 | M | 0BSD | M:1 |
| dill | 0.4.1 | M | BSD-3-Clause | M:1 |
| diskcache | 5.6.3 | M | Apache 2.0 | M:1 |
| dnspython | 2.9.0 | M | ISC | M:0 |
| einops | 0.8.2 | M | MIT | M:1 |
| email-validator | 2.3.0 | M | Unlicense | M:1 |
| et_xmlfile | 2.0.0 | S | MIT | S:0 |
| fastapi | 0.143.3 | S/M | MIT | S:1; M:1 |
| fastapi-cli | 0.0.32 | M | MIT | M:1 |
| fastapi-cloud-cli | 0.26.0 | M | MIT | M:1 |
| fastar | 0.12.0 | M | MIT | M:1 |
| filelock | 4.2.0 | M | MIT | M:1 |
| flatbuffers | 25.12.19 | S | Apache 2.0 | S:0 |
| fonttools | 4.67.0 | S | MIT | S:2 |
| formulaic | 1.2.2 | S | MIT | S:1 |
| frozendict | 2.4.7 | M | LGPL v3 | M:1 |
| frozenlist | 1.8.0 | M | Apache-2.0 | M:1 |
| fsspec | 2026.9.0 | M | BSD-3-Clause | M:1 |
| gguf | 0.19.0 | M | MIT License | M:1 |
| googleapis-common-protos | 1.75.5 | M | Apache-2.0 | M:1 |
| graphviz | 0.21 | S | MIT | S:1 |
| h11 | 0.16.0 | S/M | MIT | S:1; M:1 |
| hf-xet | 1.7.0 | M | Apache-2.0 | M:1 |
| httpcore | 1.0.9 | M | BSD-3-Clause | M:1 |
| httpcore2 | 2.13.1 | M | BSD-3-Clause | M:1 |
| httptools | 0.9.0 | M | MIT | M:3 |
| httpx | 0.28.1 | M | BSD-3-Clause | M:1 |
| httpx2 | 2.13.1 | M | BSD-3-Clause | M:1 |
| huggingface_hub | 0.36.2 | M | Apache | M:1 |
| idna | 3.20 | S/M | BSD-3-Clause | S:1; M:1 |
| interegular | 0.3.3 | M | MIT | M:1 |
| interface_meta | 2.0.1 | S | MIT | S:1 |
| itsdangerous | 2.2.0 | S | BSD License | S:1 |
| Jinja2 | 3.1.6 | S/M | BSD License | S:1; M:1 |
| jiter | 0.17.0 | M | MIT | M:1 |
| joblib | 1.6.0 | S | BSD-3-Clause | S:1 |
| jsonschema | 4.26.0 | S/M | MIT | S:1; M:1 |
| jsonschema-specifications | 2025.9.1 | S/M | MIT | S:1; M:1 |
| kiwisolver | 1.5.1 | S | BSD License | S:1 |
| lark | 1.2.2 | M | MIT | M:1 |
| lightgbm | 4.7.0 | S | MIT | S:1 |
| llguidance | 0.7.30 | M | MIT | M:1 |
| llvmlite | 0.44.0 | M | BSD | M:2 |
| lm-format-enforcer | 0.11.3 | M | MIT | M:1 |
| lxml | 6.1.3 | S | BSD-3-Clause | S:2 |
| Mako | 1.4.3 | S | MIT | S:1 |
| markdown-it-py | 4.2.0 | M | MIT License | M:2 |
| MarkupSafe | 3.0.4 | S/M | BSD-3-Clause | S:1; M:1 |
| matplotlib | 3.11.2 | S | Python Software Foundation License | S:1 |
| mdurl | 0.1.2 | M | MIT License | M:1 |
| mistral_common | 1.12.0 | M | Apache-2.0 | M:1 |
| ml_dtypes | 0.6.0 | S | Apache-2.0 | S:2 |
| mpmath | 1.3.0 | M | BSD | M:1 |
| msgpack | 1.2.3 | M | Apache-2.0 | M:1 |
| msgspec | 0.22.0 | M | BSD-3-Clause | M:1 |
| multidict | 7.1.0 | M | Apache-2.0 | M:1 |
| narwhals | 2.27.1 | S | MIT | S:1 |
| networkx | 3.7 | M | BSD-3-Clause | M:1 |
| ninja | 1.13.2 | M | Apache-2.0 | M:1 |
| numba | 0.61.2 | M | BSD | M:2 |
| numpy | 2.2.6 | M | BSD License | M:1 |
| numpy | 2.5.4 | S | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | S:17 |
| nvidia-cublas-cu12 | 12.8.4.1 | M | NVIDIA Proprietary Software | M:1 |
| nvidia-cuda-cupti-cu12 | 12.8.90 | M | NVIDIA Proprietary Software | M:1 |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 | M | NVIDIA Proprietary Software | M:1 |
| nvidia-cuda-runtime-cu12 | 12.8.90 | M | NVIDIA Proprietary Software | M:1 |
| nvidia-cudnn-cu12 | 9.10.2.21 | M | LicenseRef-NVIDIA-Proprietary | M:1 |
| nvidia-cufft-cu12 | 11.3.3.83 | M | NVIDIA Proprietary Software | M:1 |
| nvidia-cufile-cu12 | 1.13.1.3 | M | NVIDIA Proprietary Software | M:1 |
| nvidia-curand-cu12 | 10.3.9.90 | M | NVIDIA Proprietary Software | M:1 |
| nvidia-cusolver-cu12 | 11.7.3.90 | M | NVIDIA Proprietary Software | M:1 |
| nvidia-cusparse-cu12 | 12.5.8.93 | M | NVIDIA Proprietary Software | M:1 |
| nvidia-cusparselt-cu12 | 0.7.1 | M | NVIDIA Proprietary Software | M:0 |
| nvidia-nccl-cu12 | 2.27.3 | M | BSD-3-Clause | M:1 |
| nvidia-nvjitlink-cu12 | 12.8.93 | M | NVIDIA Proprietary Software | M:1 |
| nvidia-nvtx-cu12 | 12.8.90 | M | Apache 2.0 | M:1 |
| openai | 3.33.0 | M | Apache-2.0 | M:1 |
| openai-harmony | 0.0.8 | M | Apache-2.0 (actual notice read) | M:1 |
| opencv-python-headless | 5.0.0.93 | S/M | Apache 2.0 | S:2; M:2 |
| openpyxl | 3.1.5 | S | MIT | S:0 |
| opentelemetry-api | 1.45.1 | S/M | Apache-2.0 | S:1; M:1 |
| opentelemetry-exporter-http-transport | 0.66b1 | M | Apache-2.0 | M:1 |
| opentelemetry-exporter-otlp-common | 0.66b1 | M | Apache-2.0 | M:1 |
| opentelemetry-exporter-otlp-proto-common | 1.45.1 | M | Apache-2.0 | M:1 |
| opentelemetry-exporter-otlp-proto-http | 1.45.1 | M | Apache-2.0 | M:1 |
| opentelemetry-proto | 1.45.1 | M | Apache-2.0 | M:1 |
| opentelemetry-sdk | 1.45.1 | M | Apache-2.0 | M:1 |
| opentelemetry-semantic-conventions | 0.66b1 | M | Apache-2.0 | M:1 |
| optuna | 5.0.0 | S | MIT License | S:2 |
| outlines_core | 0.2.11 | M | Apache-2.0 | M:1 |
| packaging | 26.3 | S/M | Apache-2.0 OR BSD-2-Clause | S:3; M:3 |
| pandas | 3.0.6 | S | BSD License | S:1 |
| partial-json-parser | 0.2.1.1.post7 | M | MIT | M:1 |
| patsy | 1.0.3 | S | 2-clause BSD | S:1 |
| pillow | 12.3.0 | S | MIT-CMU | S:1 |
| pillow | 13.0.0 | M | MIT-CMU | M:1 |
| plotly | 7.1.0 | S | MIT | S:1 |
| prometheus-fastapi-instrumentator | 8.1.0 | M | ISC | M:1 |
| prometheus_client | 0.26.0 | M | Apache-2.0 AND BSD-2-Clause | M:1 |
| propcache | 0.5.4 | M | Apache-2.0 | M:1 |
| protobuf | 7.36.2 | S/M | 3-Clause BSD License | S:1; M:1 |
| psutil | 7.2.2 | M | BSD-3-Clause | M:1 |
| py-cpuinfo | 9.0.0 | M | MIT | M:1 |
| pyarrow | 26.0.0 | S | Apache-2.0 | S:1 |
| pybase64 | 1.5.1 | M | BSD-2-Clause | M:2 |
| pycountry | 26.2.16 | M | LGPL-2.1-only | M:1 |
| pycparser | 3.11 | M | BSD-3-Clause | M:1 |
| pydantic | 2.14.1 | S/M | MIT | S:1; M:1 |
| pydantic-extra-types | 2.11.1 | M | MIT | M:1 |
| pydantic-settings | 2.16.0 | M | MIT | M:1 |
| pydantic_core | 2.50.1 | S/M | MIT | S:1; M:1 |
| pydeck | 0.9.3 | S | Apache License 2.0 | S:1 |
| Pygments | 2.21.0 | M | BSD-2-Clause | M:1 |
| pyparsing | 3.3.3 | S | MIT | S:1 |
| python-dateutil | 2.9.0.post0 | S | Dual license; BSD / Apache classifiers | S:1 |
| python-dotenv | 1.2.4 | M | BSD-3-Clause | M:1 |
| python-json-logger | 4.2.0 | M | BSD-2-Clause | M:1 |
| python-multipart | 0.0.32 | S/M | Apache-2.0 | S:1; M:1 |
| python-pptx | 1.0.2 | S | MIT | S:1 |
| PyYAML | 6.0.3 | S/M | MIT | S:1; M:1 |
| pyzmq | 27.2.0 | M | BSD-3-Clause | M:4 |
| ray | 2.59.0 | M | Apache 2.0 | M:14 |
| referencing | 0.37.0 | S/M | MIT | S:1; M:1 |
| regex | 2026.9.29 | M | Apache-2.0 AND CNRI-Python | M:1 |
| requests | 2.34.2 | S/M | Apache-2.0 | S:1; M:1 |
| rich | 15.0.0 | M | MIT | M:1 |
| rich-toolkit | 0.20.6 | M | MIT | M:2 |
| rignore | 0.8.1 | M | MIT | M:1 |
| rpds-py | 2026.9.1 | S/M | MIT | S:1; M:1 |
| safetensors | 0.8.0 | M | Apache Software License | M:1 |
| scikit-learn | 1.9.1 | S | BSD-3-Clause | S:1 |
| scipy | 1.18.1 | S/M | BSD License | S:1; M:1 |
| sentencepiece | 0.2.2 | M | Apache-2.0 | M:0 |
| sentry-sdk | 2.72.0 | M | MIT | M:1 |
| setproctitle | 1.3.8 | M | BSD-3-Clause | M:1 |
| setuptools | 79.0.1 | M | MIT (actual notice read) | M:19 |
| shap | 0.54.1 | S | MIT License | S:1 |
| shellingham | 1.5.4 | M | ISC License | M:1 |
| six | 1.17.0 | S/M | MIT | S:1; M:1 |
| slicer | 0.0.8 | S | MIT License | S:1 |
| sniffio | 1.3.1 | M | MIT OR Apache-2.0 | M:3 |
| soundfile | 0.14.0 | M | BSD 3-Clause License | M:1 |
| soxr | 1.1.0 | M | LGPL-2.1-or-later | M:4 |
| SQLAlchemy | 2.1.4 | S | MIT | S:1 |
| starlette | 1.8.0 | S/M | BSD-3-Clause | S:1; M:1 |
| statsmodels | 0.15.0 | S | BSD-3-Clause | S:1 |
| streamlit | 1.66.0 | S | Apache-2.0 | S:0 |
| sympy | 1.14.0 | M | BSD | M:1 |
| tabulate | 0.10.0 | S | MIT | S:1 |
| threadpoolctl | 3.7.0 | S | BSD-3-Clause | S:1 |
| tiktoken | 0.14.0 | M | MIT (embedded license text) | M:1 |
| tokenizers | 0.22.2 | M | Apache Software License | M:0 |
| torch | 2.8.0 | M | BSD-3-Clause | M:1 |
| torchaudio | 2.8.0 | M | BSD License | M:1 |
| torchvision | 0.23.0 | M | BSD | M:1 |
| tqdm | 4.70.1 | S/M | MPL-2.0 AND MIT | S:0; M:0 |
| transformers | 4.57.1 | M | Apache 2.0 License | M:1 |
| triton | 3.4.0 | M | MIT License | M:1 |
| truststore | 0.10.5 | M | MIT | M:1 |
| typer | 0.27.3 | M | MIT | M:1 |
| typing-inspection | 0.4.4 | S/M | MIT | S:1; M:1 |
| typing_extensions | 4.16.0 | S/M | PSF-2.0 | S:1; M:1 |
| urllib3 | 2.8.0 | S/M | MIT | S:1; M:1 |
| uvicorn | 0.54.0 | S/M | BSD-3-Clause | S:1; M:1 |
| uvloop | 0.23.0 | M | MIT License | M:2 |
| vllm | 0.11.0 | M | Apache-2.0 | M:1 |
| watchdog | 6.0.0 | S | Apache-2.0 | S:2 |
| watchfiles | 1.3.0 | M | MIT | M:1 |
| websockets | 17.2 | S/M | BSD-3-Clause | S:1; M:1 |
| wrapt | 2.5.1 | S | BSD-2-Clause | S:1 |
| xformers | 0.0.32.post1 | M | BSD classifier; License metadata UNKNOWN | M:1 |
| xgrammar | 0.1.25 | M | Apache 2.0 | M:1 |
| xlsxwriter | 3.2.9 | S | BSD-2-Clause | S:1 |
| yarl | 1.25.1 | M | Apache-2.0 | M:1 |

### Legal and technical scoring evidence still outstanding

- `service/rules.json` has 14 types, all deterministic `enabled` flags false and all `rules` lists empty. Eleven types carry a qualified `context_source`; keys-left-in-safe, workstation-unlocked and personal-activity carry none. Public-source context is not an activated, verified automatic legal finding. Under the requested proxy, empty `rules: []` earns only the explicit half-share; no 15/15 legal claim is supported.
- `docs/action_rule_candidates.md` and `docs/case/legal.md` remain the source authority. Cash storage under NBK 231 §20 includes designated lockable storage, not only safes. Identity checks permit relevant electronic documents; intact packaged withdrawals have exceptions. Premises/device requirements in NBK 14 §16 do not establish every proposed behavior ban. The quoted temporary §16 edition must be named. No verified categorical key-removal rule, standalone food ban, private-reading ban or immediate short-absence workstation-lock rule is invented.
- Every event needs a literal observable explanation, correct source/time JPEGs, act/section where verified, and material source/applicability limitations. Evidence does not prove an item is cash, a person unauthorized, a session unlocked, or a missing step absent while offscreen. These remain recognition/applicability risks, not wording problems alone.
- **Historical 04:38 audit status:** the successful selected-GPU proof had not yet been supplied. It is now preserved and passed as recorded above. **The inventory/license table and package-notice presence coverage are complete; strict Apache/MIT/BSD-only compatibility remains false.** Table completion does not resolve that policy conflict or alone qualify the Tech gate. Full redistribution/source obligations and system notices remain a separate incomplete audit.
- **Historical checkpoint1:** 44 public jobs, zero failed jobs; 41/42 traced jobs primary, one transformed fallback. Worst observed 76.51247401 seconds; longest repeats 47.210525177 / 44.749218023 seconds lacked primary metadata. Only 28 transformed DEV examples were checked. Checkpoint3 above now qualifies all 40 clean uploads and two primary repeats; full 280-transform robustness remains outstanding.
- Final handoff still needs the final measured report/per-type limitations, complete UI and keyed API instructions/example, private jury-key handoff, full robustness evidence, and the funded judging uptime window. Selected recipe/tag, recorded code publication, full clean API qualification and the tested GPU-offline path are evidenced above. Full source/license redistribution verification and the literal licence-policy conflict remain disclosed.

<!-- END CURRENT LICENSING AUDIT -->


### Historical operational snapshot — superseded by checkpoint3 above

Earlier development configuration (18 October local): pinned Qwen3-VL-32B-Instruct-AWQ on the existing A40, vLLM0.11.0/Transformers4.57.1, loopback8002, standard16-image sweep, frame-only preprocessing, proposal policy. At this snapshot, the local64-image/32k-context candidate had passed software/coverage checks but was **not yet deployed or quality-qualified**. Default code remained16images. The measurements and pending statements in this historical block describe that earlier state.

Measured results, not a jury forecast:

| Scope | Video accuracy | Positive videos found | False alarms | Events within2s | Correct types among localized events |
|---|---:|---:|---:|---:|---:|
|30previously-seenDEV, fixed32Bproposal recipe |20/30|12/16|6|5/16|5/5|
|10previously-seen regression-reserve, public API checkpoint1 |4/10|1/5|2|1/6|1/1|

The type fractions are conditional on localization: correct typed event coverage is5/16onDEV, not100% recognition. DEV produced19events,14unmatched to the supplied anchors. Five false-alarm events were safe-open predictions and one was door-open. The engineering proxy is43.3854with missing gates at zero; it is not an official score. This candidate fails the release targets and has not earned a quality-promotion tag.

All14clean public checkpoint requests (10reserve+4DEV) completed through the primary model without failed jobs; worst51.1425seconds including upload/polling. This is a partial API sample; the two longest-video repeats and full40-video API qualification are still outstanding. FourDEV API calls reproduce the exact benchmark request hashes and answers. Augmentation checks are in progress. The reserve is not independent and no per-video reserve errors were inspected for tuning.

The historical GPU network-denied attempt was incomplete: two model processes passed the kernel guard, while the EngineCore audit rejected its same-machine TCPStore connection. A tested correction recognized only exact namespace-owned interface addresses, not arbitrary private networks; it then required the next controlled model restart. No offline/Tech credit was claimed from that failed attempt. The later successful proof is recorded above.

### Service endpoint and architecture

Public development URL: https://p7ayn21cpy6voe-8000.proxy.runpod.net . API uses X-API-Key, delivered separately; never committed here. Upload MOV/MP4, poll the returned job. Prototype availability does not establish accuracy. The existing A40 Pod is the only approved host; $0.59/hour, $15 total including storage, auto top-up off. Preserve serving time for jury testing.

The service separates durable upload/jobs, source-time video decoding, pretrained observations, explicit physical predicates/rules, a self-hosted vision auditor and source-aligned JPEG evidence. API fields remain the supplied contract. CPU fallback works locally but its numeric anomaly model is weak and lacks action-specific grounding. No face identification/audio or hosted AI inference is used by the service.

## Historical component summary

This compact summary predates the complete installed inventory and successful GPU proof above; its pending-status wording is retained as historical provenance. Use the current audit and checkpoint3 evidence for qualification status.

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
