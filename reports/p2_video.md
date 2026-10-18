# P2 — first working local video flow

Date: 2026-10-17. P2 and `opencv-python-headless`/`fastapi` were explicitly approved by the human. The local upload → analysis → result flow works. **The detection baseline has not demonstrated useful accuracy and is not jury-ready.** No precise event taxonomy, observed-action explanation or verified rule-to-event mapping is claimed.

## Implemented behavior

- Single local UI, `.mov` upload, processing status, binary verdict, one candidate event/evidence frame for positive outputs, explicit baseline limitations.
- Required keyed API: `POST /api/v1/analyze`, multipart `file`; `GET /api/v1/jobs/{job_id}`. Missing/incorrect `X-API-Key` returns 401. Core result schema matches the brief.
- Local OpenCV decoding, approximately 2 FPS sampling, 2-second windows, 74 numeric features/window and 222 pooled clip features. No IDs, filenames, duration, timestamps or labels enter feature vectors. Temporal positions are used only for localization.
- Fold-fitted imputation, scaling and fixed logistic model. A separate weakly supervised logistic window model uses nearest annotated positive windows and windows from negative clips; it does not label other positive-clip windows as negative. Disputed timestamps are omitted from window supervision.
- One killable analysis subprocess per job, bounded upload/queue/result size, deadline including upload/queue time, safe random job paths, upload/temp-result cleanup, no cloud model calls. Errors return `failed`, never a negative verdict.
- The local launcher creates a private key file without printing the key. No secret was committed or transmitted. The API is loopback-only; no public deployment or competition submission has been performed.

## Frozen validation

Protocol: `data/processed/fortebank-20261017/video_protocol.json`.

Fingerprint: `4dc3881396670f9496e6958178589cbe08325a6b1c756d9ea2257fff803fae3d`.

Forty content groups, 21 positive and 19 negative. Eight unique videos (four per class) are reserved. Development uses 32 clips (17 positive, 15 negative) and three folds: 21/11, 21/11, 22/10 train/validation groups. No group overlaps a fold boundary; all development groups receive one out-of-fold prediction. Exact duplicate aliases remain grouped and count once. Recording/session similarity beyond exact byte duplicates remains unverified. No capture dates exist, so chronological splitting and credit-label gaps are inapplicable.

The holdout was frozen before frame decoding and remains undecoded and unevaluated by P2. Generic `src/eval.py` is unchanged; video-specific splitting and metrics live in `src/video_eval.py`.

## Real development results

Canonical delivered development run: `data/processed/video-baseline-comparison/`. Initial linear run: `data/processed/video-baseline/`. Both use the same frozen protocol, feature source and fixed 0.5 threshold. The comparison reused byte/source-checked numeric caches and copied them into the new self-contained run.

| Model | ROC-AUC | Balanced accuracy | Precision | Recall | F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---|
| Constant 0.5 dummy (all flagged at threshold) | 0.500000 | 0.500000 | 0.531250 | 1.000000 | 0.693878 | 0 / 15 / 0 / 17 |
| Linear | 0.509804 | 0.460784 | 0.500000 | 0.588235 | 0.540541 | 5 / 10 / 7 / 10 |
| Fixed shallow LightGBM | 0.409804 | 0.405882 | 0.437500 | 0.411765 | 0.424242 | 6 / 9 / 10 / 7 |

Before → after comparison: **selected development ROC-AUC 0.5098039215686274 → 0.5098039215686274**. LightGBM is dropped. Linear is retained solely to exercise the product flow; it does not beat the dummy convincingly, and its thresholded metrics are worse. Selection uses the same development folds and is not an independent generalization estimate. No official aggregate jury score is inferred from these diagnostics.

Candidate localization on 16 eligible positive development clips, independently of whether their verdict was positive: mean absolute error 6.1895833333 seconds; median 5.0 seconds; 25% within one second and 31.25% within three seconds. These are diagnostic tolerances, not an official scoring rule. Sparse annotations may omit other valid events; the disputed duplicate group is unscored.

Canonical JSON: `<run>/development.json`; experiments appended to root `experiments.md` with keep/drop decisions. Saved model: `<run>/models/baseline.joblib`, feature/schema/source checks and `<run>/model_manifest.json`.

## Sanity

`uv run python -m src.sanity --task video --run-dir data/processed/video-baseline-comparison` succeeded:

- 32 development groups; zero shared groups; zero decoded holdout groups.
- No numeric feature had oriented single-feature AUC above 0.90.
- Shuffled-target ROC-AUC over 20 seeded permutations: mean **0.4718627450980392**, standard deviation **0.10829899198397151**, versus dummy 0.5.
- No problems detected by these checks. They are a limited diagnostic on a small sample, not proof that every leak or session confound is absent.

## Actual local HTTP checks

The canonical service was started on `127.0.0.1:8000`. `<run>/api_check.json` records:

| Check | Observed result |
|---|---|
| UI | HTTP 200; no API key embedded |
| Missing/wrong key | HTTP 401 |
| Low training prediction upload (11,603,296 bytes) | `done`, `no_violation`, zero evidence frames; 2.9826694280 seconds |
| High training prediction upload (4,236,340 bytes) | `done`, `violation`, one valid decoded JPEG; 2.2708361640 seconds |
| Same positive bytes under a negative-looking filename | Identical verdict/time/frame/description payload; 2.2684012640 seconds |
| Invalid `.mov` bytes | `failed`, no fabricated negative; 1.9408700860 seconds |

Evidence timestamps were verified against source duration and JPEG bytes decoded locally. No pixels, base64 payloads or clip-specific descriptions were printed to tools. These engineering checks use development examples selected by model outputs; they are not an accuracy test or cold-start/browser-render benchmark. Hidden/holdout videos used: zero. Not every duration/concurrency/camera has been benchmarked; visible-browser clickthrough remains a manual check.

## Commands

Full suite after fixing multiprocessing test isolation from the earlier Streamlit AppTest: **109 passed, 2 skipped, 12 warnings in 33.66s**. The warnings are LightGBM's existing `eval_set` deprecation. The two skips require optional ignored mock CSVs. Tests cover numeric/schema repeatability, duplicate/holdout isolation, preprocessing/weak-window supervision, authorization/upload/schema handling, real generated-video processing in a spawned worker, corrupt-video failure and a hard-killed stalled worker. No actual supplied footage was viewed by the coding model. Initial full-suite failures exposed Streamlit's changed `__main__` context in later multiprocessing tests; the test fixture now restores the pytest main context for those subprocess cases. Production code was unchanged by that isolation fix.

Network-denied direct inference on the longest development clip (78.734 seconds) completed in **1.6453856930 seconds**, with zero attempted network connections and no holdout use; `<run>/offline_check.json` records this. This is one measured CPU example, not a universal latency claim. Raw source/preserved delivery hashes still match; job temp storage was empty after checks. Frozen `src/eval.py` SHA-256 remains `0ef1963df44a04495ecf95d8c40c82c606ac7987ac410202094c5e671483a762`.

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache uv add opencv-python-headless fastapi
UV_CACHE_DIR=/tmp/datathon-uv-cache DATATHON_THREADS=2 uv run --offline python -m src.train --task video --note "P2 first local video baseline"
UV_CACHE_DIR=/tmp/datathon-uv-cache DATATHON_THREADS=2 uv run --offline python -m src.train --task video --run-dir data/processed/video-baseline-comparison --cache-dir data/processed/video-baseline/cache --models linear,lgbm --note "P2 fixed shallow comparison"
UV_CACHE_DIR=/tmp/datathon-uv-cache DATATHON_THREADS=2 uv run --offline python -m src.sanity --task video --run-dir data/processed/video-baseline-comparison
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m scripts.serve_video
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m scripts.check_video_flow
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --offline pytest -q -o faulthandler_timeout=40
UV_CACHE_DIR=/tmp/datathon-uv-cache uv sync --locked --offline
git diff --check
```

The initial sandboxed dependency attempt failed DNS; retry with network escalation installed the approved packages and their required transitive dependencies. Loopback HTTP checks required sandbox escalation. No new package beyond the two approved top-level additions was requested. `pyproject.toml` and `uv.lock` are updated together. Installed direct package metadata reports FastAPI 0.143.3 as MIT and OpenCV headless 5.0.0.93 as Apache 2.0; distribution/transitive notices still need preserving for delivery.

## Remaining work

The most valuable next model experiment is a better local visual representation. OpenCV DNN can run a commercially permitted pretrained image encoder without another inference package, subject to actual model/license checks and measured CPU speed. Generic image features still do not establish action meaning or legal grounding. Event-type definitions, verified source mappings, local semantic spot-checks and camera-angle tests remain essential.

Public UI/API hosting, fresh-machine reproduction, normative event coverage, multi-event detection (the current analyzer returns at most one candidate), hidden-holdout evaluation, final pitch and backup recording remain open. Cached `uv sync --locked --offline` resolved 120 packages and checked 105 installed packages; this is not a fresh download on another machine. The server's negative response is limited to this weak baseline and must not be interpreted as assurance of compliance.

## Windows browser connection repair

The initial P2 HTTP check verified Linux loopback only. The user then reported that the Windows browser could not connect. Current checks confirmed Linux `127.0.0.1:8000` returned HTTP 200, but Windows IPv4 localhost failed and localhost/IPv6 checks also failed. This was a host-to-WSL connection issue, not an absent model or failed backend.

A proposed WSL private-interface bind was rejected by automatic approval review because it could broaden exposure beyond localhost. That bind was not applied. The service was restored on Linux loopback, and a safer approved localhost bridge was implemented using `scripts/windows_loopback_proxy.cjs` plus `scripts/loopback_request.py`. The Windows listener is verified as **127.0.0.1:8001**; requests travel by local WSL process pipes to the fixed Linux loopback service. No firewall/network security settings, public addresses or dependencies were changed. Keys and footage are not placed in command arguments, logs or persistent relay storage.

Windows-side verification: UI HTTP 200 (7,216 bytes); missing API key HTTP 401. One 11,603,296-byte development clip uploaded through the same Windows URL returned HTTP 202 and job status `done`/`no_violation` after **5.63 seconds**. No holdout used; no footage, frames or descriptions emitted. The direct relay worker also rejects unsupported paths. Windows Node syntax check, Python compilation and `git diff --check` passed. No model/features changed or training was rerun for this connectivity repair.

Use `http://127.0.0.1:8001/` in the Windows/in-app browser. The private API key file is unchanged. Startup instructions for both processes are in README.md. This repair establishes Windows-side HTTP/upload access; it is not a visible-browser layout inspection or public deployment.
