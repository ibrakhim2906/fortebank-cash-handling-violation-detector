# Local pretrained video experiment

The user requested a local pretrained model to improve detection. The official OpenCV Zoo MobileNetV2 encoder now runs through the installed OpenCV DNN runtime. No new package was added. The first measured candidate has higher development AUC but misses more positives; a bounded temporal follow-up performs worse. Neither is promoted to the live UI. A reliable detector upgrade has not been demonstrated.

## Encoder and reproducibility

- Official [OpenCV Zoo MobileNetV2](https://github.com/opencv/opencv_zoo/tree/main/models/image_classification_mobilenet), Apache 2.0. Preserved license: `licenses/opencv-mobilenet/LICENSE`. This component allows commercial use; this is not an audit of every dependency in the project.
- Weight source: `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx`.
- ONNX: 13,964,571 bytes; SHA-256 `c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5`. Local path: `models/encoders/mobilenetv2.onnx`. Provenance: `models/encoders/mobilenetv2.manifest.json`. No download occurs during inference.
- Official RGB/256 resize/224 center crop/ImageNet normalization; pooled 1,280-dimensional output tensor `464`, corresponding to `GlobalAveragePool_97`. OpenCV 5 uses the tensor name when forwarding intermediate outputs; the operator name alone failed and was corrected before extraction/training.
- Fixed label-independent seed-42 Gaussian projection to 128 dimensions; 2 FPS; two-second windows; mean, standard deviation and absolute consecutive change. This produces 384 window and 1,152 pooled clip features. No filenames, timestamps, duration, target, audio or identity features enter the classifiers.
- All learned imputation/scaling/classification is fitted within frozen outer training folds. Sparse positive timestamps supply only eligible training windows; other positive windows are not assumed negative. The conflicting duplicate annotation remains excluded from localization supervision/scoring.

The eight holdout clips remain undecoded and unevaluated. The same 32 development content groups, three folds, seed and fixed 0.5 threshold are used. Protocol fingerprint: `4dc3881396670f9496e6958178589cbe08325a6b1c756d9ea2257fff803fae3d`. No raw file or frozen `src/eval.py` was changed.

## Actual clip-pooling results

Run: `data/processed/video-mobilenet/`. Complete output and provenance: `development.json`, `sanity.json`, `model_manifest.json` and `models/baseline.joblib` inside that run.

| Model | ROC-AUC | Balanced accuracy | Recall | Precision | F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---|
| Numeric linear, original live baseline | 0.509804 | 0.460784 | 0.588235 | 0.500000 | 0.540541 | 5 / 10 / 7 / 10 |
| Numeric shallow LightGBM | 0.409804 | 0.405882 | 0.411765 | 0.437500 | 0.424242 | 6 / 9 / 10 / 7 |
| MobileNet + same linear recipe | 0.439216 | 0.468627 | 0.470588 | 0.500000 | 0.484848 | 7 / 8 / 9 / 8 |
| MobileNet + same shallow LightGBM | 0.588235 | 0.535294 | 0.470588 | 0.571429 | 0.516129 | 9 / 6 / 9 / 8 |

Selected development AUC **0.509804 → 0.588235**. Matched classifier comparisons: linear **0.509804 → 0.439216**; shallow LightGBM **0.409804 → 0.588235**. At the unchanged threshold, the selected pretrained model reduces false alarms from 10 to 6 but catches 8/17 positives rather than 10/17. It fails the promotion gate declared before observing results: AUC at least 0.60, balanced accuracy above 0.50, no recall regression and passing sanity. No improvement in real deployment accuracy has been established by these development comparisons.

Candidate timestamp error over 16 eligible positive clips, independent of verdict: mean **6.189583 → 7.908333 seconds**, median **5.00 → 5.25 seconds**; within three seconds **31.25% → 25%**. Neither this representation nor these sparse labels establish precise localization.

Sanity passed without changing thresholds: 20 shuffled-target trials mean AUC **0.4645098039**, standard deviation **0.0825601967**, versus chance 0.5; no oriented single-feature AUC above 0.90. The 1,152 correlated features and only 32 clips make these diagnostics uncertain. Camera/session independence beyond exact-byte duplicates remains unverified.

## Actual runtime and flow checks

Temporary loopback service on port 8002 passed the actual upload/authentication/schema/JPEG-evidence checks. The regular user UI remained on Windows loopback port 8001, forwarding to the original Linux service on port 8000.

| Check | Observed result |
|---|---|
| UI and authentication | HTTP 200; missing/wrong key HTTP 401; key absent from UI |
| Lower training-score upload | `done`, `no_violation`; 2.958588 seconds |
| Higher training-score upload | `done`, `violation`, one actual decodable JPEG; 2.565879 seconds |
| Identical positive bytes renamed | Identical result; 2.467316 seconds |
| Corrupt `.mov` | `failed`; 1.733955 seconds |
| Longest development clip, direct inference | 78.734-second video processed in 3.142517 seconds, Python socket connections denied, zero attempts |

These are engineering checks on selected development examples, not independent accuracy, full-duration, concurrency or worst-case latency benchmarks. No frames, base64 or clip-specific descriptions were emitted to tools. No video was uploaded externally.

## Fixed temporal follow-up — rejected

Run: `data/processed/video-mobilenet-temporal/`. This comparison changes the unit of supervised classification and aggregation, rather than comparing the same clip-classifier recipes. Exactly one policy was declared after the initial result: reuse identical cached MobileNet descriptors; fit the existing linear window head within each frozen training fold; video score is the maximum window score, timestamp its argmax, threshold 0.5. No re-decoding or new representation was needed. Cached extraction timing messages refer to the original extraction, not elapsed decoding in this follow-up.

| Metric | Numeric baseline → max-window MobileNet |
|---|---|
| ROC-AUC | 0.509804 → 0.329412 |
| Balanced accuracy | 0.460784 → 0.419608 |
| Recall | 10/17 (0.588235) → 12/17 (0.705882) |
| Precision | 0.500000 → 0.480000 |
| F1 | 0.540541 → 0.571429 |
| TN / FP / FN / TP | 5 / 10 / 7 / 10 → 2 / 13 / 5 / 12 |
| Mean candidate timestamp error | 6.189583 → 7.908333 seconds |

It fails the original AUC/balanced-accuracy gate, falsely flags 13 of 15 negative clips and is rejected. We do not invert scores, change the threshold or choose another pooling rule from these validation outcomes. Higher recall/F1 here does not establish useful detection: the constant dummy flags every clip and has F1 0.693878.

Matched temporal sanity permutes whole supervision tuples (target, eligibility, annotated fraction of duration), maps the donor fraction to the recipient duration and refits the window head inside each outer training fold. It never selects windows from original labels during a shuffled trial. Twenty seeded permutations give mean AUC **0.4956862745**, standard deviation **0.1200627799**; no feature above oriented AUC 0.90. All original gates pass. Direct offline inference on the same 78.734-second development clip takes **2.801761 seconds**, with zero Python network-connection attempts and an actual local JPEG evidence result. No holdout use.

The clip-pooling candidate remains at AUC 0.588235, but still fails promotion because of recall regression and the predeclared AUC gate. The existing numeric service is retained for flow testing. Both pretrained candidates remain operator-selectable through `scripts.serve_video --run-dir ...`; neither is represented as an accuracy improvement in the live page.

## Commands

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.fetch_video_encoder
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m src.train --task video --encoder mobilenet --run-dir data/processed/video-mobilenet --baseline-run-dir data/processed/video-baseline-comparison --models linear,lgbm --note 'local pretrained MobileNetV2 representation'
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m src.sanity --task video --run-dir data/processed/video-mobilenet
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.serve_video --run-dir data/processed/video-mobilenet --port 8002
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.check_video_flow --url http://127.0.0.1:8002 --run-dir data/processed/video-mobilenet
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m scripts.check_video_offline --run-dir data/processed/video-mobilenet
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.prepare_temporal_cache
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m src.train --task video --encoder mobilenet-temporal --run-dir data/processed/video-mobilenet-temporal --models window-linear --note 'fixed max-window pretrained follow-up'
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m src.sanity --task video --run-dir data/processed/video-mobilenet-temporal
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m scripts.check_video_offline --run-dir data/processed/video-mobilenet-temporal
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline pytest -q
```

Completed run directories are immutable; reproduction needs a fresh `--run-dir`. The fetch script explicitly provisions verified weights; its first download needs internet even when uv itself is offline. Subsequent inference needs only the local weight and supervised artifact.

## Remaining quality limits

ImageNet appearance features are not a bank-action recognition model. Event taxonomy, source-grounded action/rule explanations, multi-event coverage and generalization to other sessions remain unestablished. The UI retains the generic review-candidate description and limitations. Both planned experiments are complete; no additional search was made against the same development clips. Action-specific labels and a representation suited to those observable actions are the next quality priorities.

## Files and final verification

- New inference: `src/video_pretrained.py`, `src/video_temporal.py`, `src/video_inference.py`. The API worker now imports the dispatcher in `src/video_api.py`; legacy numeric artifacts remain supported.
- Extended training/sanity: `src/video_train.py`, `src/video_sanity.py`.
- New operators: `scripts/fetch_video_encoder.py`, `scripts/prepare_temporal_cache.py`, `scripts/check_video_offline.py`; `scripts/check_video_flow.py` also supports maximum-window artifacts.
- Tests: `tests/test_video_pretrained.py`, `tests/test_video_temporal.py`, `tests/test_video_inference.py`, expanded `tests/test_video_train.py`.
- Plan/results/decisions: `docs/pretrained_plan.md`, this report, `docs/context.md`, `docs/decisions.md`, `CHECKLIST.md`, `README.md`, `experiments.md`; preserved component license in `licenses/opencv-mobilenet/LICENSE`.

Final full suite: **163 passed, 2 skipped, 12 warnings in 30.19 seconds**. The warnings are the existing LightGBM `eval_set` deprecation; skips require optional ignored mock CSVs. This includes real generated-video worker processing, legacy/pretrained routing, artifact/source/weight guards, offline behavior, full-tuple permutation and outer-fold isolation. Windows loopback page remained HTTP 200. Temporary candidate server on port 8002 was stopped after checks. No commits, pushes, dependency changes, external media uploads or public deployment occurred.

Frozen `src/eval.py` SHA-256 remains `0ef1963df44a04495ecf95d8c40c82c606ac7987ac410202094c5e671483a762`. Prior uncommitted preparation and original/raw files are preserved.
