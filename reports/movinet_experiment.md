# First local temporal experiment — runtime ready, human labels pending

17 October 2026, Asia/Qyzylorda. The human approved the experiment, prioritized recognition/backend quality, and explicitly approved the exact LiteRT runtime. Existing UI/API services and general label proposals are preserved. No bank footage was decoded by this new encoder and no reserved footage was used during this turn.

## Measured results

| Check | Actual output |
|---|---|
| Installed runtime | ai-edge-litert 2.3.0; only four packages added, with backports-strenum 1.3.1, flatbuffers 25.12.19 and ml-dtypes 0.6.0 |
| Exact weights | MoViNet-A0-Stream float16 TFLite v2; 7,992,000 bytes; Apache 2.0 confirmed for this exact model variation |
| Signature | float32 image `[1,1,172,172,3]`, 43 streaming states, float32 logits `[1,600]` |
| Full benchmark model load | 0.05483120999997482 seconds |
| Eight-second synthetic window, two passes | 0.3911503570052446 / 0.38038880599924596 seconds |
| Repeatability | Exact equality; maximum absolute logit difference 0.0 |
| Long synthetic source | 145 seconds, 30 FPS, 4,350 source frames; MJPEG/AVI |
| Temporal extraction | 725 sampled source frames, 70 overlapping windows, 2,795 model-frame invocations |
| Long-source extraction time | 26.42695176399866 seconds |
| Whole benchmark including source generation | 34.49151513400284 seconds |
| Peak process RSS | 208,691,200 bytes |
| Python-network-denied synthetic check | Zero connection attempts; 600 finite logits; exact repeated output |
| Windows review connectivity | Page 200, authenticated transfer queue 200, summary 200 |
| Label gate | 0 eligible positives and 0 hard negatives in each fixed pool; train and sanity both exit 2, blocked without scores |
| Full existing and new test suite | 835 passed, 2 skipped, 12 existing LightGBM deprecation warnings in 43.54 seconds |
| Focused rerun after evidence-report wording correction | 38 passed in 4.86 seconds |

The earlier isolated two-window run was 0.377667016 / 0.353899759 seconds with the same exact-repeat result. The full benchmark above is the principal recorded timing. These measurements use synthetic media only; they do not establish action accuracy, precision/recall, final end-to-end MOV/API latency, or any jury score. The Python socket guard is not system-wide network isolation. No before/after bank-detection score exists for this experiment.

Machine-readable records, runtime signature and frozen source snapshots: `data/processed/movinet-synthetic-benchmark-v1/`. Model/weight/license provenance: `licenses/movinet/provenance.json`. Weight SHA-256: `2afdb24bc127104171df2d133245beb3b8828705cc8d93f1aeb1e24392838924`.

## Human review needed

The existing private review page at `http://127.0.0.1:8004/` now defaults to **Object-transfer pilot**. Its exact queue is `data/processed/action-labels/transfer-pilot-v1/queue.json`: 33 tasks from 32 development recordings; the first 16 cover distinct recordings, eight from each fixed pool. The eight reserved recordings are excluded. Six old model proposals and the existing general review journal remain preserved.

For each task, independently choose transfer / same-person relocation / contact without transfer / background / uncertain. Confirm the full event interval and an original supporting frame for a transfer (at most eight seconds); verify at least eight seconds of clear no-transfer coverage for a counterexample. A visible transfer is an action, not automatically a bank-rule violation. The exact steps are in `docs/transfer_review.md`.

The fixed pilot requires four genuine transfers and four relocation/contact counterexamples per pool, all from distinct recordings. Screening tasks are not prelabelled or guaranteed to fill those quotas. Background and uncertainty do not fill the hard-negative quota. Reviews stay local; no pixels, per-clip descriptions, labels or keys need to enter chat.

## Implemented path and limits

The encoder uses full chronological sampling at five FPS, eight-second windows with two-second stride and fresh streaming state per window. A 600-logit representation feeds a fixed scaler/logistic classifier. The metadata gate prevents training with guessed action labels. A fitted head and its hashes must be frozen before blind-pilot labels are released. Source/group checks and twenty shuffled-label fits are prepared; with zero eligible labels, they correctly produce no accuracy or shuffled-target score.

The classification feasibility flag cannot authorize deployment. Fresh real-clip extraction repeatability, confirmed supporting-frame alignment and semantic review remain required for the complete pilot gate. A sampled frame inside the annotated interval alone is reported only as interval alignment, not verified supporting-frame relevance. No public rule mapping is activated by this action pilot.

## Commands and files

Commands run through uv: `uv add 'ai-edge-litert==2.3.0'`; `python -m scripts.provision_movinet`; `python -m scripts.prepare_transfer_review`; `python -m scripts.benchmark_video_movinet --run --output-dir data/processed/movinet-synthetic-benchmark-v1`; `python -m src.train --task transfer-pilot --stage prepare`; `python -m src.sanity --task transfer-pilot`; `pytest -q`; focused `pytest -q tests/test_video_transfer_pilot.py`. Also ran an inline uv Python synthetic socket-denial check and a Windows-local HTTP connectivity check without printing the key or media. `git diff --check` passed.

Main additions: `src/video_movinet.py`, `src/video_transfer_pilot.py`, `scripts/provision_movinet.py`, `scripts/benchmark_video_movinet.py`, `scripts/prepare_transfer_review.py`, their focused tests and model-license evidence. Existing train/sanity dispatchers, private review adapter/page and Windows/Python review bridges now expose the label-gated pilot. `pyproject.toml` and `uv.lock` record the approved runtime. Plan/context/decisions and operator documentation were updated.

Both original raw-delivery hashes still match provenance. Frozen `src/eval.py` remains SHA-256 `0ef1963df44a04495ecf95d8c40c82c606ac7987ac410202094c5e671483a762`. No commit, push, external upload, public deployment, submission or holdout decoding occurred.
