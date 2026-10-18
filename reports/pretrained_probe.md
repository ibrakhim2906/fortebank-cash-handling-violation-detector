# Direct pretrained probe — operational success, recognition not validated

17 October 2026, Asia/Qyzylorda. The human approved this inference-only experiment. The unchanged pretrained model was run on eight deterministically selected fold-0 training development recordings; all 11 fold-0 validation recordings and all eight reserved recordings were excluded. No fitting, new model/runtime, external training corpus, threshold search or live promotion occurred.

## Decision

Keep the reproducible local inference path and private inspection artifacts. Do not use the unchanged 600-class model as a direct cashier-action or bank-violation engine. Its target-class rankings are weak on this fixed sample, while the most frequent top-ranked outputs are general activities outside the intended cashier taxonomy. This is a domain-fit warning, not a measured misclassification rate. Independent action truth is absent, so neither action accuracy nor action recall can be computed. A zero top-five count is not zero recall.

## Actual measured results

| Measurement | Result |
|---|---|
| Completed sources / windows | 8 / 149 |
| Total source duration | 337.936 seconds |
| Original-pass extraction | 61.48411815000145 seconds total |
| Single-source extraction range | 2.9644293309975183–13.565390534000471 seconds |
| Whole experiment including longest-source repeat | 74.82229420099611 seconds |
| Distinct finite logit vectors | 149 / 149 windows |
| Fresh-encoder repeat | Exactly equal logits, source times, frame indices and frame hashes; same process |
| Python socket connection attempts during inference | 0 |
| Independently verified action labels / action accuracy | 0 / unavailable |
| Before → after | Synthetic-only runtime evidence → real-MOV extraction and exact repeat; no new bank-detection score |

The connection check blocks Python socket entry points, not native network syscalls or the entire OS. These timings measure extraction, not an end-to-end jury API response. The windows overlap and are not independent accuracy samples.

## Fixed target classes: unverified model scores

| Pretrained class | Top-1 windows | Top-5 windows / 149 | Recordings with a top-5 proposal / 8 | Best rank / 600 | Maximum softmax score |
|---|---:|---:|---:|---:|---:|
| counting money | 0 | 0 | 0 | 13 | 0.010383943 |
| looking at phone | 0 | 0 | 0 | 7 | 0.019243600 |
| talking on cell phone | 0 | 0 | 0 | 15 | 0.016147911 |
| texting | 0 | 5 | 2 | 2 | 0.058431386 |
| writing | 0 | 16 | 4 | 2 | 0.073101109 |

These are scores normalized across 600 competing classes, not calibrated probabilities that an action occurred. They do not identify violations, omitted counting, ownership, authorization, intent, counterfeit notes or incorrect totals. Writing and texting supply candidates for local inspection, not established successes.

## Aggregate top-ranked outputs across all 600 classes

| Model-proposed class | Windows |
|---|---:|
| photocopying | 51 |
| using a microscope | 26 |
| cooking sausages (not on barbeque) | 18 |
| opening door | 9 |
| watching tv | 6 |
| pull ups | 5 |
| playing controller | 4 |
| moving furniture | 3 |
| opening refrigerator | 3 |
| using atm | 3 |

These are aggregate model outputs, not descriptions or verified labels of individual recordings. No clip-specific output or pixels were sent to external tools.

## What still needs work

- Generic person-to-person object transfer has no matching pretrained class. It needs a different suitable recognizer or a task-specific head with independently labelled transfers and difficult non-transfer examples.
- Counting and phone-related classes need bank-camera validation. Their weak rankings do not by themselves distinguish absent actions from resolution, viewpoint, preprocessing or model limitations. Source frames are 464 × 848; the fixed 172-pixel letterbox leaves roughly 94 pixels across the actual scene. A separately frozen closer-view comparison could test whether detail is the bottleneck before investing in larger training.
- Writing/texting candidates need independent checks before any success claim. The inspected training videos cannot become untouched test material.
- Rule applicability needs transaction-stage and contextual evidence. The current verified public rule catalog is not an activated violation detector. No violations have been validated by this probe.
- Preserve the existing label-gated transfer experiment and all protected recordings. External labels can supplement training but cannot establish accuracy on this camera.

## Local inspection

Open [the private pretrained probe](http://127.0.0.1:8004/#pretrained), enter the existing API key, and load the queue. Each of the eight recordings has all chronological windows and five target-class scores. The strongest-window shortcut is explicitly model-selected. The mode is read-only, creates no labels, and excludes protected groups. The original general/transfer review workflows remain available.

Actual Windows bridge verification: page, authenticated tasks, summary and selected media returned 200; tasks contained eight recordings and 149 windows. Unauthenticated tasks returned 401; attempted pretrained label writes returned 404. A complete original MOV was transferred only across this computer’s local Windows/WSL loopback path. No media payload was printed.

## Validation and commands

- `UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m scripts.probe_pretrained_actions --stage prepare` — prepared the immutable manifest before decoding.
- Same environment with `uv run --offline python -m scripts.probe_pretrained_actions --stage run` — completed with the measurements above.
- Same environment with `uv run --offline pytest -q` — **880 passed, 2 skipped, 12 existing LightGBM deprecation warnings in 55.21 seconds**.
- Windows Node bridge: eight route assertions passed; existing and new authenticated HTTP routes checked without logging keys or media. Inline review JavaScript syntax checked.
- `git diff --check` — passed.
- No train or shuffled-target command is applicable: pretrained weights, extraction features and decision models were unchanged; no classifier or accuracy metric was fit. Do not fabricate a before/after detection score.

## Files and provenance

New probe/plan/report/tests: `scripts/probe_pretrained_actions.py`, `docs/pretrained_probe.md`, `reports/pretrained_probe.md`, `tests/test_pretrained_action_probe.py`, `tests/test_pretrained_review.py`. The public class map and download provenance are under `licenses/movinet/kinetics_600_labels*`. Existing review API/page and fixed loopback bridges were extended; their tests and progress documentation were updated.

Private derived artifacts are under `data/processed/pretrained-action-probe-v1/`, directory mode 700 and files mode 600. The manifest, per-recording predictions, original sampling audits and repeat are retained locally.
- Manifest SHA-256: `a035e3aea99fc60e6d7454542909a5ceeb8552511f6e6cb43e9d4f02e6ac9a65`.
- Records SHA-256: `28978c4f19e7b7f3dbed56f9c512a6f6ff701b6e3acbb88d21e9cdf9f03eb8ec`.
- Probe source SHA-256: `12f871ea45db86708837550b08165a85b3f781bef843c55385997ffcd273cf65`.
- Unchanged encoder SHA-256: `63323474bfdaa6f7d1aa9cfe038380019da9745329179555fbd8f87575ec5205`.
- Unchanged frozen evaluation SHA-256: `0ef1963df44a04495ecf95d8c40c82c606ac7987ac410202094c5e671483a762`.
- Raw deliveries were not modified. No reserved/blind footage was decoded by this probe; no external uploads, submissions, commits or pushes occurred.
