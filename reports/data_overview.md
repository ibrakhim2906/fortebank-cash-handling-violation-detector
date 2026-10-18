# Official release inspection — P1 complete

Inspected 2026-10-17 directly from human-provided `/mnt/c/Users/ibrak/Downloads/open.zip` and `/mnt/c/Users/ibrak/Downloads/Кейс_ForteBank.docx`. A website download URL was not supplied. The source files remain untouched; no footage or frames were transmitted to external tools. The brief's text/tables are preserved in `docs/brief.md`. After the duplicate issue was reported, the human delegated conservative handling and continuation of P1.

## Preserved delivery and derived inventory

Original copies are in `data/raw/fortebank-20261017/`; no pre-existing raw file was changed. Working extraction, `inventory.csv`, `provenance.json` and `environment.json` are under `data/processed/fortebank-20261017/`. Every extracted member was SHA-256 checked against its archive bytes. These data paths are ignored by Git and require intentional provisioning in another checkout.

| Original | SHA-256 |
|---|---|
| open.zip | `c01ee47563620f9cef14cc42fd23d81f4f58600e5d1eae39d97bb85b57961867` |
| Кейс_ForteBank.docx | `f73a951c87aca9d540076aacc5b4fd8a688810a4ec827b1a382512c770d278a0` |

## Inventory and annotation schema

The archive contains 42 `.mov` videos plus `open/open_labels.xlsx` (5,701 uncompressed bytes). Video member sizes range from 2,426,840 to 28,718,927 bytes. The listing contains 45 directory/file entries in total, including two directories. Original archive paths and filename characters remain authoritative; there is no train/test directory split.

The spreadsheet has 42 rows × 3 columns, one record per named video. Inspection used pandas/openpyxl through `uv run --offline python`; no model was fitted.

| Column | Observed pandas dtype | Missing | Missing % | Nonmissing distinct | Meaning |
|---|---|---:|---:|---:|---|
| наименование файла | str | 0 | 0 | 42 | Video stem, with Latin X or Cyrillic Х |
| таймстемп | str | 20 | 47.619 | 19 | Text event timestamp; blank on negatives |
| нарушение / без нарушения | str | 0 | 0 | 2 | Video verdict label |

Labels: 22 violations and 20 no-violation videos. Positive timestamps are all populated. Timestamps have multiple textual formats, e.g. `22 секунда` and `1минута 8 секунда`; preserve raw text and later create a parsed derived column. Treat absent negative timestamps as not applicable, not as events at zero seconds.

Examples of annotation rows, not visual interpretations:

| Filename stem | Timestamp annotation | Video label |
|---|---|---|
| X-1 | 22 секунда | нарушение |
| X-2 | 10 секунда | нарушение |
| Х-3 | blank | без нарушения |
| Х-4 | 57 секунда | нарушение |
| Х-47 | 1минута 8 секунда | нарушение |
| Х-48 | blank | без нарушения |

## Confirmed duplicate footage: reviewed handling

Archive member size/CRC comparisons identified two candidate groups. Direct SHA-256 of decompressed member bytes confirms exact duplicates:

| Videos | SHA-256 | Labels / concern |
|---|---|---|
| Х-4.mov; Х-47.mov | `5385be9420e6a70547d6bcfd0b13d60f9c0738877645afa277d09550a96b8445` | Both positive; annotated at 57s versus 68s. Could be two valid events or an annotation discrepancy; cannot infer which. |
| Х-5.mov; Х-48.mov | `1f5840bff58f298e89c4afa00cb1c18f6dafc5c4096c251d9b507604ebefe906` | Both negative. |

Full content hashing now confirms exactly 40 distinct video byte sequences: 21 positive and 19 negative. A perceptual/recording-session similarity audit is still pending. Never treat different filenames as independent observations. Authorized handling: preserve every raw row, group these pairs into the same evaluation partition, count each content group once in metrics, and exclude the ambiguous positive group's timestamp error from headline localization until reviewed. Keep that group's positive video label for detection. Do not silently choose 57 or 68 seconds. Twenty positive unique clips currently have eligible timestamp annotations.

The initial AGENTS.md rule 11 stop was reported. The human then said “do what you think would be best decision”; the above policy resolves it without guessing labels. Ingestion and header inspection resumed; no pixels were decoded and no features/models were implemented.

## Relationships and missing semantics

The relation is spreadsheet filename stem → archive video basename. The inspection script asserts unique annotation names, equal video/label row counts and exact equality of both identifier sets before reading labels by exact key. No normalization or tabular merge is performed. No person-level or session-level keys were supplied. Latin `X` in X-1/X-2 and Cyrillic `Х` in later stems must not be silently conflated.

The videos have no tabular feature columns or supplied recording dates. QuickTime header inspection measured the following without decoding frames:

| Property | Measured result |
|---|---|
| Movie duration | 11.634–142.534 seconds; mean 34.385262 |
| Average sample FPS | 29.758065–30.000000 |
| Video tracks | Exactly one per file |
| Track width × height | 464 × 848 for every video; display matrices retained in inventory |
| Codec | `avc1` (H.264) in every file |
| Audio tracks | Zero in every file |
| Positive annotation bounds | All 22 raw timestamps are nonnegative and below their movie duration |

These are container/sample-table facts, not a successful decoder/playback test. Human/local semantic inspection of development clips, event taxonomy and perceptual/session grouping remain P2 tasks. The brief describes staged overhead footage; that visual claim has not been verified by external inspection. Capture dates and camera-angle generalization remain unverified.

## Leakage and evaluation risks

- Exact duplicates can cross folds unless grouped by content.
- Frames/windows from the same clip must always remain together. Near-duplicate takes, actor/background and session similarity can also make random clip splits optimistic.
- Filenames, label sheets and annotated event times are supervision, not inference inputs. New uploads must be processed independently of names and training annotation lookups.
- Sparse one-time annotations do not establish that all events are labeled. Event type/boundary labels are absent; type accuracy requires reviewed labels.
- Temporal information inside a video is permissible for retrospective full-clip analysis. It does not justify chronological train/test splits without recording/session dates.
- Hidden test files must not enter training, threshold selection or reusable external storage.
- No single-feature scores, adversarial tests, shuffled-label results or model metrics exist for this real case yet. Tabular synthetic scores cannot be transferred to it.

## Environment and pending checks

WSL exposes AMD Ryzen 7 7730U, 16 logical CPUs and approximately 18 GiB RAM. `/dev/dxg` and `/dev/nvidia0` are absent. `ffmpeg`, `ffprobe` and `nvidia-smi` are not on PATH. Import checks found no OpenCV, PyAV, Torch, Torchvision, Transformers, ONNX Runtime or FastAPI; Uvicorn, multipart support and Pillow exist. This establishes the inspected WSL environment, not the capabilities of every host or future server. No packages were installed. Use CPU-compatible planning defaults and benchmark before making a latency claim.

Minimal next-phase package proposal: `opencv-python-headless` for local decoding/frame operations and `fastapi` for the required upload/jobs API. Inspect actual licenses and add via `uv add` only after approval; no model package/weight download is implied. Public URL/API deployment remains to be designed within the local-inference rule.

## Reproduction

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.inspect_video_release --archive /mnt/c/Users/ibrak/Downloads/open.zip --brief /mnt/c/Users/ibrak/Downloads/Кейс_ForteBank.docx
```

This copies new delivery files only, refuses to overwrite different bytes, asserts safe archive paths and hash equality, validates exact annotation/video correspondence, parses QuickTime timing metadata and writes the derived inventory/provenance. It never trains a model, decodes footage or calls a network service. `environment.json` records the separate local environment inspection.

The inspection was rerun with before/after hashes of both preserved source files, `inventory.csv` and `provenance.json`. All four remained byte-identical. Assertions also confirmed 40 content groups, consistent video labels within every group, 20 eligible positive localization rows, exclusion of both ambiguous annotation rows, and no out-of-bounds timestamps. Original Downloads files still match the preserved delivery hashes. `git diff --check` passed. Frozen `src/eval.py`, `src/config.py`, `pyproject.toml` and `uv.lock` have no changes.

P1 deliverables are complete: source brief, case context, extracted checklist, data overview and implementation plan. This is a completed discovery phase, not a passed model/leakage audit. Next checkpoint: approve `docs/plan.md` and the exact next-phase dependencies, then freeze the evaluation protocol before semantic inspection of held-out footage. No actual training scores exist for this case.
