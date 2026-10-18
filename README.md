# ForteBank cash-handling violation detector — source review snapshot

A self-hosted video-review service with typed timecoded concerns and original-frame evidence. Start with [review context](docs/REVIEW_CONTEXT.md), then `service/pipeline/analyze.py`, `auditor.py`, `vlm.py` and `state_ledger.py`. The selected recipe is32B AWQ on one A40,16images/16384context/60s call ceiling, storage veto enabled.

Measured DEV result:23/30correct verdicts,TP12/FN4/FP3,5/16events within±2s. These results fall short of the desired recognition standard. All40supplied recordings were historically reviewed; the10-video reserve is prospective regression. A new40-video actual API qualification is running, with no terminal outcome asserted in this snapshot.

This snapshot contains code, tests, UI, approved dependency manifests and public license notices. It excludes bank media, labels, detailed per-video reviews/errors, keys, job evidence, weights and downloaded executables. It is suitable for static review; it does not include the external model or numeric fallback artifact needed for complete inference.

Use `uv sync` for the exact dependency lock and `uv run pytest` for source checks. The environment usesPython3.12. No new dependency is requested by this export. The source evaluators are copied byte-for-byte. See `EXPORT_MANIFEST.json` for exact hashes, base commit and export-only legacy-helper redactions. AGENTS.md is copied unchanged from the original read-only instruction file.

For the requested audit priorities see `docs/REVIEW_CONTEXT.md`. The optional code-only native proposal is under `proposals/`, explicitly unapplied and not deployed.

Public-export test scope: four named integration tests have explicit conditional skips solely for omitted private case protocol/split, private evaluation source reports, or intentionally redacted six-recording diagnostic defaults. Their assertions remain unchanged and the original workspace tests are untouched. All synthetic unit tests remain enabled. `experiments.md` is a sanitized aggregate log because synthetic pipeline tests require that public path. Local verification can reuse an existing approved uv environment with `UV_PROJECT_ENVIRONMENT`; no new packages were installed. Exact test names/reasons and final verification results are recorded in `EXPORT_MANIFEST.json`.

`deploy/mission-best.env` records the exact selected image-list serving environment for code review; it does not activate the unapplied native-video proposal. Its exported SHA256 is recorded in the manifest after the selected configuration file is installed.

Actual public-export verification: **1196passed,6skipped,12warnings in45.55s**. The four added conditional skips apply only to the explicitly omitted private fixtures/defaults; two other skips already existed. This run used the existing approved uv environment, twoCPU affinity and twoBLAS/OpenMP threads. It verifies source behavior with synthetic fixtures, not bank recognition or a working downloaded model.
