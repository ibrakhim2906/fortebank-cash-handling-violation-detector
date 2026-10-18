# Local Qwen action-observation detector

The human approved a standalone CPU runtime and local vision-model download, and requested autonomous action-label drafts and detection improvement. This experiment uses fixed pretrained observations as inputs to the original binary video labels. Model-generated observations are not independent action labels. Detection results do not establish action-type accuracy or legal grounding.

## Frozen comparison

The original content-group protocol is unchanged: 32 development recordings, three folds, 17 positive and 15 negative labels. Eight reserved recordings remain undecoded and unevaluated. Protocol fingerprint: `4dc3881396670f9496e6958178589cbe08325a6b1c756d9ea2257fff803fae3d`.

`docs/action_detection_trial.md` predeclares four uniform windows at duration fractions 1/8, 3/8, 5/8 and 7/8, three chronological samples per window and a 384-pixel maximum image side. Neither original labels/timestamps nor human review status select these windows or enter the vision prompt. Nineteen present/unknown/visibility indicators per window become 57 fixed max/mean/std clip features. Logistic regression and shallow LightGBM use the established fold-fitted recipes and threshold 0.5.

The promotion gate is development ROC-AUC at least 0.60, balanced accuracy above 0.50, recall at least the numeric baseline's 10/17, and passed sanity/runtime checks. Timestamp regression must be reported separately. These gates were fixed before the scores. Development selection on this small repeatedly used sample remains uncertain; no official or holdout score is claimed.

## Measured result — reject promotion

The completed training command returned:

| Development metric | Numeric linear baseline | Qwen observations + linear | Qwen observations + shallow LightGBM |
|---|---:|---:|---:|
| ROC-AUC | 0.5098039215686274 | 0.5000000000000000 | 0.48039215686274517 |
| Balanced accuracy | 0.4607843137254902 | 0.5000000000000000 | 0.5000000000000000 |
| Positive clips caught | 10/17 | 17/17 | 17/17 |
| False alarms | 10/15 | 15/15 | 15/15 |
| Precision | 0.5000000000000000 | 0.5312500000000000 | 0.5312500000000000 |
| F1 | 0.5405405405405406 | 0.6938775510204082 | 0.6938775510204082 |

The AUC-selected recipe is linear. Both candidates flag every video, matching the all-positive dummy's thresholded behavior. Higher recall and F1 here do not establish a useful detector. Neither passes the predeclared promotion gate; the numeric service remains available only for flow testing.

Candidate localization also regressed across the same 16 eligible positive development clips: mean absolute error **6.189583333333333 → 12.683333333333332 seconds**, median **5 → 11.083333333333334 seconds**, and within-three-second rate **0.3125 → 0.0625**. This evaluates candidate timestamps independently of the video verdict and retains the exclusion of the conflicting duplicate annotation.

The matched sanity command completed with **20 shuffled-target trials, mean AUC 0.5, standard deviation 0, zero suspicious single features and zero shared/decoded holdout groups**. The complete numeric audit found **32/32 valid cache records, one unique clip feature vector and zero nonconstant features**. No empty/missing records were substituted, and none were entirely unobservable. These observations are operationally valid but contain no useful between-video variation for these classifiers. The audit does not verify action meaning. Bulk label drafting is paused pending a controlled synthetic diagnostic and human review of the three pilot proposals; generating more uninformative guesses would not provide verified supervision.

## Decoder recovery and provenance

The first run stopped after 21 cached development recordings because one video failed timestamp and frame-index seeking near its last frame. Sequential decoding confirmed all 473 declared frames were readable. The repair retains the successful timestamp path, then tries bounded frame-index seeking and sequential decoding from the start. It never substitutes an empty frame or feature vector.

The initial run is preserved at `data/processed/video-qwen-actions/`. A fresh run at `data/processed/video-qwen-actions-v2/` includes a checked migration of 22 records: the 21 completed recordings and the earlier independent runtime benchmark. `scripts/migrate_qwen_decoder_cache.py` verifies the old/new source snapshots, unchanged non-decoder AST, model/prompt/config provenance, original content hashes and development-only membership. It confirms copied descriptors and timestamps are unchanged. The source change does not alter the successful extraction path, classifier, labels, sampling recipe or evaluation protocol. Details are preserved in the fresh run's `decoder_migration.json`.

## Engineering checks

Full suite: **316 passed, 2 skipped, 12 warnings in 60.05 seconds**. Skips require optional ignored mock data; warnings are existing LightGBM deprecations. Three decoder regression tests cover the successful original path, sequential recovery and explicit failure on premature EOF. Additional checks cover immutable development queues, proposal/truth separation, review revisions, local request restrictions, pinned assets, cache migration and numeric observation integrity.

The supplied clips, sampled images, model responses and clip-specific descriptions stay on the computer. The CPU model listens only on `127.0.0.1:8003`, with logging disabled. The operator review page is available through the Windows loopback bridge at `http://127.0.0.1:8004/`. The cloud coding assistant receives only aggregate status and numeric evaluation results.

The final direct-inference check processed the 78.734-second longest development recording in **45.5089147510007 seconds**, with four fixed local-model connections and **zero external Python connection attempts**. Python connections were denied except literal `127.0.0.1:8003`; this is not an operating-system firewall audit or a worst-case concurrency guarantee. The saved result is `data/processed/video-qwen-actions-v2/offline_check.json`. Windows loopback health checks returned **HTTP 200** for both the existing detector on 8001 and the review page on 8004. The candidate is not served or promoted because its detection gate failed; an all-positive model would also be unsuitable for the two-verdict API quality demonstration.

Three controlled synthetic-only requests checked the model separately from supplied footage: red and blue images returned their respective colors under the same color schema, while an abstract drawn handoff under the unchanged action recipe returned unknown observations. Image ingestion and schema variability work; the drawing does not establish recognition accuracy on real actions or prove a particular cause of the constant features. The conservative prompt, limited spatial/temporal samples and model capacity remain possible limitations. Unknown observations were not forced into positive or absent labels.

## Reproduction

Start the verified local model in a separate terminal, then run from the repository root:

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.serve_action_model
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m src.train --task video --encoder qwen-actions --run-dir data/processed/video-qwen-actions-v2 --baseline-run-dir data/processed/video-baseline-comparison --models linear,lgbm --note 'local Qwen observations four uniform windows with decoder recovery'
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m src.sanity --task video --run-dir data/processed/video-qwen-actions-v2
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.audit_action_observations --run-dir data/processed/video-qwen-actions-v2
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline python -m scripts.check_video_offline --run-dir data/processed/video-qwen-actions-v2
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline pytest -q
```

Training requires a fresh run directory, or the documented compatibility-checked migration; do not overwrite an existing completed experiment. Runtime/model assets and original data are local ignored files. Source and license pins are documented in `docs/local_action_model.md`.
