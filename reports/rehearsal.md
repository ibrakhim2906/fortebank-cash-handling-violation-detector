# Preparation rehearsal — measured results

Date: 2026-10-17. **Synthetic binary rehearsal; the real competition case is still unknown.** The approved implementation and automated end-to-end rehearsal are complete. Visible-browser inspection, human presentation rehearsal and the backup video remain open. No commit, push, deployment or competition submission was performed.

## What changed

| Files | Change |
|---|---|
| `src/artifacts.py` | Shared run configuration, training-only categorical schemas, input/raw-history/source/config fingerprints, artifact hashes, frozen policy checks, checked model loading and submission validation |
| `src/train.py` | Development-only comparisons; explicit one-time holdout/finalization stage; fixed-iteration holdout fit before retrospective all-training refit; isolated experiment log |
| `src/models.py` | Runtime task/seed handling; optional thread limit; deterministic logistic seed; existing model recipes preserved |
| `src/impact.py` | Explicit selected-LightGBM OOF policy; no leftover blend auto-selection; freeze threshold/costs/model identity; cost sensitivity |
| `src/explain.py`, `app/app.py` | Same manifest-checked model/schema; plain-language names; low/high/nearest-threshold examples; app never trains |
| `src/sanity.py` | Exclude reserved holdout labels from pre-finalization scans; JSON diagnostics and nonzero exit on suspicious results |
| `scripts/make_synthetic.py`, `scripts/rehearse.py` | Safe seeded generation, isolated inputs/logs/artifacts, before/after comparison against saved source, full workflow, Streamlit checks and clean replay |
| `tests/test_pipeline.py` | Artifact/policy integrity, holdout fitting order, training-only categories, runtime seed, safe generation, CSV contracts, regression training smoke and no-training demo checks |
| `README.md`, `START_HERE.md`, `PLAYBOOK.md`, `GAMEPLAN.md`, relevant guides | Tested entry points, finalization order, separate checkouts/outputs for parallel work and explicit checkpoint staging |
| `docs/decisions.md`, `docs/prep_plan.md`, this report | Approval provenance, implementation status, measured evidence and remaining checks |

`src/eval.py` and `src/config.py` remain byte-for-byte unchanged. No new dependencies or feature families were added. The earlier context-handoff/permission edits were preserved.

## Canonical run and commands

Canonical run directory: `data/processed/rehearsals/prep-20261017-ready/`.

Manifest run ID: `a08e65c93d7641839dd77a171179e90c`. Delivered model SHA-256: `525e40ece672b9acbb67bb98361025723e53e11900a84651d280cc8e21994ce9`. The full captured test output is `<run>/logs/pytest.txt`.

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache MPLCONFIGDIR=/tmp/datathon-matplotlib DATATHON_THREADS=2 uv run --offline python -m scripts.rehearse --output data/processed/rehearsals/prep-20261017-ready --baseline-source /tmp/datathon-prep-baseline-20261017
UV_CACHE_DIR=/tmp/datathon-uv-cache MPLCONFIGDIR=/tmp/datathon-matplotlib DATATHON_THREADS=2 uv run --offline pytest
UV_CACHE_DIR=/tmp/datathon-uv-cache uv sync --locked --offline
git diff --check
```

The existing baseline snapshot was captured **before implementation**, without a branch or commit, at `/tmp/datathon-prep-baseline-20261017/`. It records the original revision, working-tree changes and protected-file hashes in `snapshot.json`. The baseline harness loads the old dataset/model/evaluation functions, computes development OOF only, and does not invoke the old all-label final refit. The same fixture, folds, seed, features and two-thread limit are used for before/after. Future fresh rehearsals omit `--baseline-source` unless a saved baseline is available; always use a new output path.

The runner records exact subprocess arguments, cwd, exit code, duration and full stdout/stderr in `logs/*.json` and `logs/*.txt`. The canonical run includes:

```text
uv run --offline python -m scripts.make_synthetic --output <run>/input
uv run --offline python -m src.train --models dummy,linear,lgbm,cat --note 'SYNTHETIC rehearsal'
uv run --offline python -m src.sanity
uv run --offline python -m src.impact --oof lgbm
uv run --offline python -m src.train --stage finalize
uv run --offline python -m src.explain
uv run --offline python -m scripts.rehearse --app-check <run>
```

`DATATHON_RUN_DIR` supplies the same runtime context to every stage. All these stages succeeded. Earlier engineering runs are retained locally for provenance; use the canonical directory above for this source version. Repeated engineering checks did not change features, model hyperparameters, split windows or the threshold-selection rule in response to holdout scores.

## Fixture and validation

- Seed 42; 12,000 training rows × 10 columns; 2,000 unlabeled test rows × 9 columns; seven application predictors. Training positive rate 0.19175; no missing fixture cells.
- Train dates 2025-01-01 through 2026-02-28; later test inputs are generated in March/April 2026. Predictions are per unique synthetic `client_id`; this is a fixture choice, not a claim about the eventual prediction unit.
- Explicit development windows, February holdout and an **illustrative 30-day label gap** are set only in the run's `config.json`. The gap is not a verified bank outcome horizon. Group exclusion is off for unique-client fixtures; opt-in group exclusion and repeated-group boundaries remain covered by unit tests.
- Fold schemas are fitted on training rows only; unknown categories map to missing. Pre-finalization sanity scans exclude the February holdout labels. OOF targets outside development prediction rows are saved as missing.

| Fold | Training dates | Validation dates | Train rows | Valid rows | Train target rate | Valid target rate | Shared groups |
|---|---|---|---:|---:|---:|---:|---|
| Development 1 | 2025-01-01–2025-08-31 | 2025-10-01–2025-11-30 | 6,969 | 1,728 | 0.191850 | 0.191551 | Not configured |
| Development 2 | 2025-01-01–2025-10-31 | 2025-12-01–2026-01-31 | 8,646 | 1,708 | 0.193731 | 0.190281 | Not configured |
| Final holdout | 2025-01-01–2026-01-01 | 2026-02-01–2026-02-28 | 10,427 | 763 | 0.190467 | 0.188729 | Not configured |

## Real measured scores

**Before → after LightGBM development ROC-AUC: 0.6874550359712229 → 0.6874550359712229.** The old/new OOF arrays agree within absolute tolerance `1e-12`, including identical missing prediction positions. This is an artifact/validation workflow hardening result, not a model-quality improvement.

| Development model | Pooled ROC-AUC | Fold 1 | Fold 2 | Rehearsal delivery decision |
|---|---:|---:|---:|---|
| Dummy | 0.5000000000 | 0.5000000000 | 0.5000000000 | Comparison floor |
| Logistic | 0.6920210783 | 0.6898100591 | 0.6947839146 | Comparison only |
| LightGBM | 0.6874550360 | 0.6763889820 | 0.6996117693 | Selected delivery path by approved preparation scope |
| CatBoost | 0.6913981620 | 0.6854545887 | 0.6994560320 | Comparison only |
| Equal LightGBM/CatBoost blend | 0.6920468503 | — | — | Comparison only |

LightGBM is not the best displayed synthetic development score. Its delivery path was fixed for this bounded preparation task, not selected using holdout results. Choosing and supporting a different final model is a real-case decision; do not claim the app serves the blend or logistic model.

**Final holdout LightGBM ROC-AUC: 0.6792541733979536 on 763 rows.** The threshold was already fixed at 0.14 and the model at 68 trees. Heldout labels were not passed to model fitting/early stopping/threshold selection. The holdout result was recorded before the separate retrospective refit on all 12,000 supplied labels. Tests explicitly inspect this execution order.

## Sanity and scenario value

The development-only sanity population contained 11,237 rows.

| Check | Actual output |
|---|---|
| Duplicate prediction IDs | 0 |
| Train/test shared prediction IDs | 0 |
| Single-feature AUC > 0.90 | None |
| Shuffled-target LightGBM ROC-AUC | 0.5071452642451217 |
| Shuffled-target dummy ROC-AUC | 0.5000000000000000 |
| Adversarial train/test ROC-AUC | 0.5112854409539913 |
| Reported problems | Empty list; sanity command succeeded |

Scenario assumptions are the existing illustrative defaults: 400,000 ₸ per caught positive and 60,000 ₸ per false alarm. This is a synthetic TP/FP intervention scenario, not calibrated expected loss, measured causal benefit or bank-approved savings. AUC does not establish calibration.

Development selected threshold 0.14 from 3,436 OOF rows, with about 60.0 M ₸ per 10,000 above the better of flag-all/flag-none under those assumptions. This is explicitly development-selected value, not an independent evaluation. `sensitivity.json` records the same locked threshold with false-alarm costs scaled to 0.5×/1×/1.5×.

On the final holdout, the locked policy caught 115 positives and flagged 366 negatives: recall 0.7986111111; fraction flagged 0.6304062910; scenario value 24,040,000 ₸ on 763 rows. These are counts/value at the previously frozen threshold, not a new threshold search. Confirm real intervention effectiveness, costs and capacity with ForteBank before any business claim.

## Artifact, submission and demo evidence

- The manifest binds selected model, run, data/source/config hashes, model parameters, frozen policy and artifact hashes. Loading after a source/config/input/artifact change fails; manually changing the frozen policy is rejected. Reserved holdout outcomes are evaluated only in finalization, which rejects accidental repeats.
- The checked submission has exactly 2,000 test IDs in input order, exactly `client_id,prediction`, no duplicate/missing/fabricated IDs or extra index, and finite probabilities in `[0,1]`.
- Reloaded-model versus CSV maximum absolute probability difference: **9.71445146547012e-17**, below the `1e-12` test tolerance.
- The app's probabilities/decisions/reasons are checked against that same loaded model and policy. Model fitting is patched to fail during app testing; initial load, changes of example and restart succeeded without training.
- The canonical automated app startup was **0.6944697090 seconds** (Streamlit AppTest, measured inside an already-started Python process). It is not a browser-render timing or a full cold OS/process-start timing. Three examples passed: low-risk `101374` at 0.0640915309 → OK; high-risk `101505` at 0.6211800120 → Flag for review; borderline `101735` at 0.1399961015 → OK. Rounding can display the last probability as 14.0%, but its exact value is below 0.14.
- An additional headless loopback server check returned health `ok` and HTML HTTP 200 in **0.7611992860 seconds**, then stopped its own server. Evidence: `data/processed/rehearsals/prep-20261017-final/reports/http_check.json` and `logs/http-server.txt`. That checks server readiness; the canonical three-flow AppTest checks execution of the app itself. Local sockets required sandbox escalation; the test succeeded after escalation.
- Visible Windows browser automation was attempted through the computer-use skill, but initialization failed with `sandboxCwd is not a local file URI: file:///home/ibrak/datathon-kit`. No Windows input was sent. Visible-browser layout/clickthrough and the backup recording remain pending; automated checks are not presented as those manual checks.

## Clean replay and environment limits

The runner copied proposed sources, including new files, into a clean temporary source tree excluding `.git`, `.venv`, data, models, reports and caches. It generated its own fixture and artifacts and reran the full workflow. The replay paths and exact command are recorded in the canonical `result.json` and `logs/clean-replay.json`.

Development OOF predictions, holdout predictions and exported submission matched across the original and replay with absolute tolerance `1e-12`. This is deterministic engineering replay, not a second independent estimate of generalization.

The locked existing `uv` environment was intentionally reused offline with `UV_PROJECT_ENVIRONMENT`; no fresh dependency-download claim is made. Separately, `uv sync --locked --offline` resolved 112 packages and checked 97 installed packages successfully. No dependencies were added or lockfiles changed.

## Tests and integrity

Before this task: **35 passed, 2 skipped in 1.53s** from the preparation audit. After implementation:

```text
collected 54 items
tests/test_eval.py ..................................ss
tests/test_pipeline.py .................
tests/test_point_in_time.py .
52 passed, 2 skipped, 12 warnings in 8.58s
```

The two skipped tests require optional ignored mock CSVs. Essential new integration fixtures run without those CSVs. The 12 warnings are LightGBM's deprecation warning for the existing `eval_set` API; tests passed with the locked version. Streamlit's bare-context warning appears in AppTest logs and is not an app exception.

Nine protected source/template/config/log/lock files match their pre-implementation hashes. Frozen `src/eval.py` SHA-256 remains `0ef1963df44a04495ecf95d8c40c82c606ac7987ac410202094c5e671483a762`. `src/config.py`, the real brief/context/jury templates, `CHECKLIST.md`, real-case `experiments.md`, `pyproject.toml` and `uv.lock` are unchanged. Raw data was absent at the start and no raw files were created, edited or deleted. Rehearsal experiments are isolated in `<run>/experiments.md`; the old reference log is under `<run>/baseline/experiments.md`.

## Kickoff handoff and remaining gaps

1. Supply the complete real brief and untouched data; send mentor/jury comments and resource/access constraints as they arrive. AI/prepared-code permission is already settled. Say `run P1`; review the resulting case plan at the existing checkpoint.
2. Agree prediction unit, target semantics, label availability, validation, official metric and money assumptions before P2. Preserve the frozen evaluation source; its existing options cover this rehearsal.
3. Use fresh run directories for each materially changed setup. Parallel chats need separate worktrees/clones **and** output directories. Merge kept source changes, then revalidate; do not combine stale generated OOF/model files.
4. Current delivery supports LightGBM. Other models/case types need deliberate adaptation. Regression training and fold behavior have smoke coverage, but the full end-to-end rehearsal is binary tabular; it does not establish every regression, forecasting, text or ranking workflow.
5. Complete visible-browser clickthrough on the presentation machine, record a one-minute backup video, verify its playback and rehearse the real pitch with a timer. Those human checks are pending. No slide deck or claimed business insight has been fabricated for the unknown case.
6. A truly fresh dependency installation on another machine/network remains unverified. Provide the locked environment/data explicitly when sharing the project; ignored artifacts are not in Git. Runtime fingerprints are accidental-mismatch guards, not a security boundary against deliberate local tampering.

To show the verified synthetic run manually:

```bash
DATATHON_RUN_DIR=data/processed/rehearsals/prep-20261017-ready DATATHON_THREADS=2 uv run streamlit run app/app.py
```

### Two-minute practice narrative (draft; not yet rehearsed)

We prepared a complete workflow that turns an application into a risk estimate, reasons and a review decision. This demonstration uses synthetic data. Once the real case arrives, the bank's decision and the official evaluation will determine what we build.

The first thing we establish is what information exists when the decision happens. We separate earlier applications from later ones, allow a delay for outcomes to become known, and reserve a final period for evaluation. Our checks also cover repeated clients, missing history and accidental reuse of incompatible artifacts.

In this rehearsal, the dummy baseline scored 0.50 ROC-AUC. The delivered LightGBM model scored about 0.687 on development and 0.679 on the reserved period. Hardening the workflow preserved its development predictions. The final period never chose the threshold or number of trees.

Here are three applications: low risk, high risk and one close to the review threshold. Each shows the same probability used in the exported prediction file and five reasons from the same model. We checked that loading and restarting the demonstration never retrains it.

The value chart uses explicit illustrative benefits and false-alarm costs. We select a policy on development predictions, then measure that locked policy on the reserved period. ForteBank's actual intervention, costs and review capacity must replace the illustration before we claim a business benefit.

Finally, we repeated the workflow in a clean source tree and reproduced the predictions. Our next inputs are the real brief, data and mentor feedback. Those establish the actual problem, the useful insight and the final presentation.

Backup recording outline: show the synthetic label → high-risk application/reasons → low-risk application → threshold-borderline application → value chart/assumptions. Record only after visible-browser inspection succeeds; replace the synthetic recording with the actual case demonstration before pitching.
