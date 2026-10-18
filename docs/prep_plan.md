# Preparation plan: harden the kit and rehearse end to end

Status: **approved by the human and implemented on 2026-10-17**. Automated end-to-end rehearsal and clean replay completed; measured results and remaining browser/video/human checks are in `reports/rehearsal.md`. The original scoped plan follows for traceability.

## Objective and scope

Make it possible to supply the real brief, untouched data, jury/mentor feedback and operational context, then build a reliable solution through the existing P1–P13 checkpoints. A successful rehearsal should prove that one selected model, its predictions, explanations, threshold, business-value report, submission and demo agree.

The actual case, metric, scoring weights and submission format remain unknown. AI tools and prepared code are confirmed allowed in `docs/decisions.md`. Synthetic results will demonstrate the kit, not predict competition performance. The earlier 8/10 and proposed 9/10 are readiness judgments, not measured scores or promises of a prize.

Keep the existing plain-function architecture and installed dependencies. No new dependencies, feature families, tuning campaign, extra app pages or deployment service are part of this preparation. Preserve raw data and the case templates. `src/eval.py`, `score()` and `cross_validate()` stay unchanged; use their existing development/holdout API. Preserve generic config defaults and keep rehearsal settings outside `src/config.py`.

## Current evidence and gaps

Latest completed audit: `uv run --offline pytest` produced **35 passed, 2 skipped in 1.53s**; the two optional mock-data tests skipped because the CSV was absent. The lockfile check and approved-library/pipeline import checks passed. No end-to-end model run or demo was performed in that audit.

| Gap | Current behavior | Required outcome |
|---|---|---|
| Model/threshold inconsistency | `src.train` saves/submits LightGBM, while `src.impact` prefers blend OOF files when present and the app loads LightGBM | Every delivered output identifies and uses the same selected model and current run |
| Missing final holdout evaluation | `src.train` evaluates development folds then refits on all labels | Evaluate a frozen setup on the reserved holdout before final all-training refit; report development and holdout results separately |
| Unsafe rehearsal data | `scripts/make_synthetic.py` writes into `data/raw/` with no overwrite protection and uses seed 0 | Generate fixtures outside raw data using `config.SEED`; reject occupied output paths |
| Parallel/output collisions | Operator docs suggest multiple branch chats without requiring separate working directories; commands overwrite common artifacts | Separate checkouts and output directories for concurrent work; verify artifact identity when consuming results |
| Incomplete checkpoint instructions | `git commit -am` omits new files, and one documented commit has no staging step | Explicit file staging and diff review; preserve unrelated work |
| Untested whole flow | Unit tests cover validation and one aggregation pattern | Run data inspection → development → sanity → threshold → holdout → final fit → submission → explanations → demo, including a clean replay |

## Implementation sequence and acceptance checks

Before changing code after approval, save a temporary baseline source snapshot and record the current revision, relevant working-tree changes, generic config and raw-file hashes. Use that saved source for the before/after comparison; do not accidentally compare the hardened implementation with itself. This requires no new branch or commit.

### 1. Bind delivered artifacts to a selected model and run — 40 minutes

Likely files: `src/train.py`, `src/explain.py`, `src/impact.py`, `app/app.py`, and a small shared artifact helper in `src/` if needed.

- Introduce an explicit run directory for produced artifacts and their experiment log, keeping current default entry points compatible. The rehearsal will always use its own directory. Thread the same run context through training, explanation, impact and app loading.
- Write a small manifest with a run ID, selected model, task/metric/seed, feature schema including categorical levels, data/config fingerprints, artifact paths and frozen policy settings. Validate compatibility when loading artifacts; avoid automatic selection by leftover file existence.
- **Bound the first implementation:** LightGBM is the supported delivered model for this rehearsal. Dummy, logistic, CatBoost and blend may be compared on development folds, but comparison outputs must not silently become the app/submission model. Clearly label this distinction in reports. Supporting CatBoost/blend delivery can be considered after the real case warrants it.
- Select the LightGBM threshold from that run's development OOF predictions. Bind it and the cost assumptions to the same manifest/model identity. Explain the same LightGBM pipeline in the demo and use it for the CSV. Comparisons with other models remain visible in the experiment table.
- Persist/restore the training feature schema and categorical mappings consistently for heldout/test/app rows. Fit learned preprocessing on training partitions only.
- Make the demo's borderline example the application nearest the chosen threshold, with low/high examples selected from the same run's predictions.

Pass criteria: saved-model, reloaded-model, submission and app probabilities agree within a stated numerical tolerance on identical rows; their threshold/model/run identities agree. Stale or incompatible artifacts fail clearly. A run without a supported delivered model fails clearly rather than loading an old LightGBM artifact.

### 2. Add an explicit final holdout stage — 45 minutes

Likely files: `src/train.py`, `src/impact.py`, `src/sanity.py`, `src/models.py` only where necessary for a fixed-iteration final fit, and `docs/guides/validation.md`.

- Separate development from finalization. Development uses only `make_folds(df)` for model comparison and policy selection; it produces no final all-training model that could be mistaken for an untouched-holdout model.
- Keep pre-finalization feature/target scans, sanity-driven feature decisions and insight selection on the development population. The current single-feature scan in `src.sanity` uses all supplied labels; make it respect the reserved holdout so it cannot influence feature selection before final evaluation.
- Freeze selected model parameters, iteration count from training-only early stopping, feature/preprocessing recipe, any calibration and threshold/cost assumptions before final evaluation. Do not add calibration to the preparation unless an existing probability-to-money claim requires it.
- Finalization calls `make_folds(df, holdout=True)`, fits preprocessing/model only on its eligible training rows, uses the frozen iteration count without holdout-based early stopping, and scores predictions on the holdout. It must not fit or tune with holdout labels.
- Report holdout ROC-AUC and locked-threshold confusion counts/value separately from development results. No threshold search or optimistic value maximization on holdout labels. If no holdout is configured for a generic run, report that explicitly; the rehearsal must configure one.
- Persist the frozen policy and holdout evaluation record. Accidental repeated finalization should fail or be explicitly labeled a repeat; a local guard cannot replace the human rule against tuning after seeing the holdout.
- Only after recording the holdout result, refit for the retrospective submission using all supplied training labels and the frozen settings. Preserve the holdout evaluation evidence and distinguish the submission refit from the model used to measure holdout performance. A live-use model may only use labels available at its actual training date.

Pass criteria: test spies show holdout labels never reach fitting, early stopping, calibration or threshold selection; the holdout metric is computed only after predictions exist. Final all-training refit follows the recorded holdout stage. No changes to frozen evaluation code are needed.

### 3. Make rehearsal inputs and outputs safe — 20 minutes

Likely files: `scripts/make_synthetic.py`, a new `scripts/rehearse.py`, and only minimal runtime/path plumbing in `src/`.

- Give the synthetic generator an explicit output directory under `data/processed/rehearsals/<run-id>/input/`; refuse `data/raw/` destinations and existing output files. Use `config.SEED` and avoid work at module import.
- Generate a deterministic, clearly labeled mock binary case with sufficient dates/classes for development and a later holdout. Treat any illustrative label gap and economic costs as rehearsal assumptions, not bank facts. Keep the final test input unlabeled.
- A small rehearsal runner supplies filenames, IDs, split windows, gap and run-specific output/log paths at runtime. It must not rewrite generic config defaults or populate the real brief, jury/context templates or real-case experiment log with synthetic claims.
- Snapshot raw-file hashes and generic config before rehearsal and compare afterward. All fixtures, fitted artifacts, OOF predictions and logs stay in the rehearsal directory. Never clean up with wildcard deletion of `data/raw/`.

Pass criteria: repeat generation is deterministic; unsafe/occupied destinations are rejected; raw/config snapshots are unchanged after successful and intentionally failing runs. No new package or external paid service is required.

### 4. Repair the operator workflow — 15 minutes

Likely files: `START_HERE.md`, `GAMEPLAN.md`, `PLAYBOOK.md`, `README.md`, and relevant guides.

- Specify a separate Git worktree/clone **and output directory** per concurrent chat; changing branches in one shared checkout is not isolation. Document how to bring over a kept feature and revalidate it on the frozen split.
- Use the current checkout for the primary implementation unless isolation is needed; no new branches, commits or agent delegation are implied by this plan. Separate-checkout instructions are preparation for human-authorized parallel tasks.
- Replace incomplete checkpoint snippets with explicit staging of intended files, `git diff --cached` review and commit. Do not stage unrelated existing changes automatically.
- Add the explicit holdout/finalization checkpoint, selected-artifact rule and safe rehearsal command to the phase prompts. Keep the AI/prepared-code permission confirmed.
- Explain that a fresh checkout needs intentionally provisioned rehearsal fixtures; ignored real data/models are not magically present after cloning. Keep the real-case setup and synthetic rehearsal clearly distinguished.

Pass criteria: documented commands match implemented flags; no instruction depends on a teammate's uncommitted files, unsafe raw-data cleanup, implicit staging or a previously generated model.

### 5. Add focused regression and integration tests — 30 minutes

Likely files: new tests beside `tests/test_eval.py` and `tests/test_point_in_time.py`.

- Test that prediction, threshold, explanation and CSV consumers use the selected run/model; reject stale, missing and incompatible artifacts.
- Test the holdout execution order and label isolation, frozen threshold/iterations, and separation of development and holdout metrics.
- Test the generator's seed/output guards and isolation of config/raw data/run directories.
- Validate exported CSVs: configured ID and prediction columns, exact input-ID coverage/order, no duplicate/missing/fabricated IDs, no index column, finite output and binary probabilities in `[0,1]`. Keep checks generic for regression outputs where appropriate.
- Add a small deterministic pipeline smoke fixture so essential coverage does not depend on ignored raw CSVs. Preserve all existing tests, including no-time and regression fold behavior.
- Use Streamlit's installed testing facilities to exercise the three selected examples, reasons and policy display. Check that loading/rerunning the app does not invoke any model fitting.

Pass criteria: `uv run pytest` passes, required integration coverage runs without real case data, and failures are actionable. Optional raw-file tests may still skip; record skips separately. `git diff --check` passes.

### 6. Run the full isolated rehearsal and clean replay — 45 minutes

Commands/options below are **planned interfaces**, to be implemented and tested after approval. The runner is responsible for applying runtime settings consistently in each subprocess, capturing exact commands and exit codes, and enforcing stage order. It is a small local script, not a new orchestration framework.

Proposed one-command entry point:

```bash
uv run python -m scripts.rehearse --output data/processed/rehearsals/<fresh-run-id>
```

The runner will execute and record this sequence:

1. Check the locked environment, create the isolated synthetic input and print a compact data overview plus date ranges/class rates. Do not copy synthetic case assumptions into the real templates.
2. Print the development/holdout fold table with train/validation dates, row counts, class rates and shared groups. Exercise explicit label-gap and group-exclusion behavior in focused tests; choose feasible rehearsal windows rather than silently relaxing the settings.
3. Measure the pre-hardening baseline using the source snapshot saved before implementation, with the same fixture, feature columns, split, seed and model settings. Then measure the hardened development run. Baseline and after runs use separate artifact/log directories. Record real before → after scores; artifact hardening is expected to preserve behavior, not manufacture an improvement. Investigate material differences.
4. Run dummy → logistic → LightGBM → CatBoost comparisons, with the blend as a development comparison only. Record every actual experiment and keep/drop decision in the rehearsal log. Freeze LightGBM delivery for this scoped test rather than automatically deploying the highest displayed comparison score.
5. Run `uv run python -m src.sanity` in that run context, excluding reserved holdout labels from pre-finalization scans. Inspect the shuffled-target result, single-feature scan and train/test differences. Stop and report suspicious output; do not call a failed check a pass or change the split to make it pass. No deliberate leaky example is included in the production rehearsal path.
6. Select the LightGBM threshold and record value/cost sensitivity from development OOF predictions. Clearly mark monetary values as synthetic scenarios. Include the no-model baseline.
7. Freeze the policy, evaluate the untouched holdout, then fit the final retrospective submission model. Validate the CSV and verify numerical agreement with the reloaded artifact.
8. Generate global drivers and three plain-language explanations, including a threshold-borderline example; verify the reasons against the input rows. Produce the impact report/chart using the same run/model identity.
9. Launch the app using that rehearsal context. Exercise low/high/borderline selections, verify probability/action/reasons/chart agreement, measure initial startup with a target under ten seconds, and check a restart needs no retraining. Use a visible browser where available, alongside automated Streamlit checks; retain evidence and label any manual check that remains pending.
10. Replay the documented one-command workflow in a fresh temporary checkout/source snapshot with no pre-existing generated artifacts. Use the locked environment via `uv`; report cache/network assumptions. The snapshot must contain the proposed changes, including newly added files, without requiring an unrequested commit. Compare deterministic split membership and predictions within tolerance across the two runs, and check that outputs never spill into the main checkout.

If dependencies are cached, also exercise the replay offline. If a genuine fresh dependency download is blocked by network availability, report that limitation separately rather than claim a fresh-install success from a reused environment. Keep any browser/server process scoped to the rehearsal and shut it down afterward.

### 7. Deliver the readiness evidence and handoff — 15 minutes

Write `reports/rehearsal.md` with exact environment/commands, baseline → after scores, per-fold and holdout results, sanity outcomes, experiment locations, artifact identities, CSV validation, app tests/startup timing, clean-replay evidence, raw/config integrity and remaining limitations. Preserve full logs in the rehearsal directory. Every synthetic output must say it is synthetic.

Include a concise runbook: what the human supplies at kickoff, how to start P1, where to place clarifications, what the next approval checkpoint is, and how to select the right run for the demo. Prepare a two-minute rehearsal narrative and identify the inputs needed for a one-minute backup recording. If recording or a live presentation rehearsal needs a human, mark it pending; do not claim it happened.

Finish with a reviewable diff and the real test/rehearsal output. Preserve the existing `START_HERE.md` and `docs/decisions.md` edits already in the working tree. No commit, push, deployment or official submission is included without an explicit request.

## Definition of ready for a 9/10 preparation assessment

- [x] One selected model/run supplies all delivered predictions, explanations, threshold, value report, CSV and demo; stale artifacts are rejected.
- [x] A final holdout is evaluated with frozen settings before the final all-training refit, with evidence that heldout labels were excluded from fitting/tuning.
- [x] Fixtures, experiment logs and artifacts are isolated; raw data, generic defaults and real-case templates remain untouched.
- [x] Tests cover the integration gaps and pass; the automated rehearsal and clean replay complete with recorded commands and actual results.
- [x] The CSV validates and the three automated demo examples work from reloaded artifacts; startup timing and pending human checks are explicit.
- [x] Parallel/checkpoint instructions describe separate checkouts/output directories and explicit staging.
- [x] The runbook documents context handoff and discovery entry points. Actual human pitch/browser/video checks remain open in the report.

This is readiness for the supported tabular starter workflow. An unusual real case can still require adaptation using `docs/guides/problem_types.md`; synthetic rehearsal success is not proof of real-data generalization or business savings.

## Budget, boundaries and stop conditions

| Step | Estimate |
|---|---:|
| 1. Selected artifacts | 40 min |
| 2. Final holdout stage | 45 min |
| 3. Safe rehearsal inputs/outputs | 20 min |
| 4. Operator workflow | 15 min |
| 5. Focused tests | 30 min |
| 6. Rehearsal and clean replay | 45 min |
| 7. Evidence and handoff | 15 min |
| **Total** | **210 min (3 h 30 min)** |

These are planning estimates; training/runtime and repairs depend on this machine. If time is short, retain the LightGBM delivery path and reduce fixture size or optional comparisons first. Do not drop label isolation, raw-data protection or numerical-consistency checks to meet the estimate.

Approval would authorize the scoped implementation, isolated synthetic training/tests and rehearsal described here. It does not authorize changes to `src/eval.py`, new packages, real-data writes, external spending, commits, pushes or deployment. If the existing frozen splitter cannot express a needed case, a dependency is missing, or a sanity result is suspicious, stop and report the concrete issue. Do not silently expand the plan.

**Current action: implementation and automated rehearsal complete. Report evidence and remaining manual checks; no commit, push or deployment has been performed.**
