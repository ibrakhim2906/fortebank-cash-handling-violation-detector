# Datathon '26 kit (team-only — replace with the solution README tomorrow)

A prepared repo for a 24-hour datathon where Codex writes most of the code under our direction.

The latest full case/data/rubric/hardware assessment and proposed recognition pivot are in [`docs/recognition_recovery_plan.md`](docs/recognition_recovery_plan.md). External datasets are explicitly permitted by FAQ7. The proposal prioritizes independently reviewed action intervals and one compact temporal model; new dependencies and external data have not been installed for it.

## Current case: local cashier-video reviewer (P2 baseline)

The actual case is a local video-review service. The first working flow accepts `.mov`, runs a saved local model, and returns the jury JSON with a verdict, candidate time and uploaded-video JPEG evidence. The UI/API are implemented. **Detection quality is not yet useful:** development ROC-AUC is 0.509804 for the linear model versus 0.500000 dummy; the shallow LightGBM comparison is 0.409804. Precise event types and normative rule grounding are not implemented. This is an engineering baseline, not a competition-ready violation detector. See `reports/p2_video.md`.

The requested local pretrained experiment is implemented and documented in [`reports/pretrained_video.md`](reports/pretrained_video.md). OpenCV runs pinned MobileNetV2 weights locally with no additional package. Its best clip-pooling candidate reaches development AUC 0.588235 but catches only 8/17 positives versus the baseline's 10/17. A fixed maximum-window follow-up reaches AUC 0.329412 with 13/15 false alarms. Both fail the predeclared promotion gate, so the default live model remains the original flow-testing baseline. Both experimental artifacts are preserved for review; the report contains exact training, sanity, offline and temporary-serving commands. These are development results, with the eight reserved videos untouched.

Action labeling is implemented in [`reports/action_labels.md`](reports/action_labels.md). The private Windows review page is `http://127.0.0.1:8004/` while its local processes are running; it is separate from the detector on Windows port 8001. The 33 development tasks preserve both disputed anchors. Approved pinned Qwen3-VL/llama.cpp assets are installed. Three autonomous draft/critique rechecks completed, bringing the journal to six proposals, with zero agreed present actions or independently verified labels. The original constant-feature action detector is rejected; its history is in [`reports/qwen_action_detection.md`](reports/qwen_action_detection.md).

The latest fixed spatial comparison improves development AUC **0.509804 → 0.615686**, recall **10/17 → 11/17**, and false alarms **10/15 → 5/15**, while median timestamp error worsens **5 → 9.733333 seconds**. It passes the minimum numerical detection gate, but fresh descriptors differ from its training cache and flip the saved verdict, so **live promotion is withheld**. No action-type or rule accuracy is established. The independent object-track comparison fails promotion. Actual metrics and replay evidence are in [`reports/recognition_followup.md`](reports/recognition_followup.md).

The separate native-video motion gate passes only **1/3** controls; cache-off repeats pass **2/6**, with every answer `left_to_right` and reported `cache_n = 0`. **The native bank-video pilot remains unrun and blocked.** Red/blue input checks pass **4/4** across native video and JPEG in **15.454847 seconds**; fixed RRBB/BBRR first/last-color controls pass **2/2** in **10.037758 seconds**. Exact FFmpeg reproduction correctly decodes all three synthetic sequences. Color delivery and coarse first/last order work; square-motion and cashier-action accuracy remain unproven. These native results do not resolve the separate spatial replay drift. All checks use zero bank or holdout clips, and eight reserved recordings remain untouched. See [`docs/inference_cache_repair_plan.md`](docs/inference_cache_repair_plan.md) and [`docs/local_video_decoder.md`](docs/local_video_decoder.md).

The prepared run in this workspace is `data/processed/video-baseline-comparison/`. Start it with:

```bash
uv sync --locked
uv run python -m scripts.serve_video
```

Linux-local endpoint: `http://127.0.0.1:8000/`. **For the Windows browser on this machine use `http://127.0.0.1:8001/`**, through the verified Windows loopback bridge described below. The launcher creates a private local API key at `data/processed/video-service/api-key.txt`, without logging its value. Open that file locally and paste the key into the page. Upload a `.mov`. Both `POST /api/v1/analyze` (multipart `file`) and `GET /api/v1/jobs/{job_id}` require `X-API-Key`. Uploads are bounded at 64 MiB; one job runs at a time, with two waiting slots and a five-minute total deadline. Uploaded job files are deleted after processing; restart loses in-memory job results. Run one server worker, without reload.

Windows-to-WSL localhost forwarding did not work in this environment. Keep the Linux service above running and start the existing bundled Windows Node runtime in a separate Windows PowerShell terminal:

```powershell
& 'C:\Users\ibrak\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' '\\wsl.localhost\Ubuntu-24.04\home\ibrak\datathon-kit\scripts\windows_loopback_proxy.cjs' --distro Ubuntu-24.04 --root /home/ibrak/datathon-kit --port 8001
```

The bridge binds **only Windows 127.0.0.1** and forwards the UI/API through local WSL stdin/stdout to Linux 127.0.0.1:8000. It does not bind a LAN address, change firewall settings, use an external service or log keys/media. Both the UI and keyed video upload have been verified from Windows. These executable/distro paths describe this machine; other machines should use their own existing Node runtime and WSL distribution name. Both processes must remain running.

To prepare the same flow in a checkout with intentionally provisioned input files, use the original source paths or replace those paths with the actual permitted copies:

```bash
uv run python -m scripts.inspect_video_release --archive /mnt/c/Users/ibrak/Downloads/open.zip --brief /mnt/c/Users/ibrak/Downloads/Кейс_ForteBank.docx
uv run python -m src.train --task video --run-dir data/processed/video-baseline-comparison --models linear,lgbm --note "P2 local baseline"
uv run python -m src.sanity --task video --run-dir data/processed/video-baseline-comparison
uv run pytest
uv run python -m scripts.serve_video
```

Training requires a fresh run directory and preserves the frozen `data/processed/fortebank-20261017/video_protocol.json`. It counts 40 unique content groups once, reserves eight clips, and runs three folds on 32 development clips. P2 never decodes or evaluates the reserved eight. Raw delivery and generated data/model/key files are ignored by Git. No cloud AI or face identification is used. Dependencies are locked; no pretrained external model weights are required for this baseline. CPU inference is measured, but generalization to another camera is unverified.

Optional local verification against a running server:

```bash
uv run python -m scripts.check_video_flow
```

This checks two development examples, renamed-upload invariance, authorization, evidence JPEGs and corrupt-file handling. It prints only numeric/status summaries, never footage or frame payloads. The tabular rehearsal below remains an isolated preparation example; its model scores do not describe this video case.

## What's inside
| Path | Purpose |
|---|---|
| `AGENTS.md` | Standing rules Codex reads on every task |
| `PLAYBOOK.md` | Copy-paste prompts, phase by phase |
| `CHECKLIST.md` | Every requirement from the brief (judges grade against it) |
| `experiments.md` | Log of every result (auto-appended by train.py) |
| `docs/brief.md` | Paste the case text here at 11:00 |
| `docs/context.md` | Case context template (Codex drafts, humans confirm) |
| `docs/jury.md` | Questions for ForteBank mentors + their answers |
| `docs/decisions.md` | Decisions log so Codex stays consistent |
| `docs/guides/` | Phase guides: validation, features, modeling, sanity, explain/impact, problem types, demo/README, pitch |
| `src/` | Working starter pipeline for tabular binary classification / regression |
| `app/app.py` | Streamlit demo: client → prediction → reasons → impact |
| `scripts/make_synthetic.py` | Fake data for tonight's dry run |
| `examples/multitable/` | Tested pattern for applications + history tables, with a leakage demo |
| `tests/` | Point-in-time leakage test to copy for new feature families |

Pitch deck template: the "Datathon ’26 Pitch Template" Slides artifact in your claude.ai artifacts (8 slides with placeholders and timed speaker notes). Share it with teammates from its Share menu.

## Environment: uv
Python and every package are managed by [uv](https://docs.astral.sh/uv/). `pyproject.toml` lists the dependencies, `uv.lock` pins the exact tested versions, `.python-version` pins Python 3.12.
- Install uv once: macOS/Linux/WSL `curl -LsSf https://astral.sh/uv/install.sh | sh` · Windows PowerShell `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Create the environment: `uv sync` (downloads Python 3.12 if needed, installs the locked versions into `.venv`)
- Run anything: `uv run <command>` (no activation needed)
- Add a package (only after the team agrees): `uv add <package>`; never `pip install`

## Rehearse safely

```bash
uv sync --locked
uv run python -m scripts.rehearse --output data/processed/rehearsals/my-first-run
uv run pytest
DATATHON_RUN_DIR=data/processed/rehearsals/my-first-run DATATHON_THREADS=2 uv run streamlit run app/app.py
```

Use a fresh output directory each time. The runner generates seed-42 synthetic inputs outside `data/raw/`, measures development models, runs sanity checks, freezes the LightGBM threshold, evaluates the final holdout, refits for submission, creates explanations and value scenarios, checks three demo examples, and replays in a clean source snapshot. Logs, experiment rows, artifacts and `result.json` stay inside that run. The replay uses locked cached dependencies offline; it is not a test of a new package download.

The fixture is a synthetic binary task, not the unknown competition case. No bank-validated costs, real-data scores, official metric/format assumptions or competition-readiness claims follow from its numbers. Raw data and the real-case templates remain untouched. Read `reports/rehearsal.md` for the actual completed preparation results and remaining human checks.

## Case workflow

Follow P1/P2 in `PLAYBOOK.md` to configure the real case and choose validation. Set agreed economic assumptions before development. Each run may supply a `config.json` containing overrides of the existing case settings and input filenames. Relative input filenames resolve against the run directory. Without a run config, settings come from `src/config.py`.

```bash
uv run python -m src.train --run-dir data/processed/case-run-1 --models dummy,linear,lgbm,cat --note baseline
uv run python -m src.sanity --run-dir data/processed/case-run-1
uv run python -m src.impact --run-dir data/processed/case-run-1 --oof lgbm
uv run python -m src.train --run-dir data/processed/case-run-1 --stage finalize
uv run python -m src.explain --run-dir data/processed/case-run-1
DATATHON_RUN_DIR=data/processed/case-run-1 uv run streamlit run app/app.py
```

These commands require P2 to supply the actual input data/settings first. For regression skip the binary impact step; the policy is frozen without a decision threshold. Development never creates a final model or submission. Finalization evaluates a configured holdout with frozen settings, records its results, then refits retrospectively on all supplied training labels. A live-use model must obey real label availability. Repeating finalization is rejected.

LightGBM is the supported delivered model. CatBoost/blend scores are development comparisons. The app, explanation, threshold and CSV must identify the same selected run/model. Manifests check source/config/input fingerprints and artifact hashes; use a fresh run after changes. Preserve the thread setting when one was configured with `DATATHON_THREADS`.

Parallel chats require separate worktrees/clones and separate output directories. Stage intended files explicitly, review `git diff --cached`, then commit a checkpoint; `git commit -am` misses new files. Rehearsal and case data/models are ignored by Git and must be intentionally provisioned in a fresh checkout. Never delete raw files as rehearsal cleanup.

## Tomorrow
Follow `PLAYBOOK.md`. AI tools and prepared code are confirmed allowed by the human; obtain the actual brief, permitted data and any additional organizer restrictions. Supply context as it arrives using `START_HERE.md`.

## Roles
- **Driver:** runs Codex, reviews diffs, merges to main.
- **Verifier:** owns `src/eval.py` and the sanity checks, checks every number.
- **Analyst / pitch:** talks to ForteBank mentors, owns the insight, business case and deck (starts the deck by hour 6).
- **Demo:** app, README, backup video.
