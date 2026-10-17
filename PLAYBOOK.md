# Playbook — ForteBank video case (phases V0–V10)

Say `run V<n>` (optionally `with: …`) to Codex. Commit after every phase that passes.
Deadline 12:00. The jury's checker uses the API → **a live, correct API outranks any model improvement.**

| Phase | Goal | Target time (matches GAMEPLAN.md) |
|---|---|---|
| V0 | Service skeleton per spec, contract tests, laptop public URL | 20:00 |
| V1 | Perception: frames, detector, tracker, detection cache | 20:45 |
| V2 | Eval script (then frozen) | 21:15 |
| G1 | GPU pod setup (vLLM + service, pod public URL) | 22:00 |
| V3–V4 | Scene calibration, fact timeline, rules engine, rules.json | 01:00 |
| G2 | Protocol Auditor + fusion | 03:00 |
| V5 | Tune rules per type (recall first) | 04:00 |
| G3 + V5c + V5t | Auditor tuning, timecode calibration, type names | 06:00 |
| V7 | Robustness (new angle, own clips) + speed | 07:30 |
| V8 | Product polish: fact timeline UI, `/rules` page (feedback/Docker only if time) | 08:30 |
| V9 | Legal check + §8 documents | 09:30 |
| V10 | Freeze, external API test on both URLs, hand in by 11:00 | 10:30 |

---

> **V0 — Service skeleton**
> Read AGENTS.md, docs/case/requirements.md, docs/case/api_spec.md, docs/case/ui_spec.md. Build `service/app.py` (FastAPI, background worker, disk-persisted jobs, API exactly per api_spec.md, UI endpoints, /health), `service/static/index.html` per ui_spec.md, and `service/pipeline/analyze.py` with a stub `analyze(path)` that samples frames and returns `no_violation`. Add `tests/test_api.py` (contract tests from api_spec.md, pipeline stubbed). Add deps with `uv add` (pre-approved list in AGENTS.md). Run `uv run pytest`, start the server, upload one real video through the UI and through curl, test a wrong key. Report.

**Human:** `~/cloudflared tunnel --url http://localhost:8000` (install: docs/case/deploy.md); open the public URL from a phone on mobile data.

> **V1 — Perception**
> Read docs/case/architecture.md §3–4. Write `scripts/get_models.sh` (YOLOX-S ONNX from opencv_zoo, Apache-2.0), `service/pipeline/frames.py`, `service/pipeline/detector.py` (follow the decoding notes exactly), `service/pipeline/tracker.py`, and `scripts/cache_detections.py` (every 0.5 s, all videos → data/processed/dets/<video>.json with per-frame detections). Add `weights/`, `data/` and `reports/` to .gitignore (never commit the dataset — R57). Run it on all videos; report time per frame and per video, and show detections on 3 sample frames (save annotated JPEGs to reports/). Check that people behind the glass wall are filtered.

> **V2 — Evaluation (freeze after human review)**
> Read docs/case/evaluation.md and docs/case/data.md. Write `scripts/eval_video.py` exactly per the spec (first-3-columns label reading, number-based file mapping, duplicate merge, modes cached/full/api/augment/only, `--videos/--labels` override, type metric from docs/case/types.csv, all metrics). Run it with the stub pipeline (expect recall 0) to prove it works. **Human reviews it, then: `git commit -m "freeze eval"`.**

> **V3 — Scene + facts**
> Read docs/case/rules_engine.md §1 and architecture.md §4. Implement `service/pipeline/scene.py` (auto-calibration per video, relative geometry, reference crops) and `service/pipeline/facts.py` (fact timeline + evidence boxes; thresholds in one CFG). Add `scripts/plot_facts.py` that renders the fact timeline for a video as a PNG strip with label markers. Produce strips for 6 violation and 6 clean videos (incl. the look-alikes Х-3, Х-25, Х-27, Х-29, Х-35) into reports/facts/ and report whether the facts separate them.

> **V4 — Rules engine + annotations**
> Read docs/case/rules_engine.md §2, docs/annotations.md, docs/case/violations.md, docs/case/rules.seed.json. Implement `service/pipeline/engine.py` and `tests/test_rules.py` (synthetic timelines, cases listed in rules_engine.md §4). Create `service/rules.json` from the seed, refining each `logic` from the annotations (what separates violation from look-alike). Wire facts → engine → analyze.py (evidence frames, descriptions per architecture.md §6). Run eval (cached mode) and report per type.

> **V5 — Tune one type at a time** (`run V5 with: <type key>`)
> For that type only: inspect every missed violation and every false alarm (print the facts around the event), adjust fact thresholds in CFG or the rule logic in rules.json, re-run eval. Keep the change only if recall did not drop. No per-video special cases, no fixed pixel zones. Log in experiments.md.

> **V5c — Timecode calibration** (after V5 for the main types)
> For each type, compute the median offset between the rule's trigger time and the labelled second over the training events (eval output). Add a per-type `timecode_offset_sec` to rules.json (applied in the engine), only where the median offset is consistent (|median| > 0.5 s and spread small). Re-run eval and report hits within ±1 s / ±2 s before → after.

> **V5t — Type names and ordering** (see docs/jury.md)
> Make the API `type` = `name + " / " + name_en` from rules.json, order violations most-confident-first (max 3), keep the UI showing both. Never change type keys. Re-run eval (type metric) and `uv run pytest`.

> **G1 — GPU pod setup** (human created the pod and pasted the SSH command)
> Read docs/case/deploy.md "GPU mode" and docs/case/auditor.md §1. (If your sandbox blocks SSH/network, write the script and print the exact commands for the human to run instead.) Write `scripts/pod_setup.sh` (idempotent): uv, ffmpeg, repo clone/pull, `uv sync`, get_models, VLM weights download, vLLM on 127.0.0.1:8001, service on 0.0.0.0:8000 with `AUDITOR=on`, both in tmux restart loops, then `HF_HUB_OFFLINE=1`. Run it on the pod over SSH, copy data/raw/open for eval, and report: GPU model, vLLM health, `/health` via the public proxy URL, and one video processed end-to-end with its time.

> **G2 — Protocol Auditor**
> Read docs/case/auditor.md fully. Implement `service/pipeline/auditor.py` (episode frame selection, checklist prompts, JSON validation, retries, time guard, auditor.jsonl log) and `service/pipeline/fuse.py` (decision table §4). Add `auditor_item` mappings in rules.json for Tier-3 types. Run eval in full mode on the pod with AUDITOR=on and off; report per type and per video (esp. X-1, X-2, Х-22, Х-32, Х-33, Х-34, Х-47) and time per video.

> **G3 — Auditor tuning**
> For each checklist item: keep it only if it adds true detections without false alarms on clean videos (Х-3, 5, 25, 27, 29, 35 …). Tune frame selection/prompt wording, not per-video hacks. Re-run eval; log in experiments.md. Confirm worst case < 240 s with AUDITOR=on.

> **V7 — Robustness + speed** (`run V7 with: own clips` once `data/raw/own/` exists: evaluate on those clips with `data/raw/own/labels.csv` as a separate, never-tuned test set and report it separately)
> Add augmentations to the full-mode eval (flip, 10–15 % crop/zoom, ±10° rotation, darker, 0.75× resolution). Report accuracy per augmentation and the leave-one-setup-out check (evaluation.md). Fix anything that depends on absolute positions. Then run `--full` on Х-14 twice in a row and report times; add a time-budget guard per architecture.md §8.

> **V8 — Product polish**
> Read architecture.md §9. Add: the fact timeline strip in the UI result (from the UI endpoint), a `/rules` page rendering rules.json as rule cards (name, logic in plain Russian, act, point, quote), «Подтвердить / Отклонить» feedback per violation saved to data/feedback.jsonl, and a `Dockerfile` per deploy.md. Keep the API schema untouched; `uv run pytest` must stay green.

> **V9 — Legal check + documents**
> Read docs/case/legal.md and docs/case/submission_template.md. Verify every rules.json quote against legal.md (incl. the п. 16 НБРК №14 edition note — cite the edition in force), make every description cite act + point, and write `docs/submission.md` with real eval numbers and known failure cases. List models and licences.

> **V10 — Freeze and hand in**
> From a fresh clone: get_models, `uv sync`, start the service, run `uv run python scripts/eval_video.py --api <PUBLIC_URL> --key <KEY>`. Report failures, timeouts, max time. Then stop changing code.

> **MISSION — long-running quality goal** (after G1; replaces manual V5/G2/G3 loops)
> Read docs/case/mission.md and execute it fully. Keep iterating until its done-conditions are met or 06:30. Don't stop to ask; log every iteration in reports/progress.md.

> **VS — Status** (any time)
> No code changes: current eval numbers, CHECKLIST.md status, open problems, next 3 most valuable tasks.

---
## Human checkpoints
1. Annotations (`docs/annotations.md`) are complete before V4.
2. eval_video.py reviewed and frozen at V2.
3. Fact strips (V3) visibly separate violations from look-alikes — if not, fix facts before rules.
4. Public URL works from outside; laptop stays awake until the jury finishes.
5. Every legal quote checked on adilet.zan.kz.
6. Worst case < 5 min, twice in a row.
