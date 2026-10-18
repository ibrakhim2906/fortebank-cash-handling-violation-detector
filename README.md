# ForteBank cash-handling violation detector

A local video-review service for cash-desk CCTV. Upload a recording and get back:
- a verdict (violation or not);
- each event's type, timecode and evidence frames;
- a description that cites the specific clause of НБРК №231 the event relates to.

All inference runs on our own GPU with an open-weight model. No cloud AI APIs are used, there is no face identification, and the service works with network access blocked.

**Public API:** `https://p7ayn21cpy6voe-8000.proxy.runpod.net` (the API key is delivered privately).

## How it works

```
.mov upload → frame sampling (16 overview images, source timestamps kept)
            → Qwen3-VL-32B-Instruct-AWQ sweep: typed candidate events with onset/evidence frames
            → dense focused confirmation of each proposal (rejects unsupported concerns)
            → storage veto: counter-examples for "cash/safe left out" (cuts false alarms)
            → rescue pass: phase-shifted second sweep re-checks missed/uncertain events
            → result JSON: verdict, typed events, timecodes, evidence JPEGs, cited rule
```

- **Model:** `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` (Apache-2.0), served by vLLM 0.11.0 on one A40 at `127.0.0.1:8002`. The model port is never exposed.
- **Service:** FastAPI (`service/app.py`). Jobs are asynchronous, so the request returns a job ID immediately. One worker handles one job at a time, with a 185 s analysis budget inside the 5-minute deadline.
- **Pipeline:** `service/pipeline/analyze.py` orchestrates the steps above. `service/pipeline/auditor.py` holds the model stages (sweep, confirmation, rescue).
- **Rules:** `service/rules.json` defines 14 violation types. Each type has a Russian and an English name and, where one has been verified, an exact quote from the official act.

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/v1/analyze` | multipart `file` (.mov), header `X-API-Key` → `{job_id}` |
| `GET` | `/api/v1/jobs/{job_id}` | header `X-API-Key` → status, then result |
| `GET` | `/` | web UI: upload, timeline, evidence frames |
| `GET` | `/health` | liveness and queue length |

The result has this shape:

```json
{"verdict": "violation",
 "violations": [{"type": "Наличные деньги оставлены вне сейфа (хранилища) / Storage of previously accepted cash outside designated lockable storage units",
                 "timecode_sec": 9.0,
                 "description": "На 9.00 с видно: … Норма: НБРК №231, п.20, ч.2: «…» (https://adilet.zan.kz/rus/docs/V1900019680). Применимость к операции требует проверки человеком.",
                 "evidence_frames": [{"timestamp_sec": 9.0, "image_base64": "…"}]}]}
```

## Measured results

All numbers come from real runs with the frozen evaluator (`scripts/eval_video.py`). The team reviewed all 40 supplied videos during development, so none of these are independent test scores.

| Recipe | Data | Correct verdicts | TP / FN / FP | Events within ±2 s |
|---|---|---:|---|---:|
| Selected (storage veto) | DEV30 | 23/30 | 12 / 4 / 3 | 5/16 |
| **+ rescue recall pass** | DEV30 | **24/30** | 13 / 3 / 3 | 5/16 |
| Selected, public API | all 40 | 27/40 | 13 / 8 / 5 | 6/22 |
| **+ rescue recall pass** | all 40 | **28/40** | 14 / 7 / 5 | 6/22 |

- **How rows were measured:** the public-API row was measured through the live service. The rescue rows were run on the same model in an isolated copy of the service; with recall off, that copy reproduced all 30 DEV results exactly.
- **Latency:** the slowest of the 40 public-API uploads took 73.0 s, and two repeats took 64.9 s and 65.9 s, all within the 200 s limit.
- **Event types:** every event matched in time got the correct type.
- **Log:** every experiment, including the ones we dropped, is in `experiments.md`.

## Rule grounding

Seven violation types cite exact wording from the current official revision of **НБРК №231 «Правила ведения кассовых операций»**. The text was retrieved from the Adilet API and its SHA-256 is recorded in `service/rules.json`.

| Type | Clause |
|---|---|
| Cash left outside storage, safe left open | п.20, ч.2: accepted cash is kept in lockable storage |
| Counting without the client | п.20, ч.1: cash is accepted and counted in the client's presence |
| No note-by-note count | п.19, пп.2 |
| No ID check | п.19, пп.1 |
| No receipt issued | п.19, пп.6 |
| Personal items in the cash unit | п.12 |

Types without a directly applicable clause (door, keys, food, reading) make no legal claim. Every description says that applicability needs human review. The sources and their limits are documented in `docs/action_rule_candidates.md`.

## Local inference and licences

- **Network-denied run:** passed on the selected GPU recipe (`data/processed/mission-selected-gpu-offline-proof.json`).
- **Licence check:** `uv run python -m scripts.verify_license_policy` checks all 238 installed distributions, plus CPython, the FFmpeg decoder and the model weights, against the organisers' rule, with 0 failures. Under that rule, code and models must be open source with commercial use permitted, and the model weights must be Apache/MIT/BSD. The full table is in `docs/submission.md`.

## Run it

```bash
uv sync --locked
uv run pytest                                   # 1234 passed, 2 skipped
# GPU pod: model server + API (see docs/runpod_setup.md)
bash scripts/pod_setup.sh prepare               # pinned vLLM env + model download
set -a; . deploy/mission-best.env; set +a       # selected recipe; add AUDITOR_RECALL=rescue
bash scripts/pod_setup.sh supervise model &
bash scripts/pod_setup.sh supervise service &
```

To run the DEV30 evaluation and compare runs, use `tools/claude-recall/run_dev30.py`, then `tools/claude-recall/fetch_score.sh <run>`, then `scripts/compare_runs.py`.

## Repository

| Path | Contents |
|---|---|
| `service/` | API, web UI and the recognition pipeline |
| `src/` | video I/O, result schema, evaluation helpers |
| `scripts/` | evaluation, qualification, pod setup and licence verification |
| `deploy/` | pinned model manifests and the selected recipe (`mission-best.env`) |
| `tests/` | 1234 tests |
| `docs/` | case brief, decisions log, submission and licence audit, legal sources |
| `experiments.md` | every experiment with its measured result |

## Limitations

- **Timing:** events where a state persists (cash left out, door open) are often flagged when the state first becomes visible, before the moment the annotators marked, so most of them land outside ±2 s.
- **Omissions:** steps that should have happened but didn't (no receipt, no ID check, counting without the client) are rarely detected from sampled frames. The pipeline deliberately never infers an omission from frames that weren't seen.
- **Validation:** the team reviewed all 40 supplied videos, and nothing here is validated on an unseen camera layout.
