# API spec (R20–R28) — the jury's checker uses ONLY this

Base: `https://<public-host>` (cloudflared tunnel → `http://localhost:8000`). All `/api/v1/*` require `X-API-Key`.

## POST /api/v1/analyze
- Request: `multipart/form-data`, field **`file`** (.mov; also accept .mp4). Size up to 200 MB (expected ~30 MB).
- Response 200: `{"job_id": "3f2a9c1e5b7d"}` — job id: lowercase hex, 12 chars.
- Errors: 401 missing/wrong key · 413 too large · 415 wrong extension · 422 no `file` field.
- Must return **immediately** (save file, enqueue, respond). Never analyse inside the request.

## GET /api/v1/jobs/{job_id}
Response 200, while processing:
```json
{"job_id": "3f2a9c1e5b7d", "status": "processing", "verdict": null, "violations": []}
```
When done — **exactly these keys, nothing else** (R22, R28):
```json
{
  "job_id": "3f2a9c1e5b7d",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Оставление рабочего места без обеспечения сохранности наличных денег",
      "timecode_sec": 14.0,
      "description": "На 14.0 с кассир встаёт и покидает помещение; на столе остаются наличные/документы, сейф открыт. Это может указывать на нарушение п. 22 Правил № 231 НБРК («Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента»). Требуется проверка человеком.",
      "evidence_frames": [
        {"timestamp_sec": 14.0, "image_base64": "/9j/4AAQ..."},
        {"timestamp_sec": 16.0, "image_base64": "/9j/4AAQ..."}
      ]
    }
  ]
}
```
- `verdict`: `"violation"` or `"no_violation"` (never null when done). `violations: []` when `no_violation`.
- `type`: `"<name> / <name_en>"` from rules.json (bilingual — see docs/jury.md). Violations ordered **most confident first**, max 3.
- `timecode_sec`: float seconds from video start (2 decimals).
- `evidence_frames`: 1–3 items, **JPEG** base64 without `data:` prefix, ≤ 720 px long side (keeps JSON small), first frame = the timecode frame. Highlight boxes allowed (people/objects), never faces.
- `description`: Russian, 1–3 sentences: what is visible → "может указывать на" → act + point (+ short quote). The rule reference lives inside the description because the schema has no separate field.
- Failed: `{"job_id": "...", "status": "failed", "verdict": null, "violations": []}` (+ optional `"error"` short text). A job must never stay `processing` forever: wrap the pipeline in try/except and set `failed`.
- Errors: 401 wrong key · 404 unknown job.

## Behaviour
- One background worker processing jobs FIFO; jobs persisted to `data/jobs/<job_id>/result.json` (atomic write: tmp → rename) so a restart doesn't lose results. On startup, re-queue jobs still `processing`.
- Worst-case time (145 s video) < 240 s, twice in a row (R26).
- `GET /health` → `{"ok": true, "queue": n}` (no key) for monitoring.
- UI uses separate endpoints without key: `POST /ui/analyze`, `GET /ui/jobs/{id}` (may include extra fields: `rule` list, `processing_sec`).
- API key: from env `API_KEY` (long random string). Never hard-code it in the repo.

## Contract tests (`tests/test_api.py`, stub the pipeline)
1. Missing/wrong key → 401 on both endpoints.
2. POST returns only `{"job_id"}`; GET when done returns exactly the 4 top-level keys; violation has exactly the 4 keys; frames 1–3 with exactly `timestamp_sec`, `image_base64`.
3. Unknown job → 404. Pipeline exception → `status: failed`.

## curl (from the brief)
```
curl -X POST https://<host>/api/v1/analyze -H "X-API-Key: <key>" -F "file=@video_017.mov"
curl https://<host>/api/v1/jobs/<job_id> -H "X-API-Key: <key>"
```

## MISSION execution resolutions (human: resolve issues and run)

The supplied brief, original raw data and current verified legal research prevail over inconsistent seed/proposal wording.
- Preserve read-only root AGENTS.md; incoming AGENTS is a supplied reference in the bootstrap archive. This mission is directly authorized by the human. Use uv, config.SEED, preserve raw data, frozen src/eval.py, existing unrelated changes and source provenance.
- All40unique recordings were already reviewed. Freeze10as a previously-seen prospective regression reserve, NOT an untouched independent holdout. Report exposure and missing genuinely new test footage. No split undoes inspection history.
- Preserve full-content SHA grouping/aliases, both57/68annotations and the Х-13disagreement. The mission evaluator uses one-to-one event matching and reports disputed annotations explicitly; original src/video_eval.py remains historical.
- Camera-group CV uses five folds only if five real groups exist; otherwise leave-one-setup-out with actual group counts. Augmentations stay with their original group. No fake camera groups.
- Use existing verified raw/derived inventory paths. No rewriting data/raw/ to match archive examples.
- No arbitrary three-event cap. Unknown never satisfies absence; lit screen is not proof of unlocked session, wrist proximity is not transfer, a collection bag is not proven cash/personal property, and no counter use is not proof of no manual count.
- Seed rules remain inactive until predicate/source/applicability tests. Team video notes do not verify legal claims. Preserve docs/action_rule_candidates.md and current docs/case/legal.md over seed interpretations. No invented internal ForteBank policy.
- Rubric number is the requested engineering proxy, not a jury score. Missing evidence earns no unverified points. Report eligibility/exposure and incomplete gates.
- Rollback only mission-owned edits from snapshots/patches; never reset prior uncommitted work. No media, caches, weights, keys or per-job evidence in Git. Code-only commits/tags are explicitly requested; push awaits valid remote/access.
- Existing A40 Pod p7ayn21cpy6voe only,$0.59/h,$15all-in,no auto top-up. Deadline06:30means18October2026Asia/Qyzylorda (01:30UTC). Preserve funded jury-serving time.
