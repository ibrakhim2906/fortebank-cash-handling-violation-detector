# Checklist — judges grade against this

**Active authority, 18 October:** the human approved autonomous MISSION execution and the existing A40 deployment under the $15 cap. `docs/plan.md` and the latest `reports/progress.md` entries are the current work queue. Earlier CPU proposals and approval-pending statements are historical. At 05:49, the selective storage contender improves 30-video DEV accuracy from20/30 to23/30, with12/16 positives detected and false alarms reduced6→3. Final public qualification is pending; recognition targets remain unmet. No requirement is complete merely because software tests pass.

**Evaluation disclosure, 17 October:** the human explicitly chooses semantic review of all 40 unique supplied recordings (42 files), including the eight formerly reserved examples. The original protocol remains unchanged as provenance; those eight no longer provide an independent holdout. Visual-review proposals are separate from model evaluation and verified action labels. See `docs/case_review_scope.md` and `reports/case_video_review.md`.

> Codex fills the "From the brief" section in hour 1, quoting the brief's wording.
> In the pitch, every box must be visibly ticked on a slide or in the demo.

## From the brief
- [ ] Working frontend/backend: «пользователь загружает видео и получает результат без ручного запуска кода» (§5).
- [ ] Upload video, show processing status, then one of «Нарушение не обнаружено», «Обнаружены признаки возможных нарушений» (§4).
- [ ] Detect visually observable actions; support multiple possible events in one video (§4–5).
- [ ] For every event: «его тип, таймкод и описание» (§4).
- [ ] Show «один или несколько кадров-доказательств» and a concise observable explanation without unsupported assumptions (§4–5).
- [ ] Give «правило и нормативный источник: название документа, пункт/раздел» (§4); document FAQ31's allowance for less precise references.
- [ ] Independently research public RK sources, including NBK/Adilet, and demonstrate source → rule → video evidence (§6, FAQ30–32).
- [ ] Do not claim invisible facts, including bookkeeping or reconciliation correctness; output supports human review (§7).
- [x] «все модели выполняются внутри развёрнутого командой сервиса»; no cloud AI processing or transfer of video, frames or descriptions (§7). P2 code and local API check: `reports/p2_video.md`.
- [x] «после загрузки весов сервис не должен требовать доступа в интернет для обработки видео» (§7). Saved local baseline passed network-denied inference; `<video-run>/offline_check.json`.
- [ ] Supply public UI address, API address and jury API key (§8, FAQ23).
- [ ] Supply run/use instructions, architecture/approach/models, normative-source mapping, recognized event types, limitations and known failure cases (§8).
- [x] Keyed `POST /api/v1/analyze`: `.mov`, multipart `file`, returns `job_id` (§10). Local HTTP check and API tests.
- [x] Keyed `GET /api/v1/jobs/{job_id}`: `processing`, `done`, `failed` and final result (§10). Local HTTP check and API tests.
- [ ] Reject missing/incorrect `X-API-Key`; generate/store actual key securely and hand it to jury (§10). Local rejection/key storage verified; jury handoff still pending.
- [x] JSON verdict is `no_violation` or `violation`; `violations` is empty for the negative verdict (§10). Schema/API tests and real local uploads.
- [x] Event JSON includes `type`, `timecode_sec`, `description`, `evidence_frames`; each frame has `timestamp_sec` and JPEG `image_base64`; 1–3 frames per event (§10). Format verified; semantic type/rule quality remains open above.
- [ ] Finish each evaluation video within «5 минут»; unreachable/timeouts count unprocessed (§10, FAQ14).
- [ ] Never train on hidden test videos, publish them or transfer them to third parties (§10).
- [ ] Quality of detection: «25» points; correctness of video verdict (rubric, FAQ15).
- [ ] Quality of localization: «15» points; precise timestamp and relevant evidence frame (rubric, FAQ17).
- [ ] Quality of classification: «15» points; correct event type (rubric).
- [ ] Rule grounding: «15» points; accurate source/interpretation, no fabricated citations (rubric).
- [ ] Robustness: «10» points; unseen videos and different camera angles (rubric, FAQ4).
- [ ] Product readiness: «10» points; usable UI, reliable processing and clear result (rubric).
- [ ] Technical solution/locality: «10» points; practical architecture and extensibility (rubric).
- [ ] Evaluate and document behavior on unseen event types; there is no closed taxonomy (FAQ8–9). Do not promise universal detection.
- [ ] Document timing convention for omitted actions; team chooses it (FAQ10).
- [ ] Minimize false positives and explicitly report misses; FAQ16: «пропуск нарушения недопустим» is a desired outcome, not an evidenced guarantee.
- [ ] Evaluation returns the common JSON schema and is automatic; at least five hidden clips expected (FAQ12,18).
- [ ] Finalists prepare «5 минут на выступление и 2 минуты на вопросы»; slides optional (FAQ19).
- [ ] Use team compute; organizer provides no GPU or cloud credits (FAQ22).
- [ ] Check licenses: prepared code and pretrained models permitted when open source with commercial-use rights (FAQ24); clarify “готовых решений” prohibition (FAQ35).
  Component evidence: the tested OpenCV Zoo MobileNetV2 weights have their Apache-2.0 license preserved under `licenses/opencv-mobilenet/`, with pinned weight/license checksums and source URLs. See `reports/pretrained_video.md`; this does not complete the full deliverable/license audit.
  Approved local Qwen3-VL GGUF weights/projector (Apache-2.0) and standalone llama.cpp runtime (MIT) are also pinned, downloaded and license-checked; see `docs/local_action_model.md`. Their detector trial failed and is not promoted.
- [ ] Accept `.mov` around 29MB; inspect actual supplied sizes rather than impose an unverified hard cap (FAQ25).
- [x] No audio processing: «анализ строго визуальный» (FAQ28). OpenCV video-only feature pipeline.
- [x] No face recognition/identification: «это запрещено» (FAQ29). Numeric scene/motion features only; no identity model.
- [ ] Data use only «в рамках хакатона» (FAQ6); clarify retention/deletion after the event.
- [ ] Meet «В 12:00» deadline; confirm date and submission destination (FAQ33).
- [ ] Disclose ForteBank's stated commercial-use rights and track component licenses (FAQ34).
- [ ] No extra login/history/export/batch interface is required (FAQ27); API key remains mandatory.

Case note: the always-required items below are team standards. The tabular `src.sanity` command is not evidence of video leakage safety; propose video-specific checks without modifying frozen `src/eval.py`. Credit-default economics must not be reused. No requirement is marked complete merely because it has been extracted.

## Always required (our standard)
- [ ] Validation that mimics real use (time-based or grouped by client) — documented
- [ ] Leakage checks passed (`uv run python -m src.sanity`)
- [ ] Explanations: global drivers + reasons for individual predictions
- [ ] Business impact in ₸ with stated assumptions
- [ ] Working demo (one clear flow)
- [ ] README with exact run commands
- [ ] Reproducible from a fresh clone
- [ ] Backup demo video
- [ ] Pitch fits the time limit (rehearsed with a timer)
