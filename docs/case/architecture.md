# Architecture & approach — aiming for 95–100

## 0. What wins this case
Scoring is automatic on 5+ hidden videos (one from a new angle) → **each video is ~5 points of detection alone**, plus timecode/type/legal.
Only ~30 teams reach the pitch, so qualification = this automatic score. Priorities, in order:
1. **Never fail a request** (down, timeout, bad JSON = 0 for that video). Robust service first.
2. **Recall = 100%** on known violation patterns (brief: a miss is unacceptable), then cut false alarms.
3. **Timecode at the moment the action starts** (labels mark the onset, not the end) and evidence frame showing it.
4. **Stable, legally grounded types** (15 + 15 points are almost free if done carefully).
5. **Angle robustness**: logic relative to detected objects, never fixed pixels.

## 1a. Two modes, one codebase (decision 19:16)
- **GPU mode (jury deployment):** rented GPU pod runs our service + **Protocol Auditor** (local Qwen2.5-VL via vLLM, `auditor.md`). Rules engine for what/where + auditor for procedure steps and unseen types. Public URL = the pod's HTTPS proxy.
- **CPU mode (fallback + bank story):** same service with `AUDITOR=off` on the laptop + cloudflared. Must always work.
Both modes run the same eval; the jury URL points to whichever scores higher and is stable.

## 1. Compute, environments, time budget
- **Laptop (CPU mode):** Ryzen-class 8 cores, ~40 GB RAM, no GPU. Measured (2 vCPU sandbox): YOLOX-S ONNX ≈ **0.18–0.25 s/frame** on 464×848 → 145 s video at 2 fps ≈ 60–70 s; the laptop is faster. OWLv2 on CPU only on ≤ 20 keyframes.
- **Pod (GPU mode):** RTX 4090 24 GB (or A5000/L4/A40). The service itself still runs detector/pose/OWLv2 on the pod's CPU; **only vLLM uses the GPU**.
- **Two Python environments, on purpose:**
  1. the project env (`uv sync`) — CPU-only libraries (onnxruntime, torch **CPU** build for OWLv2/pose). Identical on laptop and pod. Never add vllm or CUDA torch here (it breaks the laptop install).
  2. a separate vLLM env on the pod only (`uv venv /opt/vllm && uv pip install --python /opt/vllm vllm`), serving the VLM on 127.0.0.1:8001. The service talks to it over local HTTP.
- In GPU mode the auditor answers "cash on desk / safe open / door closed" too, so OWLv2 can be skipped there; in CPU mode OWLv2 keyframes cover cash.
- Time budget per 145 s video: perception ≤ 90 s, auditor ≤ 90 s, encoding/IO ≤ 20 s → ≤ 200 s; guard at 200 s (skip remaining auditor calls).
- **NVIDIA hosted API credits (build.nvidia.com / NIM endpoints) are cloud AI APIs → never called by the service** (§7, FAQ 21, 35). A rented GPU server running our own service and open weights is our own deployment (FAQ 22: teams use their own resources).
- Never send dataset frames to any hosted model for labelling (FAQ 6). Humans annotated (`docs/annotations.md`).

## 2. Concept: Facts → Rules (see `rules_engine.md`)
Perception turns any video into a **fact timeline**; a small **declarative rules engine** turns facts into violations,
each rule linked to a real НБРК article. New violation type = new JSON rule, no retraining → sustainable for the bank.

```
video ──► frames.py   sample 2 fps (sequential grab, exact timestamps)
      ──► detector.py YOLOX-S → people + objects per frame
      ──► tracker.py  IoU/centroid tracks → identities (cashier = track behind the desk at start)
      ──► scene.py    auto-calibration per video: desk, cashier zone, client zone, entry side, safe region,
                       empty-desk / closed-safe reference crops      (relative → angle-robust)
      ──► facts.py    fact timeline every 0.5 s (staff_present, cashier_leaving, safe_open, desk_items, …) + evidence boxes
      ──► engine.py   evaluate rules.json logic → violations (type, onset, evidence offsets)
      ──► auditor.py  GPU mode: Protocol Auditor (local Qwen2.5-VL) on episodes + sweep → checklist answers (auditor.md)
      ──► fuse.py     rules + auditor → final violations (auditor.md §4)
      ──► analyze.py  evidence frames, descriptions, legal refs → API JSON; full fact timeline → UI
```
Everything runs inside the FastAPI worker; models loaded once at startup. Optional credits/cloud: only if they buy a GPU VM
that runs *our* service with open models and organizers confirm it — otherwise ignore; the CPU design must stand alone.

## 3. Models (licences matter — R52)
| Role | Model | Licence | Notes |
|---|---|---|---|
| Object/person detector | **YOLOX-S ONNX** (opencv_zoo `object_detection_yolox_2022nov.onnx`, 36 MB) | Apache-2.0 | COCO classes: person, cell phone, handbag, backpack, suitcase, cup, bottle, laptop, book, chair, dining table, refrigerator (a grey safe may be detected as this). Runs via onnxruntime CPU. **Do not use Ultralytics YOLOv5/8/11 (AGPL-3.0).** |
| **Pose (needed)** | RTMPose ONNX via `rtmlib` or MediaPipe Pose | Apache-2.0 | Wrists for `hand_to_hand`, `drink_at_face_or_desk`, `book_in_hands`; sitting vs standing for `cashier_leaving`. Run only on frames with ≥ 2 people or near events. |
| **Open-vocabulary detector (needed for cash/documents)** | OWLv2 (`google/owlv2-base-patch16-ensemble`) or Grounding DINO tiny | Apache-2.0 | Prompts: «banknotes», «stack of money», «cash», «passport», «ID card», «receipt», «key». COCO has no money class. CPU ~1–3 s/frame → **keyframes only** (event windows, ≤ 20 frames/video), crops of desk/safe. |
| Crop classifier (optional) | OpenCLIP ViT-B/32 (LAION weights) | MIT | Few-shot on desk/safe/door crops: cash vs empty desk, safe open vs closed, door open vs closed, laptop screen on/off. CPU ~0.1 s per crop. Train a logistic regression on crops labelled from these videos. |
| **Protocol Auditor VLM (GPU mode)** | Qwen2.5-VL-7B-Instruct via vLLM (32B on an 80 GB GPU) | Apache-2.0 (verify card) | See `auditor.md`. Not Qwen2.5-VL-3B (research licence) and not 72B (Qwen licence). |

YOLOX ONNX decoding pitfalls (verified): input 640×640, **letterbox with pad value 114, BGR, float32 0–255, no mean/std**; output `(1, 8400, 85)` raw → decode with grids for strides 8/16/32: `xy = (out[:, :2] + grid) * stride`, `wh = exp(out[:, 2:4]) * stride`, `score = obj * cls_prob`; then per-class NMS (cv2.dnn.NMSBoxes), rescale by the letterbox ratio.

## 4. Scene understanding (angle-robust)
Compute once per video from the first ~10 s of detections (and update if the scene changes):
- **Desk**: union of `dining table` boxes (fallback: area around laptop + chair cluster).
- **Cashier zone**: side of the desk where the person sits most of the time (the person present longest near the desk at the start). Client side = opposite side.
- **Entry side**: where people appear/disappear at the frame border (track first/last positions). In the training data the glass door is on the left/back wall.
- **Safe**: tall grey box near the desk (try COCO `refrigerator`; fallback: crop-classifier on candidate regions or a one-time manual click is NOT allowed — must be automatic).
- **People filter**: drop reflections/people behind the glass wall: low confidence, small height (< ~14% of frame height), or outside the room polygon (above the desk line and beyond the glass). Verified: background people behind the glass get detected (conf 0.3–0.5, small boxes) and cause false counts.
- **Tracking**: simple IoU/centroid tracker to keep identities (cashier = track seated in cashier zone).

## 5. Rules per violation type
The 14 annotated types, their cues, tiers and legal mapping are in `violations.md` §2; the JSON logic per type is in
`rules.seed.json`; the facts they need are in `rules_engine.md` §1. Build order by value:
1. **Tier 1** (detector/pose only): `unauthorized_access`, `personal_items`, `food_drink`, `personal_activity`, `workstation_unlocked`.
2. **Cash on desk** (OWLv2 / crop classifier on desk crops): `cash_left_out` — 5 videos, the most frequent type.
3. **Door & safe states** (reference-crop change): `door_left_open` (4 videos), `safe_left_open` (2).
4. **Hand-to-hand** (pose wrists): 3 videos, also strengthens `unauthorized_access`.
5. **Procedure checks** (transaction episodes): `counting_without_client`, `no_count_check`, `no_id_check`, `no_receipt`, `keys_left_in_safe` — ship `enabled: false` until silent on clean client videos.
Generic fallback for unseen types (R36): none — better silent than a false alarm without a legal basis.

**Timecode policy (R33, R37):** labels mark the *start* of the wrongful action (e.g. Х-22: label 14 s = cashier walks to the door; room becomes empty a few seconds later). Report the onset: last frame where the cashier was still seated/at the desk before leaving, or the first frame where the extra person appears. For omissions ("left without locking"), report the moment the duty was due (when the cashier walks away).

**Verdict:** `violation` if ≥ 1 confirmed event. One event per type per contiguous episode; up to 3 per video, sorted by time.

## 6. Evidence frames & descriptions
- Frame 1 = exactly the timecode, with red boxes around the relevant person/object (never faces highlighted as such). Frame 2 = +1.5–2 s showing the consequence (empty desk, open safe). Optional frame 3 = context before.
- JPEG, ≤ 720 px, quality 85, base64 (no `data:` prefix).
- Description template (Russian): «На {t} с {что видно}. Это может указывать на нарушение: {акт}, {пункт} («{цитата}»). Требуется проверка человеком.» — only visible facts (R55, R7).

## 7. Robustness (R38, 10 pts)
- No absolute pixel zones; everything relative to desk/person boxes and frame size.
- Augmentation eval: horizontal flip, 10–15% crop/zoom, ±10° rotation, brightness ±30%, 0.75× resolution. Accuracy must not drop materially.
- Different camera angle = different desk/door positions → scene.py must infer them per video.
- Lighting change exists in training (Х-11, Х-12 turn dark at the end) — don't treat global brightness change as an event.

## 8. Reliability (R26, R27)
- Models loaded once; worker catches every exception → `failed` (never stuck).
- Decode with OpenCV sequential `grab()`/`retrieve()` (fast). If OpenCV can't open a .mov, fall back to `ffmpeg` (bundle imageio-ffmpeg, BSD) to transcode.
- Limit: stop at 300 s of video; time budget guard: if elapsed > 200 s, skip remaining auditor calls and return the rules-based result.
- If vLLM is down or times out (10 s per call), the job continues in CPU mode automatically — a job never fails because of the auditor.
- Keep the laptop awake and plugged in; cloudflared with auto-restart (see deploy.md).

## 9. Product features that make it a real solution (cheap, high pitch value)
- **Fact timeline in the UI**: a strip per fact (cashier at desk, safe open, extra person…) with violation markers — the reviewer sees *why*.
- **Rule cards page** `/rules`: every rule, its logic in plain Russian, the act, point and quote. Compliance can read and audit it.
- **Reviewer feedback**: «Подтвердить / Отклонить» per violation → `data/feedback.jsonl`; a script reports precision per rule (sustainable tuning).
- **Docker image** with weights inside, `docker run -p 8000:8000 -e API_KEY=…` → deployable in the bank's closed network (R51).
- **Privacy option**: blur the head region of person boxes in evidence frames (no face model needed) — off by default so evidence stays verifiable.

## 10. What NOT to do
- No cloud APIs (NVIDIA NIM, OpenAI, Gemini…) in the service → disqualification.
- No training a deep model on 40 videos end-to-end (overfits; angle shift kills it).
- No face recognition, no audio.
- No third verdict like "insufficient data" (penalised).

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
