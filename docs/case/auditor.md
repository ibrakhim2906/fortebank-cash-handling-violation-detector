# Protocol Auditor — local vision-language model stage (GPU mode)

**Role:** the rules engine sees *what is where* (people, objects, door, safe, cash, hands). The auditor sees *what happened*:
was the ID shown, was the tray used, was the cash counted on the counter, was a receipt given, are keys in the lock, is this
person an outsider, and anything unusual not covered by rules (unseen violation types, FAQ 9).
It runs **locally inside our service** on our own GPU server — no external API (R50).

## 1. Model
| Option | Licence | GPU | Notes |
|---|---|---|---|
| **Qwen3-VL-8B-Instruct** (default, released Sep 2025) | Apache-2.0 (verify card) | 24 GB (RTX 4090 / A5000 / L4) | Newer and stronger than Qwen2.5-VL; multi-image + video input; vLLM-supported. bf16 ~17–18 GB. |
| Qwen2.5-VL-7B-Instruct (baseline) | Apache-2.0 | 24 GB | Fallback if Qwen3-VL has a vLLM issue. |
| Qwen2.5-VL-32B-Instruct | Apache-2.0 (verify card) | 80 GB (A100/H100) | Stronger reasoning; ~3× slower. Use if the 7B misses procedure steps. |
| Newer Apache-2.0 Qwen-VL release | check card | — | Allowed only if the licence permits commercial use. **72B uses the Qwen licence — avoid.** |
Serve with **vLLM** (Apache-2.0) **in its own env on the pod** (never a project dependency):
```
uv venv /opt/vllm && uv pip install --python /opt/vllm vllm huggingface_hub
/opt/vllm/bin/huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir /workspace/models/qwen3-vl-8b
HF_HUB_OFFLINE=1 /opt/vllm/bin/vllm serve /workspace/models/qwen3-vl-8b --served-model-name qwen-vl \
  --host 127.0.0.1 --port 8001 --max-model-len 32768 --limit-mm-per-prompt '{"image": 16}' --gpu-memory-utilization 0.9
```
(Flag syntax differs between vLLM versions — check `vllm serve --help`.) The service calls the OpenAI-compatible endpoint
`http://127.0.0.1:8001/v1/chat/completions` with base64 images — same machine, so still local (R50). Serving from a local
directory with `HF_HUB_OFFLINE=1` proves no internet is needed after download (R51).

## 2. What the auditor looks at (keep it fast: ≤ 60 s per 145 s video)
1. **Transaction episodes** (from facts): client enters client zone → leaves. Sample 12–16 frames across the episode (denser around `hand_to_hand`, `cash_received`, `cash_given`).
2. **Leaving episodes:** 3 s before → 5 s after `cashier_leaving`: 8 frames incl. desk, safe, door, laptop crops.
3. **Entry episodes:** each `person_entered`: 6 frames.
4. **Whole-video sweep:** 16 frames uniformly (catches unseen types).
Frames: 448 px long side; crops of desk/safe/door/laptop added at 336 px where the scene module found them. Each frame is labelled with its timestamp in the prompt.

## 3. Prompt (Russian or English; JSON only)
System: «Ты — аудитор кассовой дисциплины банка. Отвечай только о том, что видно на кадрах. Не идентифицируй людей и не описывай лица.»
User: the frames with `t=…s` labels + the **protocol checklist** for that episode type, e.g. transaction episode:
```
Checklist (answer each: "yes" | "no" | "not_visible", + the frame time where it is visible):
id_document_shown   — client shows/passes an ID document before cash is accepted
tray_used           — cash/documents pass through the transfer tray/device, not hand to hand
counter_used        — banknotes are counted on the banknote counter in the client's presence
receipt_given       — a receipt/document is handed to the client after the operation
client_stays_until_done — client stays at the desk until cash is counted
outsider_present    — a third person is inside the booth / on the cashier's side
other_issue         — anything else that breaks cash discipline (describe in ≤ 15 words) or "none"
Return JSON: {"items": {"<id>": {"answer": "...", "t": <sec or null>, "evidence": "<≤ 15 words, visible facts only>"}}}
```
Leaving episode checklist: cash_put_away, safe_closed, keys_removed, laptop_locked_or_closed, door_closed_behind.
Entry episode: role (client to client side / staff / outsider to cashier side or equipment).
Sweep: open question «Есть ли на кадрах нарушения кассовой дисциплины? Выбери только из списка rule_id: [...] или "none"» — **the model chooses rule_ids from our catalogue, it never writes legal references.**

## 4. Fusion with the rules engine (decision logic)
Each checklist item maps to a type in `rules.json` (`auditor_item` field). For every candidate:
| Rules engine | Auditor | Decision |
|---|---|---|
| fires (Tier 1/2) | agrees or not_visible | **violation** (rules timecode) |
| fires | clearly "no violation" twice (2 samplings) | violation if Tier 1 with high conf, else drop |
| silent | "no" on a required step (Tier 3) with a frame time | **violation** at the auditor's frame time, evidence = that frame |
| silent | other_issue with a catalogue rule_id | violation only if confirmed by a 2nd, focused call on those frames |
Recall first: a single strong signal is enough; weak signals need two agreeing sources.
Timecode: rules time when available (calibrated, V5c); else the auditor's frame time snapped to the nearest sampled frame.
Descriptions: auditor evidence text (visible facts) + rule quote from rules.json. Model text is sanitised: max 2 sentences, no names, no identity, no legal claims.

## 5. Robustness & safety
- Deterministic decoding (temperature 0), JSON schema validation, one retry on invalid JSON, then ignore that call.
- 10 s timeout per call; if vLLM is unreachable, log it and continue rules-only (never fail the job).
- Descriptions never state amounts, names or identities (X-1's note mentions sums — those are not visible facts; describe only "кассир передаёт наличные, документ клиенту не передаётся").
- Time guard: if elapsed > 200 s, skip remaining auditor calls and return rules-only result.
- `AUDITOR=off` env → pure CPU mode (laptop fallback, bank without GPU).
- Log every auditor answer to the job folder (`auditor.jsonl`) → shown in the UI fact timeline as "проверки протокола" (great for the pitch).

## 6. Evaluation
eval_video.py full/API mode with `AUDITOR=on` vs `off`: report verdict accuracy, recall, false alarms, type accuracy, timecode hits, time per video. Keep the auditor only for checklist items that improve the score without new false alarms on clean videos.
