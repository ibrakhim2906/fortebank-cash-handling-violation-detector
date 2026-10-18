# Local action labels — autonomous recheck completed, truth still unverified

The human requested autonomous action-specific labels and continued detection improvements, then explicitly approved the additional local runtime/model download. The original three-task pilot saved **three local-model proposals with zero failures**. After the human clarified that actual local video review was wanted, a separate chronological draft-and-critique pass rechecked **three tasks successfully**, appending **three further proposed revisions**. It found **zero present-action agreements and zero verified labels**; the fixed guard stopped bulk expansion because all actions remained unknown. The model has processed the actual local media, but that is not independent human verification. The original label-independent detector failed promotion: all 32 recordings produced the same clip feature vector and the selected detector flagged every video. Full results: `reports/qwen_action_detection.md`.

A separate fixed spatial-observation recipe completes development scoring at selected AUC **0.615686**, recall **11/17** and **5/15** false alarms; timing worsens and fresh descriptors flip one development example's verdict relative to its training cache despite identical source frames/views. Its live promotion is withheld. The independent fixed YOLOX object/anonymous-track comparison fails the unchanged detection gate. Both keep the original folds and threshold; neither supplies verified action labels. Full evidence is in `reports/recognition_followup.md`. A synthetic-only native-video motion gate and bounded three-task pilot are prepared in `docs/recognition_recovery_plan.md` and `docs/native_action_pilot_plan.md`.

## What is ready

- Immutable development-only queue: **32 groups, 33 tasks, 18 timestamp anchors**, including both disputed anchors, plus 15 negative-video review tasks. Eight reserved videos remain excluded and undecoded.
- Eight draft observable primitives: object handoff, object relocation, repeated separation/restacking, container opening/closing, person region transition, hand/object contact, handheld-device interaction and an explicit unknown action. These are generic vocabulary choices, not assignments to any clip or bank-rule violations.
- Official binary labels/timestamps preserved as weak annotation anchors. Negative violation labels do not mean routine actions are absent. Unreviewed coverage remains unknown.
- Private local review UI: **Windows `http://127.0.0.1:8004/`**, through a loopback-only bridge to Linux port 8002. The existing detector remains on Windows port 8001 / Linux port 8000.
- Keyed development-only media playback, editable intervals, present/absent/unknown actions, visibility, reviewer identity, interval/full-clip coverage and verified/ambiguous status. The form hides official targets and does not prefill model guesses as truth.
- Hash-chained append-only journal at `data/processed/action-labels/reviews.jsonl`. Model proposals never replace human reviews; a later ambiguous human revision revokes earlier verified training labels. Export includes only known, independently reviewed observations for the actual reviewed interval.
- A local model client that verifies provisioned weights/runtime before opening media, samples three bounded in-memory JPEGs, sends them only to literal `127.0.0.1:8003`, disables environment proxies/redirects, validates structured output and appends only proposals. It does not send official targets or annotation metadata to the model. Sparse samples cannot establish absence, so model absence guesses become unknown. No clip-specific output is printed.

The case forbids transmitting videos, frames **or their descriptions** to cloud AI. Therefore the cloud coding assistant cannot inspect these clips to manufacture manual labels. Media observations and review notes stay in the local application/files. Only aggregate status is reported through tools.

## What is needed

1. **Dependency approval received:** the standalone CPU llama.cpp runtime and official Qwen3-VL-2B GGUF weights/projector, approximately **1.57 GB**, are installed and running on fixed loopback port 8003. Exact URLs, hashes, licenses and launcher configuration are in `docs/local_action_model.md`. Sixty runtime files were materialized and checked against the pinned archive. No Python package was added. Provisioning still defaults to inspection only.
2. Independent action supervision remains missing. A short local human check at `http://127.0.0.1:8004/` can confirm directly observable actions or mark uncertainty; unknown fields need no invented answer. The human previously agreed to review the pilot. Autonomous local rechecking has now proceeded without waiting for that review, but same-model agreement cannot independently verify action labels. This input is needed to measure action-specific accuracy or train a supervised action head, rather than for continued label-independent binary experiments. Do not send screenshots or clip descriptions through chat. No further technical setup is required for the existing local page.

The chosen [publisher-official Qwen3-VL GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) declares Apache-2.0. The [pinned llama.cpp runtime](https://github.com/ggml-org/llama.cpp/releases/tag/b11707) uses MIT. Its [multimodal API](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) supports local image requests. A strict-schema synthetic request returned eight structured observations in 10.119785 seconds; the four-window extractor processed the 78.734-second longest development clip in 47.694127 seconds. One process snapshot measured 4,412,584 KiB RSS. These are measurements on this machine, not a general latency or accuracy guarantee.

## Actual checks and results

Earlier action-stage full test suite: **316 passed, 2 skipped, 12 warnings in 60.05 seconds**. Later follow-up verification is tracked in `reports/recognition_followup.md`. Warnings remain the existing LightGBM deprecation; skips require optional ignored mock data. New checks cover holdout/queue integrity, ambiguous anchors, provenance, proposal/truth separation, label revision revocation, authentication, range streaming, path/hash gating, local-only requests, blocked proxies/redirects, pinned provisioning, archive traversal/link handling, fixed loopback relaying, uniform action features, numeric observation integrity and the compatibility-checked decoder recovery.

Windows HTTP checks passed: review UI 200; missing key returns 401 for metadata and media; keyed task count 33; a development video of 2,928,496 bytes was delivered with its original content SHA-256 verified locally; original detector UI remains 200. No pixels, video payloads, labels or clip descriptions were emitted to tools. No real review record was written merely to exercise the application.

The completed Qwen candidate's direct local inference check processed the longest development clip in **45.508915 seconds**, using four local model requests and zero attempted external Python connections. Both Windows pages were checked again and returned HTTP 200. The candidate remains rejected for accuracy despite working offline. The existing detector service is unchanged.

Subsequent user-requested preview check: an isolated installed Firefox loaded all 33 tasks, decoded the development preview to readyState 4 with no media error, and passed seeking/playback checks. Two UI bugs were fixed: save readiness now requires playable media, and delayed saves update their original task. The 23 focused API/relay tests pass. No labels or model artifacts changed. See `reports/action_review_preview.md`; refresh the review tab for the updated controls.

Detector score: numeric linear baseline development AUC **0.509804 → 0.500000** for Qwen observations plus linear; Qwen plus shallow LightGBM reaches **0.480392**. Both flag all 32 clips, including **15/15 false alarms**. The complete audit found **one unique clip vector, zero nonconstant features, no missing records and zero holdout use**. Twenty shuffled-target runs average **0.5**, with zero suspicious single features. These checks pass operational integrity but demonstrate no useful detection signal. Neither candidate is promoted. Localization median error regresses **5 → 11.083333 seconds**. The original numeric live model remains only for flow testing; useful detection is still unestablished.

Controlled synthetic red/blue image requests returned the matching color, confirming that images reach the model and constrained JSON can respond to input. The unchanged action recipe on an abstract drawn handoff returned unknown observations; this is not an action-accuracy estimate. The audit therefore does not justify blaming a broken image connection or converting unknowns into invented labels. A different action recipe needs separate predeclaration and independent review. No wider search or threshold tuning against the same development set was performed.

Public source work also progressed: current official Adilet text and commercial-bank scope are verified for two candidate rule clauses in `docs/action_rule_candidates.md`. Their application to actions/videos remains unverified and disabled; they are not inserted into model results as established explanations.

## Reproduce / operate

All Python commands run from the repo root through uv:

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.prepare_action_labels
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m src.video_action_review --port 8002
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.provision_action_model
UV_CACHE_DIR=/tmp/datathon-uv-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DATATHON_THREADS=2 uv run --offline pytest -q
```

Use the existing API key from `data/processed/video-service/api-key.txt` in the local review page. It is never embedded or printed. Windows bridge command, using this machine's existing Node runtime:

```powershell
& 'C:\Users\ibrak\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' '\\wsl.localhost\Ubuntu-24.04\home\ibrak\datathon-kit\scripts\windows_loopback_proxy.cjs' --distro Ubuntu-24.04 --root /home/ibrak/datathon-kit --port 8004 --service review
```

The dependency approval has been received and installation completed. To reproduce provisioning/launch and the bounded pilot:

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.provision_action_model --install
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.serve_action_model
# Separate terminal, with the local model ready:
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m src.video_action_proposals --limit 3
```

The install command needs internet for the bounded downloads; uv's `--offline` governs package resolution. Model inference itself uses local files and loopback only. Assets stay under `models/local-action-model/` and `tools/local-action-model/`; existing mismatches are preserved and fail verification. No hidden evaluation footage is used.

## Files changed

- Schema/queue: `src/video_actions.py`, `scripts/prepare_action_labels.py`.
- Review UI/API: `app/action_review.html`, `src/video_action_review.py`.
- Local model preparation/client: `scripts/provision_action_model.py`, `scripts/serve_action_model.py`, `src/video_action_proposals.py`.
- Bounded detector experiment and diagnostics: `src/video_qwen_actions.py`, encoder routing in `src/video_train.py`, `src/video_sanity.py` and `src/video_inference.py`, `scripts/audit_action_observations.py`, `scripts/migrate_qwen_decoder_cache.py`, `scripts/check_video_offline.py`.
- Existing Windows loopback relay extended with a fixed review-service mode: `scripts/windows_loopback_proxy.cjs`, `scripts/loopback_request.py`.
- Tests: `tests/test_video_actions.py`, `tests/test_video_action_review.py`, `tests/test_video_action_proposals.py`, `tests/test_action_model_provision.py`, `tests/test_loopback_review.py`.
- Plan/status: `docs/action_label_plan.md`, `docs/action_detection_trial.md`, `docs/local_action_model.md`, `docs/action_rule_candidates.md`, this report, `reports/qwen_action_detection.md`, `README.md`, `docs/context.md`, `docs/decisions.md`, `experiments.md`.

Raw data, frozen evaluation, existing model artifacts and unrelated uncommitted files remain preserved. No commits, pushes, external media uploads or public deployment.
