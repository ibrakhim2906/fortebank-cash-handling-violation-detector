# Local action-review preview check — 17 October 2026

The requested preview was checked in an isolated instance of the already installed Windows Firefox 158.0, using a disposable profile and loopback browser automation. No new dependency or browser was installed. The normal Windows review page remains `http://127.0.0.1:8004/`.

## Measured results

- The blank page renders correctly, with no horizontal overflow at the tested desktop viewport. A screenshot was captured only before authentication, with an empty key input, hidden workspace and unloaded video.
- An attempt to load the queue without a key is rejected.
- The existing private key loads all **33 development review tasks**.
- Local playback reaches **readyState 4**, with **media error code 0**. The tested video reports 464 × 848 dimensions and a finite duration.
- Seeking succeeds and the playback clock advances after play. Save becomes enabled only after playback is ready.
- **No labels were written and no bank-video screenshot was captured.** The existing annotation journal and all model sources/artifacts remain unchanged.

The browser returned only numeric/boolean playback and authentication results. Actual footage, model suggestions, action labels and clip descriptions stayed local. The check verifies one example's browser playback, not every video or the correctness of any proposed action label. Qwen detector quality remains rejected as documented in `reports/qwen_action_detection.md`.

## Fixes made during inspection

Changed `app/action_review.html`:

1. The page previously announced Ready and enabled Save immediately after fetching bytes. It now waits for the current video's `canplay` event with `readyState >= 3`. Decoder/format errors show a clear failure and disable Save. Listeners are removed when tasks change.
2. A delayed save could previously update whichever task was selected when the response arrived. It now captures the submitted task ID, updates only that queue entry and avoids writing status text into a different task. An in-flight save cannot enable Save while the next video's playback is still loading.

The existing review API and loopback tests passed: **23 passed in 2.13 seconds**. A synthetic JavaScript harness also passed readiness gating, decoder errors, listener cleanup and the delayed-save/navigation race without loading real media or writing the journal.

## Commands and evidence

Run from the repository root:

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline pytest -q tests/test_video_action_review.py tests/test_loopback_review.py
```

The existing Windows Node runtime executed the private check script at `data/processed/review-preview-check/check_preview.cjs`. Numeric output is preserved at `data/processed/review-preview-check/result.json`; the safe blank-page image is `data/processed/review-preview-check/preview_blank.png`. These ignored local artifacts contain no frame payloads or action descriptions. The disposable browser profile was closed and removed.

Computer Use initialization was attempted first, but its Node REPL rejected this WSL workspace with `sandboxCwd is not a local file URI: file:///home/ibrak/datathon-kit`. The isolated Firefox check does not establish that the Codex embedded browser uses the same decoder; it provides direct local-browser playback evidence. Refresh the review tab to load the updated UI.
