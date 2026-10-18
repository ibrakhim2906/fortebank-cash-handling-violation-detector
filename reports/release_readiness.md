# Delivery audit — 18 October 2026

Owner: release readiness chat. Scope: public service/authentication checks, official source refresh, and a concrete jury handoff. Frontend and recognition work remain with their active chats.

## Completed work

- Added `scripts/audit_release_surface.py`: bounded health/UI/auth probes, optional private-key acceptance check, no redirects, no media uploads or inference calls. It records scoped results and always leaves full-release qualification false.
- Added `tests/test_release_surface.py`: real local HTTP checks for authentication failure, unhealthy/malformed health JSON, redirect handling with a private key, safe origins, immutable evidence and raw-directory protection.
- Wrote `docs/release_rule_matrix.md`: refreshed three official Adilet texts and mapped all fourteen annotation types to source context, visible conditions, exceptions and source gaps. No rule or prediction was changed.
- Wrote `docs/jury_handoff.md`: current URLs, private-key handling, executable audit/checkpoint commands, selected-GPU offline-proof procedure, final ownership/acceptance gates and demo/submission sequence.
- Wrote `docs/release_readiness_plan.md`: bounded scope and file ownership. Captured observations and provenance under ignored `data/processed/release-readiness-20261018/`.

## Fresh measured observations

| Observation | Real output / interpretation |
|---|---|
| First default-urllib surface audit, 04:51:38 local | Six checks failed with HTTP 403; no key was supplied. Cloudflare error 1010 was reproduced separately. |
| Explicit application User-Agent, 04:52:03 | Six observed checks pass; private-key acceptance still not checked. |
| Keyed audit, 04:52:37 | **7/7 pass**: health/UI 200; missing/wrong-key POST and GET 401; valid-key unknown job 404. |
| Later keyed audit, 04:54:33 | **0/7 pass**, every request returned Runpod HTTP 502. Curl, requests and the explicit audit client also observed 502. |
| Pod inspection, 04:55:14 | Model endpoint 8002 returned 200; application 8000 did not respond. |
| Pod inspection, 04:55:34 | Both 8000 and 8002 had TCP listeners. No application/model restart was performed by this audit. |
| Recovered keyed audit, 04:55:46 | **7/7 pass** again, with the same expected 200/401/404 statuses. |
| Final keyed audit, 04:57:41 | **7/7 pass** after restoring the original service environment. This is the latest completed public audit here, not a promise of continued uptime. |
| Focused tests | `26 passed in 7.42s`. |
| Whitespace check | `git diff --check` exited 0 at the test checkpoint. |
| Source refresh | Three full official texts retrieved and hashed; all three text hashes match the existing catalog. One initial SSL read failed and was successfully retried. |
| Service environment restoration, 04:57:23 | Original 81 service package names/versions match exactly after removing only the 28 unintended dev additions. |
| Frozen evaluators | `src/eval.py`: `0ef1963df44a04495ecf95d8c40c82c606ac7987ac410202094c5e671483a762`; `scripts/eval_video.py`: `8effdc878f842accee556e349d5a768ae2b7ea16a79b6d058cdc3a000e61b658`. Neither file was edited by this work. |

Private evidence filenames: `public-surface-first.json`, `public-surface-with-user-agent.json`, `public-surface-keyed.json`, `public-surface-later.json`, `public-surface-recovered.json`, `default-client-health*.json`, `inspect-serving*.log`, `rule-source-refresh.json`, three saved official API responses/clauses, and `delivery-audit-provenance.json`.

The 403 client-filter behavior and subsequent 502 service interruption are different observations. Only the 403 was isolated to the default Python urllib User-Agent; the 502 affected curl and explicit clients too. Do not attribute all failures to the client or assume that changing a User-Agent fixes an unavailable backend. The diagnosis confirmed an application-availability interruption with a healthy model endpoint, followed by recovery; the exact initiating concurrent action was not established.

## Commands run

From the repository root:

```bash
uv run --offline pytest -q tests/test_release_surface.py tests/test_api.py tests/test_network_guard.py tests/test_network_local_peers.py tests/test_offline_registration.py tests/test_offline_connection_timeout.py
git diff --check
uv run --offline python -m scripts.audit_release_surface --api https://p7ayn21cpy6voe-8000.proxy.runpod.net --key-file data/processed/runpod-setup-v1/remote-api-key.txt --json data/processed/release-readiness-20261018/public-surface-recovered.json
```

Earlier audit runs used distinct output filenames to retain failures. Additional read-only work used `git status`, `rg`, `cat`/`sed`, curl, direct HTTPS retrieval of public legal text, and existing authenticated SSH transport for bounded Pod process/listener/log inspection. No video bytes, frames, private clip descriptions or keys were sent to web/AI tools. The key was read locally and sent only to its already configured HTTPS service.

## Diagnostic side effect and unresolved gates

The first nominally read-only Pod inspection used `uv run --offline`, which synchronized the existing locked development group and installed **28 packages**. The installation is visible in `inspect-serving.log`. This was an unintended environment mutation. Subsequent remote diagnostics use `uv run --no-sync --offline`; no declared dependency, model weight, prompt, threshold or inference setting was changed.

The 109-package service inventory was compared with the earlier 81-package service subset of `pod-installed-licenses.json`: exactly 28 additions and zero changed/missing original versions. A `uv sync --frozen --no-dev --offline --dry-run` proposed exactly those additions for removal. A guarded corrective command required that the inventory still matched and no remote pytest process was active before restoring the service environment. Postcheck output at 04:57:23: `service_distributions: 81`, `matches_original_service_inventory: true`, `removed_only_own_locked_dev_packages: 28`. Evidence: `inspect-runtime-inventory.json`, `inspect-dev-restore.log`, `restore-dev-side-effect.log`. No model environment, model process or pre-existing package was removed. The pre-existing overall licensing-policy gaps still require resolution.

This chat did not edit serving/frontend/recognition code, train or score a model, upload a video, consume a reserve checkpoint, run the irreversible GPU guard, signal/restart model processes, deploy code, create paid resources, publish Git content or submit the entry. Recognition score before → after: **not applicable; no recognition change or evaluation was performed**.

The following remain outside what these passing tests prove:

1. Selected-GPU actual offline proof. The existing script arms irreversible process-wide network restrictions and must run in a coordinated qualification window for the frozen release. No new proof is claimed.
2. Final selected recipe and its live primary-model quality/latency; the recognition chat is still responsible for measured candidate selection.
3. Finished frontend visible-browser/live-upload verification, served legal applicability and original evidence correctness.
4. Full requested robustness/API scope, actual license-policy resolution, correct organizer submission destination, private key hand-in and funded judging uptime.

The temporary HTTP 502 is an observed release risk: repeat external checks after every final restart and freeze. The public-surface audit deliberately cannot convert HTTP health, a test count or a source catalog into recognition accuracy, offline portability or a final jury score.
