# Jury handoff — current service and remaining release gates

Prepared 18 October 2026, 04:54 UTC+05:00. This packet supports the delivery workstream; it does not select or deploy the recognition chat's final candidate. `reports/progress.md` and its timestamped experiment artifacts remain authoritative for recognition changes after this snapshot.

Availability update, 04:57:41 local: all seven surface/auth checks pass, following another pass at 04:55:46. At 04:54:33 they all returned Runpod HTTP 502; a Pod inspection found model port 8002 healthy while application port 8000 did not respond. A subsequent inspection found port 8000 listening again, without a restart by this audit. Keep all snapshots and recheck after the final deployment; the initial pass was not continuous-uptime proof.

## Addresses and credential handling

- UI: https://p7ayn21cpy6voe-8000.proxy.runpod.net/
- API base: https://p7ayn21cpy6voe-8000.proxy.runpod.net
- Health: https://p7ayn21cpy6voe-8000.proxy.runpod.net/health
- Keyed upload: `POST /api/v1/analyze`, multipart field `file` containing the video.
- Keyed status: `GET /api/v1/jobs/{job_id}`, header `X-API-Key` on both API endpoints.

The key is delivered privately through the organizer's designated submission channel. The current local private key file is `data/processed/runpod-setup-v1/remote-api-key.txt` (checked mode `0600`); its value is deliberately absent here. Do not copy the ignored setup directory into Git, slides, screenshots or a public artifact. Successful authentication was checked against this public origin without displaying the key.

Required completed results use `violation` or `no_violation`; a negative result has an empty `violations` array. Events carry `type`, `timecode_sec`, `description` and 1–3 original JPEG/base64 evidence frames with `timestamp_sec` and `image_base64`. Put qualified source references in `description`, preserving the official schema. A failed job must remain a failure; a timeout/fallback must not be described as successful primary-model recognition.

## Re-run the public-surface check

From the repository root, use a fresh output filename each time:

```bash
uv run --offline python -m scripts.audit_release_surface \
  --api https://p7ayn21cpy6voe-8000.proxy.runpod.net \
  --key-file data/processed/runpod-setup-v1/remote-api-key.txt \
  --json "data/processed/release-readiness-20261018/public-surface-$(date -u +%Y%m%dT%H%M%S).json"
```

This sends seven small requests and no media: health, the UI document, missing/wrong-key rejection on both API endpoints, and a valid-key unknown-job lookup. A pass proves those observations only. It does not execute recognition, inspect browser rendering, qualify the deployed recipe, or establish offline operation. The origin must be HTTPS, except for a local loopback test. Redirects are rejected to avoid moving a credential to another origin. Response bodies and keys are not saved in the audit.

Cloudflare returned `403`/error `1010` to the default Python urllib User-Agent, while curl and the explicit `datathon-release-audit/1.0` User-Agent worked. The audit sets the latter. Check the organizer's actual client externally before handoff; a browser/curl success does not establish acceptance of every automated client. This observation does not establish a recognition failure or a guaranteed jury-client incompatibility.

## Actual primary-model and API qualification

After the recognition chat freezes a candidate, record the model profile, exact weight revision, code hashes, all serving environment flags and recipe SHA-256. The frontend recipe selector presents recorded benchmark comparisons; it must not imply that changing that selector changes the deployed model.

Run the existing checkpoint against that frozen public service with the matching honest report and freeze JSON:

```bash
uv run --offline python -m scripts.mission_checkpoint \
  --api https://p7ayn21cpy6voe-8000.proxy.runpod.net \
  --key-file data/processed/runpod-setup-v1/remote-api-key.txt \
  --honest "$RELEASE_HONEST_REPORT" \
  --freeze "$RELEASE_FREEZE_JSON" \
  --run-dir "$NEW_CHECKPOINT_DIR"
```

The three shell variables are operator-selected paths for the actual frozen candidate and a new ignored directory under `data/processed/`, not example results to fabricate. This command is a substantial evaluation: it performs uploads/model calls, consumes one of at most four reserve checks, and full scope includes 40 clean uploads plus 280 transforms and runtime repeats. Schedule it with recognition after the active experiment; do not start it concurrently from this packet. `--quick` explicitly produces a partial checkpoint and cannot establish full-set robustness. Keep reserve details private; every supplied recording has already been reviewed, so this reserve is not independent validation.

Confirm two distinct actual completed jobs through the selected primary recipe within the required five-minute limit; the existing mission uses a stricter 200-second target. The runner now selects the slowest observed completed clean source for repeats. Preserve upload-to-result latency, job IDs, engine, recipe and trace hashes; replay time does not qualify live latency. A changed deployment invalidates proof for an earlier recipe.

## GPU offline proof — coordinate before running

The CPU-only offline result does not qualify the selected GPU model. The existing GPU proof signals each model process and installs an irreversible seccomp socket-creation restriction; it must not be run while recognition needs unrestricted workers or may download/restart a model. No process was signaled by this release audit.

At the controlled qualification window:

1. Finish the active experiment. Recognition/deployment owner confirms the exact selected runtime and model processes, guard registration directory, frozen source, and model readiness.
2. Select a file programmatically asserted to belong to frozen DEV, not a reserve or hidden jury video. Preserve the same frames/context, prompts, policy, per-call timeout, preprocessing and model endpoint as the release. Do not silently use the script's environment defaults.
3. On the existing Pod, in `/workspace/datathon-kit`, inherit the frozen release environment, including `AUDITOR=on`, `AUDITOR_PROFILE`, `VLLM_URL=http://127.0.0.1:8002`, `AUDITOR_CALL_TIMEOUT`, `AUDITOR_FRAMES`, `AUDITOR_CONTEXT`, `AUDITOR_RESOLUTION`, `AUDITOR_SCAN`, `AUDITOR_WORKFLOW`, `AUDITOR_CONFIRMATION`, `PERCEPTION_MODE`, and `OFFLINE_GUARD_LOG_DIR`. Use the installed, matching guard registration; a required restart belongs to the deployment owner.
4. Run the existing proof with the actual selected profile and DEV file:

```bash
uv run --no-sync --offline python -m scripts.check_gpu_offline \
  --profile "$RELEASE_PROFILE" \
  --video "$DEV_VIDEO" \
  --json "$NEW_GPU_PROOF_JSON"
```

5. Require `passed: true`, `status: done`, `engine: mission-v1`, positive `primary_model_requests`, and current successful guards for every live model process plus worker denial probes. Verify the runtime signature and environment against the release; inspect the evidence, not only the exit code. Retain all guard records. A fallback, missing registration, stale proof, guard failure or zero model calls does not pass.
6. Restore/verify normal serving through the deployment owner if needed. Recheck the final public URL with a real upload. Do not restart or replace the model just to make this document appear complete.

This proof retains audited same-machine IPC and one pre-established model connection while denying new internet sockets. Describe that mechanism accurately; it does not mean the GPU machine or public browser stopped requiring network transport between them.

## Final hand-in and demo sequence

| Gate | Acceptance evidence | Owner |
|---|---|---|
| Final recognizer | Completed measured comparison, exact frozen recipe and limitations; no assumed 90–95% result | Recognition |
| Finished UI | External browser upload → processing → result → correct source time/frame/rule; positive, clean and failure/loading flows | Frontend |
| Public API | This audit, fresh selected-primary uploads, schema/auth and actual latency evidence | Delivery with recognition |
| Offline processing | Successful actual GPU proof for the same selected configuration | Delivery with deployment/recognition |
| Normative sources | `docs/release_rule_matrix.md`, reviewed served source/evidence applicability | Delivery with human review where context is missing |
| Licenses | Preserve `docs/submission.md` inventory/notices and actual runtime terms; resolve any literal internal allowlist conflict without relabeling licenses | Delivery / human scope decision |
| Handoff | UI URL, API URL, private key, run/use instructions and final report sent through confirmed organizer destination | Human |
| Availability | Confirm judging window, funded existing Pod, laptop wake settings if used for backup, and external recheck | Human / deployment |

Present: problem → live upload → one independently checked event with original evidence/time → qualified source → measured scope/limitations → deployment. State that results support human review. If shortlisted, the official FAQ gives five minutes plus two minutes for questions; a three-minute walkthrough can be a rehearsal version. Pick demonstration recordings deliberately and identify them as known examples. A backup recording is a fallback presentation artifact, not evidence of live inference.

The plan calls for hand-in by 11:00 local on 18 October ahead of the stated 12:00 deadline; the organizer's actual destination and judging uptime window still require confirmation. Keep the existing Pod funded through testing within the approved total cap. No new allocation, model download, key rotation, public sharing, upload evaluation or final submission was performed by this audit.

Diagnostic environment side effect, corrected at 04:57:23 local: the first Pod inspection used `uv run --offline`, which automatically installed 28 already-locked development packages. This was unintended; subsequent diagnostics use `uv run --no-sync --offline`. A saved inventory and dry run established that these were exactly 28 additions, with all 81 original service package versions unchanged. After checking that no remote pytest process was active and that the inventory had not changed, `uv sync --frozen --no-dev --offline` removed only those additions. A postcheck confirms that all 81 original service package names/versions match again. No project dependencies, model settings or model-environment packages were changed. Preserve `data/processed/release-readiness-20261018/restore-dev-side-effect.log` alongside the inspection logs. This restoration does not resolve the existing overall licensing-policy gaps.

## Outstanding at this snapshot

Final recognition recipe/deployment, fresh full-flow browser/API qualification, successful selected-GPU offline proof, complete applicable licensing resolution, private organizer hand-in and judging uptime confirmation remain open. The source refresh and public-surface checks are complete. Older README/Checklist statements about the numeric baseline or an untouched holdout must not be presented as the final GPU release state.
