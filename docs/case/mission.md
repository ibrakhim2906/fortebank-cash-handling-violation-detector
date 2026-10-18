# MISSION — reach an estimated rubric score ≥ 90 on honest validation

You (Codex) own recognition quality until **06:30**. Keep iterating — measure, diagnose, change, re-measure — until the
targets in §1 are met on **honest** validation (§2), or 06:30 arrives. Do not stop to ask questions: decide using the docs.
Stop only if blocked by missing permissions/credentials, the GPU pod being down, or a conflict with `docs/decisions.md`.
If your session is interrupted, resume from `reports/progress.md`.

## 0. State check (first 10 minutes)
- Read AGENTS.md, all of `docs/case/`, `docs/annotations.md`, `docs/decisions.md`, `docs/jury.md`, `PLAYBOOK.md`.
- Verify prerequisites: V0 (service), V1 (perception), V2 (eval frozen), G1 (pod: vLLM + service, public URL). If any is missing, build it first per PLAYBOOK, then continue here.
- You are authorised to add a `--json <file>` output flag to `scripts/eval_video.py` (output only — no change to how any metric is computed). Note it in docs/decisions.md.

## 1. Objective: one number to maximise — `scripts/rubric_score.py`
Write `scripts/rubric_score.py`: reads eval JSON outputs (dev/honest, holdout, augmented, API run) and prints the
**estimated rubric score** (brief §9 weights) plus each component:
| Component | Formula (proxy) | Max |
|---|---|---|
| Detection | 25 × honest verdict accuracy (per video) | 25 |
| Localisation | 15 × (0.5·hit@±1 s + 0.5·hit@±2 s) over labelled events | 15 |
| Classification | 15 × type accuracy over hit events (`docs/case/types.csv`; a hit counts if any annotated type matches) | 15 |
| Legal | 15 × share of reported violations whose type has ≥ 1 verified rule in rules.json (types with `rules: []` count 0.5) | 15 |
| Robustness | 10 × min(1, augmented accuracy / clean accuracy) × leave-one-setup-out accuracy | 10 |
| Product | 10 × (pytest green) × (worst-case time ≤ 200 s) × (0 failed/timeout jobs in API mode) | 10 |
| Tech | 10 × (no outbound network during processing — run once with `HF_HUB_OFFLINE=1` and network blocked) × (licence table complete) | 10 |
**Targets:** estimated score ≥ 90 · recall = 1.0 (honest) · false alarms ≤ 1 clean video · hit@±2 s ≥ 0.9 · type accuracy ≥ 0.85 · worst-case ≤ 200 s on the pod, twice in a row.

## 2. Honest validation (mandatory — 40 videos overfit easily)
- **Holdout:** at start, freeze 10 videos (≈5 violation / 5 clean, spread across camera setups and types; never Х-4/Х-47 split across sets). Write the list to `reports/holdout.txt`. Never look at holdout per-video errors while iterating; evaluate it only at checkpoints (max 4 times).
- **Dev = the other 30.** Thresholds/offsets fitted on dev with 5-fold CV grouped by camera setup; "honest" metrics = out-of-fold.
- **Robustness set:** augmented copies of every video (h-flip, 12 % crop, ±10° rotation, −35 % brightness, 0.75× resolution, re-encode).
- **Own clips** (`data/raw/own/`, if the human adds them): test-only, never tuned on.
- **Forbidden (= cheating, also fails on the hidden test):** using filenames, durations, fingerprints or labels at inference; per-video special cases; hard-coded timecodes. Self-check at each checkpoint: renamed + re-encoded copies must give the same results.

## 3. The loop (repeat)
1. Run eval (dev, honest). Print the per-video error table with what the pipeline saw (facts + auditor answers) vs truth.
2. Pick the **error cluster with the largest estimated-score impact** (detection > localisation = classification > others).
3. Form one hypothesis → make the smallest change that tests it (prompt, frame selection, crop, threshold, fact, rule, fusion weight, model).
4. Re-run. **Keep** if estimated score ↑, recall not ↓, false alarms not ↑, worst-case time OK; otherwise `git checkout` back.
5. Append one line to `experiments.md` and to `reports/progress.md`: time · change · score before → after · kept/reverted · why.
6. On improvement: commit, `git tag good-<HHMM>`, `git push --tags`, `git pull` + restart on the pod.
**Checkpoint every 2 h or 10 iterations:** holdout + robustness + API-mode run through the pod's public URL → summary in progress.md.

## 4. Levers, in order of expected value
A. **Auditor model.** Default **Qwen3-VL-8B-Instruct** (Apache-2.0, vLLM-supported; verify the licence on the Hugging Face card and record it). Compare against Qwen2.5-VL-7B on dev. Our pod is an **NVIDIA A40, 48 GB** (Ampere: no native FP8). Run the 8B in bf16. Also try a larger Apache-2.0 Qwen3-VL (e.g. 32B) as an **AWQ/GPTQ 4-bit** build that fits in ~20–25 GB (the quantised repo's licence must also be Apache-2.0/MIT). Keep whichever scores best on dev-honest within the 200 s worst-case budget.
B. **Frame selection.** Denser frames around candidate events; extra high-res crops of desk / safe / door / hands / laptop; timestamps on every frame; pass short clips as video input if the model supports it.
C. **Prompting.** One focused question per checklist item on its best frames vs one big checklist; ask for the frame time of the evidence; self-consistency (2 samplings or 2 crops must agree) for Tier-3 items; optional reference images of the *correct* protocol taken from **other** training videos (never from the video being evaluated).
D. **Perception & facts.** Person filtering (glass wall), role assignment (cashier / client / outsider), door & safe state references, cash-on-desk detection, pose for hand-to-hand.
E. **Fusion.** Per-type trust weights between rules and auditor; enable/disable each type based on honest metrics; recall-first tie-breaks.
F. **Timecode.** Per-type offset calibration fitted on dev folds only (V5c).
G. **Types.** Bilingual names per docs/jury.md; most-confident-first ordering.
H. **Speed.** Batch auditor calls, vLLM prefix caching, lower resolution where accuracy holds, run perception and auditor concurrently.

## 5. Guardrails (all AGENTS.md hard rules apply)
- **Budget:** pod costs $0.59/h, $15 total, auto top-up off → never create extra pods or bigger GPUs; never leave long jobs running needlessly. Report remaining runtime in each checkpoint.
- **No cloud AI APIs** in the service. Internet is allowed only to read documentation/papers and download open-source packages and weights.
- Licence of every model/library recorded in `docs/submission.md`, under the organisers' rule (R50–R56): open source with commercial use permitted. Model weights Apache-2.0 / MIT / BSD only; the supporting runtime may also use MPL, LGPL and PSF terms, and NVIDIA CUDA runtime terms for `nvidia-*` distributions. GPL/AGPL, non-commercial or unidentified licences fail. Check with `uv run python -m scripts.verify_license_policy`. (18 Oct 2026, human decision; replaces the earlier Apache/MIT/BSD-only wording for libraries.)
- `uv run pytest` green after every change touching `service/`; `AUDITOR=off` (CPU fallback) must still run end-to-end.
- Legal quotes in rules.json change only to match verified act text (docs/case/legal.md).
- Never delete a working capability to gain score elsewhere without logging why.

## 6. Done = by 06:30
- Best configuration deployed on the pod, tagged `mission-best`, CPU fallback working.
- `reports/mission_report.md`: estimated rubric score (dev-honest, holdout, robustness, API), per-type table, per-video failures that remain, levers tried and dropped, final model choices + licences.
- Then continue PLAYBOOK from V7.

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
