# Game plan — one human + Codex, no sleep, now → 12:00 Oct 18

**Goal:** the most capable solution within the rules — Facts → Rules engine + local Protocol Auditor (GPU), with a CPU fallback that always works.
Score = detection 25 · timecode 15 · type 15 · legal 15 · robustness 10 · product 10 · tech 10.
Two tracks run in parallel: **Codex builds**, **you do the human-only work** (pod, review, filming, legal check).

## Timeline
| Time | Codex | You |
|---|---|---|
| 19:20–20:00 | `run V0` (service skeleton, contract tests) | Commit kit. **Create the RunPod pod** (deploy.md, 10 min). Push repo to a private GitHub. |
| 20:00–20:45 | `run V1` (perception; cache in background) | cloudflared on laptop → test URL from phone (CPU fallback ready early) |
| 20:45–21:15 | `run V2` (eval) | Review → `git commit -m "freeze eval"`, `git tag good-v2` |
| 21:15–22:00 | `run G1` (pod setup over SSH, vLLM + service, public pod URL) | Check pod URL from phone |
| 22:00–01:00 | `run V3` → `run V4` (scene, facts, rules engine, Tier 1 + cash/door/safe) | Review fact strips; tell Codex what the facts miss |
| 01:00–03:00 | `run G2` (Protocol Auditor + fusion) on the pod | Watch the auditor log on X-1, X-2, Х-22, Х-32, Х-33, Х-34, Х-47 |
| 03:00–04:00 | `run V5 with: <type>` loop (rules) | **Film own clips** (see below) |
| 04:00–06:00 | `run G3` (auditor tuning) + `V5c` timecode calibration + `V5t` type names | Approve keep/drop by eval only |
| 06:00–07:30 | `run V7 with: own clips` (new-angle test, speed, AUDITOR on/off) | Review |
| 07:30–08:30 | `run V8` (fact timeline + auditor checks in UI, `/rules` page) | Check UI on phone |
| 08:30–09:30 | `run V9` (legal check, docs/submission.md) | Verify every quote on adilet.zan.kz |
| 09:30–10:30 | `run V10` on the pod URL **and** the laptop URL; 5 videos back-to-back | Pick the jury URL (higher score + stable) |
| **by 11:00** | Freeze | **Hand in URL + key** (1 h buffer) |
| 11:00 → jury done | — | Pod running + laptop awake as backup; `/health` checks |

## Scoring assumptions
Decided ourselves in `docs/jury.md` (bilingual type names, onset timecodes, all violations most-confident-first). `run V5t` applies them.

## Own clips (03:00–04:00, alone, ~45 min)
Phone on a stack of books, 2 new angles (side, high-front). Desk + box as safe + a door. Act out: leave with paper "cash" on the desk, leave door open, drink at the desk, read a book, bag on the desk, laptop open when leaving, keys left in a lock; plus 3 clean runs (desk cleared, door closed). Save to `data/raw/own/` + `labels.csv` (video, second, type). This is the never-tuned new-angle test.

## If you fall behind (cut in this order)
1. V8 extras (feedback, Docker) → 2. own clips → 3. G3 auditor tuning → 4. procedure rules that fire on clean videos → 5. G2 auditor entirely (CPU mode is then the jury URL).
**Never cut:** V0, V2 freeze, a working public URL, V7 speed check, V9 legal check, V10 external test, hand-in by 11:00.
If a phase takes > 2× its slot: `run VS`, then cut.

## Safety rails (no sleep)
- `git tag good-<time>` and `git push --tags` after every passing phase (the pod pulls from GitHub); after 08:00 roll back instead of debugging.
- Codex needs network/SSH permission for G1 (Codex settings → full access, or run the printed commands yourself).
- Never approve changes to `service/app.py` without `uv run pytest` green.
- The CPU fallback URL must work at all times; the pod is an upgrade, not a single point of failure.
- 10-minute break every 2 hours. Hand-in at 11:00 is non-negotiable.

## Review loop (with Claude)
After every phase paste Codex's final report + the eval summary lines. You get back: approve / reject, what to fix, the next exact prompt.
