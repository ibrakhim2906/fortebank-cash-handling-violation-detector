# START HERE — ForteBank video case

**What we build:** a local AI service that checks cashier videos for cash-discipline violations and returns verdict,
type, timecode, evidence frames, description and the НБРК rule — through a UI and an API the jury tests automatically.
**Our edge:** Facts → Rules. Perception turns video into a timeline of facts; editable JSON rules linked to real НБРК
articles turn facts into violations. New violation type = new rule, no retraining. A local Protocol Auditor (open VLM on our own GPU server) checks procedure steps. Runs offline; CPU-only fallback; auto-calibrated per camera.

## Read in this order (10 minutes)
1. `docs/case/requirements.md` — every requirement with its check
2. `docs/case/architecture.md` — the approach, models, licences, timing
3. `docs/case/rules_engine.md` — facts and rule format
4. `docs/case/violations.md` — what we think each video shows (confirm by watching!)
5. `PLAYBOOK.md` — phases V0–V10 for Codex; `GAMEPLAN.md` — who does what when

## Setup (once)
```
mkdir -p data/raw/open && unzip open.zip -d data/raw/      # → data/raw/open/videos, open_labels.xlsx
sudo apt install -y bubblewrap ffmpeg                       # Codex sandbox + video fallback
```
Then in Codex (project = `~/datathon-kit`, agent environment WSL): `run V0`.

## Talking to Codex
`run V0` … `run V10`, `run V5 with: cash_left_out`, `run VS`. Reasoning: **High** for V3, V4, V5, V7; **Medium** otherwise.

## You (solo) + Codex
- Codex builds; you review every phase report and eval numbers (paste them to Claude for a second opinion).
- Human-only tasks: create the RunPod pod, check URLs from your phone, film own clips (03:00), verify legal quotes on adilet.zan.kz (V9), hand in by 11:00.
- Scoring assumptions are fixed in `docs/jury.md`; timeline and cut order in `GAMEPLAN.md`.
