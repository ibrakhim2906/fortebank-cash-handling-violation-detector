# Facts → Rules engine (the core idea)

**Why:** ForteBank has no closed list of violations, regulations change, cameras differ. An end-to-end classifier trained
on 40 staged clips can't be maintained. We separate:
1. **Perception** (ML, generic, reusable): video → *fact timeline* — what is true at each moment.
2. **Compliance** (declarative, editable by non-ML staff): JSON rules over facts → violations, each linked to a real legal article.

A new violation type = a new rule in `service/rules.json` (+ at most one new fact). No retraining.

## 1. Fact timeline
Sampled every 0.5 s. Each fact is a boolean or number per timestamp, produced in `service/pipeline/facts.py`
from detections + tracking + per-video scene calibration (`scene.py`). Smooth with a median over ~1.5 s.

| Fact | Type | Meaning (relative to auto-calibrated scene) |
|---|---|---|
| `n_people` | int | People inside the room (reflections / people behind glass removed) |
| `staff_present` | bool | Cashier track is in the room |
| `cashier_at_desk` | bool | Cashier is within the cashier zone (behind the desk) |
| `client_present` | bool | A person is in the client zone (opposite side of the desk) |
| `extra_person` | bool | More people than cashier + 1 client, or a non-cashier on the cashier side / at the safe |
| `person_entered` | event | A new track appears from the entry side |
| `cashier_leaving` | event | Cashier leaves the cashier zone heading to the exit (onset) |
| `safe_open` | bool | Safe region differs from its closed reference (door open) |
| `desk_items` | bool | Desk region contains objects vs. the empty-desk reference (cash/bags/documents) |
| `phone_in_hand` | bool | `cell phone` box overlapping the cashier's upper body / hands |
| `bag_on_desk` | bool | handbag/backpack/suitcase or coat on the cashier's desk |
| `drink_on_desk` | bool | cup/bottle on the cashier's desk |
| `outsider_present` | bool | A person who is not the cashier or the served client is in the booth, on the cashier side, or at the equipment/safe (roles, not identities — see violations.md §4) |
| `door_open` | bool | Door region differs from its closed reference (door ajar/open) |
| `person_in_doorway` | bool | A person is currently passing through the door region |
| `cash_on_desk` | bool | Banknotes visible on the desk (OWLv2/Grounding DINO on keyframes, or desk-crop classifier) |
| `safe_open` | bool | Safe door open (safe-region change vs closed reference / crop classifier) |
| `cashier_at_safe` | bool | Cashier within reach of the safe |
| `key_in_safe_lock` | bool | Key visible in the safe lock (OWLv2 on the safe crop) — experimental |
| `laptop_screen_on` | bool | Laptop open with a lit screen (laptop box + brightness of the screen area) |
| `hand_to_hand` | event | Wrists of the cashier and another person within ~0.1 desk widths over the desk (pose) |
| `counter_active` | bool | Banknote counter in use (motion inside the counter box while cash present) |
| `client_left` | event | Client track leaves the client zone / the room |
| `cash_received` / `cash_given` | event | Cash passes client → cashier / cashier → client (hand or tray motion at the boundary, during a transaction episode) |
| `id_document_shown` | bool | Small card/document passed or shown before cash (OWLv2 «passport», «ID card») — experimental |
| `document_given` | bool | Receipt/paper handed to the client after the operation — experimental |
| `drink_at_face_or_desk` | bool | cup/bottle near the cashier's face/hand or on the cash desk |
| `book_in_hands` | bool | `book` box overlapping the cashier's hands/lap |

Every fact also stores **evidence**: the boxes that produced it (for highlighting in frames).

## 2. Rule format (`service/rules.json` → `types.<key>.logic`)
```json
"cash_left_out": {
  "name": "Наличные деньги оставлены вне сейфа (хранилища)",
  "logic": {
    "trigger": "cashier_leaving",
    "require_any": ["cash_on_desk", "bag_on_desk"],
    "timecode": "trigger",
    "evidence": [0, 2.0],
    "cooldown_sec": 15
  },
  "rules": [{"source": "NBRK_231", "point": "п. 20", "quote": "…"}]
}
```
Semantics (keep the engine tiny, ~150 lines, fully unit-tested):
- `trigger`: an event fact or the rising edge of a boolean fact → candidate time `t0`.
- `require_all` / `require_any` / `require_none`: boolean facts evaluated at `t0` (window ±`window_sec`, default 1.0).
- `require_after` / `require_before`: a fact must hold (`equals`) for ≥ `min_sec` within `within_sec` after/before `t0`.
- `absent_before` / `absent_after`: the fact is **never true** in `[t0 − within_sec, t0]` / `[t0, t0 + within_sec]` (for "a required step didn't happen": no ID shown, no receipt, counter never used). Clamp the window to the video bounds (a step can only happen inside the video — e.g. X-2's acceptance at 10 s has only 10 s of "before"); fire only if at least `min_observed_sec` (default 3 s) of the window was observed.
- `min_duration_sec`: for state triggers, how long the state must last.
- `timecode`: `"trigger"` (default) or `"trigger-1.0"` etc.
- `evidence`: frame offsets to export (first = timecode frame).
- `cooldown_sec`: merge repeated triggers of the same rule into one episode (default 10).
- `alt`: an alternative condition set (same keys) — the rule fires if the main set OR `alt` matches.
- `enabled`: false → rule is evaluated and logged (for eval) but not reported. Enable only after it is silent on clean videos.
Rules are evaluated independently; results sorted by time; max 3 per video (strongest first).

## 3. Why this scores well
| Criterion | How Facts→Rules helps |
|---|---|
| Detection 25 | Each rule encodes exactly what humans annotated as the difference between violation and look-alike |
| Localisation 15 | Onset defined per rule (`cashier_leaving`, `person_entered`) = what labels mark |
| Classification 15 | Type = rule that fired → consistent names |
| Legal 15 | Rule ↔ article ↔ quote is one record; nothing generated by a model |
| Robustness 10 | Facts are relative to auto-calibrated scene, not pixels |
| Product 10 | Fact timeline in the UI makes every verdict explainable |
| Tech 10 | Clean separation, extensible, CPU/offline, Docker, tested engine |

## 4. Tests (`tests/test_rules.py`)
Synthetic fact timelines (no video needed): leaving with items → fires at onset; leaving after clearing desk and closing safe → silent; client entering and sitting → silent; third person → fires; cooldown merges duplicates; timecode offsets applied.

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
