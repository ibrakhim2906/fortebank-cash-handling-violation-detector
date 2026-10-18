# Local object-transfer pilot review

Open the existing private action-review page (Windows loopback port 8004 while its bridge/service run). Choose **Object-transfer pilot**, enter the existing local API key and your initials, then **Load queue**. Do not send footage, screenshots, labels or clip descriptions to chat.

The queue starts with 16 screening tasks from distinct development recordings: eight from the training pool and eight from the blind-pilot pool. These are starting points, not predicted classes. Further development tasks follow if the genuine labels do not fill the quotas. The eight reserved recordings are excluded. Official positive/negative labels and machine proposals are hidden in this mode.

1. Watch the video locally. The anchor is only a starting point; it does not establish a transfer or a violation.
2. Set start/end with the player buttons. Select only one of these outcomes:
   - **An object passes between two people:** both sides of the exchange must be visible. Mark the whole exchange in an interval of at most eight seconds. Select an original supporting frame with **Use current player frame**, then confirm that it supports the exchange.
   - **The same person moves an object:** watch at least eight complete seconds and confirm that no transfer between people occurs in that interval.
   - **Hand/object contact, but no transfer:** use the same complete, clearly visible eight-second minimum and no-transfer confirmation.
   - **Other activity:** save genuine background coverage of at least eight seconds. This is retained but does not fill the hard-counterexample quota.
   - **I cannot tell reliably:** use this when occlusion, an unclear object or incomplete coverage prevents a confident answer. It remains ambiguous and is excluded from training.
3. Choose visibility, confirm that you watched the selected coverage, and **Save local review**. Revisions append; earlier reviews are preserved. **Next unreviewed** advances the queue.

“I watched the entire clip” means exactly that; it changes coverage to the complete duration. Do not use it after watching only an anchor interval. Action absence applies only to the fully watched coverage. No option asks whether a bank rule was broken, who owns the object, or what a person intended.

The feasibility pilot needs four confirmed transfers and four relocation/contact counterexamples in **each** pool, from eight distinct recordings per pool. Counts are screening progress, not proof that the balanced pilot is available. Do not change a label to meet a quota. If genuine examples are missing, the pilot stays blocked.

Local files are under `data/processed/action-labels/transfer-pilot-v1/`: `queue.json` contains the exact task IDs, anchors, intervals and frozen fold membership; `reviews.jsonl` is the private append-only human journal. Existing general reviews/proposals remain in the parent action-label directory.

Preparation command (metadata only, no decoding):

```bash
uv run --offline python -m scripts.prepare_transfer_review
```

The trainer's default label export contains training-pool outcomes only. Blind-pool labels are released only after the fitted head/artifact has been frozen and its manifest fingerprint supplied. Review and model predictions must stay separate; machine agreement is not human verification.
