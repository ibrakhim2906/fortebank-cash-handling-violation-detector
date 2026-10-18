# Sanitized aggregate experiment log for public source review

Private run records and per-video details are omitted. The source test harness expects this public log path to exist; synthetic tests continue to write their own isolated run-local logs.

| Recipe | DEV correct | TP | FN | FP | Localized within2s |
|---|---:|---:|---:|---:|---:|
|32B image16 proposal baseline|20/30|12|4|6|5/16|
|32B image16 selective storage veto|23/30|12|4|3|5/16|

All40recordings were historically reviewed. These are development/regression aggregates, not an independent jury score. Full new40-video API qualification remains in progress in the original workspace. No model training or tests ran to produce a new recognition claim in this export.

Pending, not run (no GPU/footage in this export): `AUDITOR_RECALL=rescue`, `=localize`, `=on` on DEV30 with the selected32B recipe. No result is claimed until the frozen evaluator produces one.
