# Isolated workspace-focus candidate

Read `docs/recognition_handoff.md` first. This package contains an **unapplied experimental patch**, the exact generated modules used by its focused tests, the patch generator, and validation provenance. It is not the selected recognizer and has no bank accuracy measurement.

The patch was generated against the selected pipeline at source snapshot `2f02b3f` (the pipeline itself retains the earlier selected behavior). SHA-256: `6210b587f99e53a3616da702dc30c747f365807d43145f58220e4bcbb8c2a86f`.

Run the nine source-only tests from the repository root:

```bash
uv run --offline pytest -q tools/recognition-recovery/workspace-focus-v1/test_focus.py
```

For an experiment, copy the source to an isolated checkout and apply the patch there:

```bash
git apply --check /absolute/path/to/workspace-focus.patch
git apply /absolute/path/to/workspace-focus.patch
```

`AUDITOR_FOCUS=off` preserves original observer requests. `AUDITOR_FOCUS=on` enables additive grounded detail observations. Do not change the primary model, frame policy, base flags or admission guards simultaneously. Run OFF parity before ON evaluation; reuse the approved shared environment without syncing or restarting the live model.

The generator retains its original workspace-root assumption (`/home/ibrak/datathon-kit`); edit only that path if regenerating elsewhere, then review the new diff/hash and rerun tests. Generated `*_candidate.py` files are intentional exact test inputs, not modules imported by the service. The private trial controller/config and data caches remain in the existing ignored workspace and are described in the handoff.

Preserve unknown/absence guards, primary and auxiliary failure attribution, original source timestamps and original API evidence. Qualification of a changed recipe requires fresh real uploads; a cached comparison cannot establish public API latency.
