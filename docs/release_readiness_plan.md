# Release audit and jury handoff ownership

18 October 2026. The human assigns this chat the delivery work outside the active frontend and recognition chats. This is a bounded audit/documentation task; implementation changes to the shared service or recognition pipeline are outside this pass.

1. Add a reusable public-surface audit: health, UI availability, and missing/wrong API-key rejection. Optionally verify a private key using an unknown job lookup. No video uploads, model calls, process signals, restarts, deployments or paid allocations.
2. Recheck the public normative sources and preserve retrieval metadata privately. Produce a source-to-event applicability matrix without enabling any rule or changing predictions.
3. Write jury handoff and offline-proof instructions with explicit outstanding gates, evidence locations, exact commands, and ownership. Distinguish recorded results from freshly measured checks.
4. Run focused tests for the audit and a live public audit. Report observed results and blockers, without treating health/auth checks as recognition, browser, or GPU-offline qualification.

Owned files: `scripts/audit_release_surface.py`, `tests/test_release_surface.py`, this file, `docs/jury_handoff.md`, `docs/release_rule_matrix.md`, `reports/release_readiness.md`. Machine-readable observations go under ignored `data/processed/release-readiness-20261018/`.

Frontend owns `service/static/`, `styles/` and its serving integration. Recognition owns `service/pipeline/`, model/evaluation scripts, GPU experiments and their current reports. Preserve all pre-existing edits and frozen evaluators. Do not run the irreversible GPU network guard while recognition uses the resident model. Do not publish keys, private footage, job evidence or license inventories.

Completion: executable audit and useful handoff documents exist; focused tests pass; a timestamped live audit and source-check evidence are saved; remaining GPU/offline, browser, final recipe, license-policy and organizer handoff decisions are explicit. Completion of this bounded task does not certify the final submission.
