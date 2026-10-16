# Case context

> Codex drafts this from `docs/brief.md` + the data in hour 1. A human confirms every line.
> Mark anything uncertain with ❓ and add it to "Open questions".

## The business decision
- Who at ForteBank uses the output?
- What decision does it support (approve / reject, limit, offer, contact, flag)?
- How often, and at what scale?

## Prediction unit
One row = ? (client / application / transaction / client-month)

## Target
- Column:
- Meaning of each value:
- Observation window (e.g. default within 6 months after issue):

## Metric
- Official metric:
- Also tracking (business-facing):

## Decision moment
- At what moment would the bank make this prediction?
- Which data exists at that moment? Which columns would NOT exist yet (leakage risk)?

## Data
| Table | Rows | Grain | Key(s) | Time range | Notes |
|---|---|---|---|---|---|

## Validation
- Split type (time / group by client / stratified) and why:

## Deliverables and format
- (submission file? notebook? repo? demo? slides? deadline?)

## Rules and constraints
- AI tools / prepared code allowed?
- External data allowed?
- Pitch length, Q&A length:

## Open questions for organizers / mentors
-
