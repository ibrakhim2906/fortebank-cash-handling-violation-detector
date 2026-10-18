# Rule grounding for the final release

Checked 18 October 2026, 04:52 UTC+05:00. This is an engineering applicability checklist, not an enabled ruleset or a claim that these event types are recognized. Preserve the existing catalog and legal interpretation in `docs/action_rule_candidates.md` and `docs/case/legal.md`.

## Refreshed official sources

The official Adilet API returned the following texts. All three text hashes match the earlier catalog. Both current-version metadata and the substantive text were checked; `is_actual` alone is insufficient to establish that an act remains operative. Stored revision dates are not treated as legal commencement dates. The browser retrieval tool could not read these API URLs; direct HTTPS retrieval succeeded. One initial №231 retrieval failed with an SSL record error; the subsequent complete retrieval was saved and hashed.

| Source | API status / current flag | SHA-256 of exact `text_content` |
|---|---|---|
| [NBK №231/2019, registration 19680: cash operations](https://adilet.zan.kz/api/documents/by-ngr/V1900019680?language=rus) | `upd` / `true` | `1798a349b2326ba4c32a007a62db532ea6de906e71f0249a9de8dc81aa915de8` |
| [NBK №14/2020, registration 20075: cash premises](https://adilet.zan.kz/api/documents/by-ngr/V2000020075?language=rus) | `upd` / `true` | `89b91567941acfa3d4fca1d374f04b1716cdf20fab43bd4d50ca7074b17ee16d` |
| [ARRFR №53/2026, registration 38336: information security](https://adilet.zan.kz/api/documents/by-ngr/V2600038336?language=rus) | `new` / `true` | `8d3ffc326265de22e160bcee437c58d5aa954951ab7922db2cd596089b07898c` |

Private retrieval evidence: `data/processed/release-readiness-20261018/rule-source-refresh.json`, the three raw public API responses, and their extracted `*-clauses.json`. Re-extracting clauses required handling paragraph start tags as well as end tags: the HTML contains adjacent/nested paragraph boundaries. The saved raw response and full text hashes are the authoritative provenance, not an incomplete first extraction.

For №14 §16, preserve the temporary replacement wording applicable through 01.01.2027, rather than quoting only the underlying paragraph. For №53 §71(9), the stated inactivity condition is more than 30 minutes; §72 also specifies exceptions to §71. These sources do not prove that a short cashier departure violates a workstation-lock requirement. No categorical food, private-reading or key-left-in-any-lock prohibition was verified.

## Fourteen annotation types: what a source can and cannot establish

These are existing team annotation names. A label in this table is not evidence of recognition coverage or a proved legal finding. Clause references below are candidates subject to the visible facts and the source's full conditions.

| Type | Source context | Evidence required / reason to qualify or withhold |
|---|---|---|
| `cash_left_out` | №231 §20: storage of previously accepted cash | Distinguish stored cash from the current customer's active transaction; establish cash and custody. A stack or unknown bag alone is insufficient. |
| `safe_left_open` | №231 §§20, 80: storage and internal controls | Establish the storage device, relevant contents and unattended interval. An open device during legitimate use is insufficient. |
| `personal_items` | №231 §12: personal-property storage and exceptions | Establish personal ownership, storage and cash-unit boundary. Keep the permitted medicine/medical-product/communication-device exceptions. |
| `no_count_check` | №231 §19(2), §§24–25, §31(4), §32 | Establish acceptance/withdrawal/collection stage and continuous coverage. Preserve intact packaged-withdrawal and sealed collection-bag conditions. Machine absence is not counting absence. |
| `counting_without_client` | №231 §20, §31(4): client presence during applicable counting | Establish the customer role, operation and counting interval; an unseen customer could be outside the camera view. |
| `no_receipt` | №231 §19(6), §31(6): transaction confirmation | Establish completed service and the full opportunity to issue the confirmation. A missing visible paper handover does not establish that no confirmation was issued. |
| `no_id_check` | №231 §19(1), §31(1): identification, including digital-document data | Absence of a physical ID handover is insufficient. Video cannot establish matching identity or unseen electronic processing. |
| `hand_to_hand_transfer` | №14 §16(4): protected transfer-device design | Determine actual customer/cashier sides and bypass of the applicable device. An ordinary handover is an observation, not automatically a violation. |
| `door_left_open` | №14 §16(1), §19(1), §21: premises and access devices | Establish room type, boundary and applicable condition. A lockable-door requirement does not establish that every visible door must remain continuously shut. |
| `unauthorized_access` | №14 §16 and §21: protected premises/access context | Establish the protected boundary and a documented authorization condition. Appearance or an unknown person's presence cannot establish authorization status; do not identify faces. |
| `keys_left_in_safe` | №231 §80: controls specified by internal bank documents | The applicable internal control is missing. Report a visible key cautiously; do not invent a blanket key-removal rule. |
| `workstation_unlocked` | №53 §71(9), subject to §72 | Establish actual session state and the applicable inactivity interval. A lit display or short departure proves neither condition. |
| `food_drink` | No standalone food prohibition verified; conditional personal-storage context is separate | Describe visible handling. Do not cite §12 as an unconditional food ban or infer the contents of an unreadable cup. |
| `personal_activity` | No standalone private-reading prohibition verified; №53 §76 concerns protected paper information | Reading-like conduct is insufficient to establish document status, ownership, prohibited activity or an information-security breach. |

## Served explanations and pitch boundary

Describe visible conduct and source-video time first. State the supported concern and material uncertainty, then cite the applicable source with its actual scope. When the source is contextual only, label it as context; an unrelated citation must not make an alert appear grounded. Do not activate rules or change descriptions merely by importing this matrix.

Example structure, to be filled only with independently supported observations: “At [source time], [literal visible action]. This may warrant review because [qualified concern]. Relevant context: [document and paragraph]; [missing fact or exception].”

Avoid claims of theft, intent, personal identity, incorrect totals, forged currency, ledger errors or omitted electronic verification from unreadable/offscreen evidence. Public guidance from another bank may be attributed as that bank's practice; it must never be presented as ForteBank's internal policy.

Final release acceptance: every emitted type must have a reviewed explanation/source match, or an explicit source gap and appropriately qualified observation. Source completeness and recognition accuracy are separate checks.
