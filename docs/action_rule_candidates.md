# Public rule candidates for human source review

Research date: 17 October 2026; successful official-API retrieval checked at 08:58 UTC. This is a source catalog, separate from action labels, detector features and individual clips. No footage, frames, clip descriptions or model proposals were inspected for this research. **The public clause text is now verified; neither candidate is approved for automatic video-to-rule grounding.** The eight model action primitives remain observations, not violation classes.

The brief permits public Kazakhstan rules and accepts source references without a precise clause (FAQ 30–31). That acceptance does not establish a rule's applicability to a particular video.

## Verified source and scope

The [canonical Adilet cash-operations act](https://adilet.zan.kz/rus/docs/V1900019680) initially returned a JavaScript shell; the older page and comparison view timed out. Reading the public site's own frontend established its unauthenticated act-reading API. The following official endpoints returned complete text/metadata successfully; no login or private session was used:

- [Current №231 act and full text, resolved by its public identifier](https://adilet.zan.kz/api/documents/by-ngr/V1900019680?language=rus).
- [Specific current №231 document](https://adilet.zan.kz/api/documents/c4cd4052-63e3-475b-ba9e-93f23de13fdb).
- [Official revision history](https://adilet.zan.kz/api/documents/c4cd4052-63e3-475b-ba9e-93f23de13fdb/versions?page_size=100).
- [Amending resolution №71 of 10 November 2025, full text and publication reference](https://adilet.zan.kz/api/documents/by-ngr/V2500037429?language=rus).

The source is NBK Board resolution №231 of 29 November 2019, registered №19680. Paragraph 1 explicitly covers second-tier banks, nonresident-bank branches, the national postal operator and specified collection organizations. The commercial-bank category is relevant to this case; these are not rules limited to the National Bank's own cash offices. No person or institutional role is inferred from footage.

On retrieval, the API marked document `c4cd4052-63e3-475b-ba9e-93f23de13fdb` as `is_actual=true`, status `upd`; the history marked this same document as the sole current revision among 19 records. Its exact stored revision timestamp is `2026-07-11T18:00:00.000Z`. The text includes amendments expressly effective on 12 July 2026. Preserve that distinction rather than converting the stored timestamp into a legal commencement date. This verifies the official current version served on **17 October 2026**, rather than an unchanged 2019 or search-indexed copy.

The amendment's human-readable official publication reference states the electronic reference bank publication date **20 November 2025**. Its commencement clause uses ten calendar days after first official publication, with specified provisions deferred to 1 January 2026. Both are before the case date; the current consolidated paragraphs below retain the amendment. Raw API date-storage fields differ from the human-readable publication reference and are not used to infer publication or commencement.

Retrieval provenance: SHA-256 of the exact UTF-8 `text_content` field was `1798a349b2326ba4c32a007a62db532ea6de906e71f0249a9de8dc81aa915de8` for the current act and `df90633b6503c249f77d8d74a3c28e40486d6811a66d7043294c4703797820b5` for amendment №71. Text was read directly from the official API; full act copies were not added to the dataset. The summaries below are paraphrases, not long quotations.

## Candidate 1 — Employee personal belongings in the cash unit

- **Candidate ID:** `RK_NBK231_P12_PERSONAL_STORAGE`.
- **Exact section verified:** paragraph 12, read in the [current consolidated text](https://adilet.zan.kz/api/documents/by-ngr/V1900019680?language=rus) and independently in the [amending act](https://adilet.zan.kz/api/documents/by-ngr/V2500037429?language=rus).
- **Limited source proposition:** employee personal belongings, personal cash and valuables are stored outside the cash unit, subject to the bank's internal procedure. The paragraph also permits medicines, medical products and communication devices without photo/video recording functions.
- **Version status:** source wording verified in the official current revision served on 17 October 2026. The paragraph's own note attributes its wording to amendment №71 of 10 November 2025. Rule-to-video applicability remains unverified.
- **Visible evidence needed before even proposing a rule association:** a reviewer must establish the cash-unit boundary and that an item is a personal belonging being stored there. The camera must show the relevant item and its placement over a sufficient interval.
- **Excluded inferences:** an object moving toward a bag does not establish ownership, storage, theft or cash diversion. A handheld object is not necessarily a personal phone; recording capability, permitted-item status and any internal procedure cannot be inferred from appearance alone.
- **Activation requirement:** the human must confirm premises/item context and a narrow observable predicate, including the stated exceptions. The source text alone does not validate a detector's interpretation. Until then, retain only an ungrounded observable-action proposal.

## Candidate 2 — Counting and checking cash when accepting it

- **Candidate ID:** `RK_NBK231_P19_2_CASH_ACCEPTANCE`.
- **Exact section verified:** paragraph 19, subparagraph 2, read in the [current consolidated text](https://adilet.zan.kz/api/documents/by-ngr/V1900019680?language=rus) and the [2025 amending act](https://adilet.zan.kz/api/documents/by-ngr/V2500037429?language=rus).
- **Limited source proposition:** subparagraph 2 describes accepting banknotes with note-by-note counting, coins individually, and checking authenticity/payment fitness with reference to paragraphs 57 and 58. Both referenced paragraphs were read: paragraph 57 refers to rules №230 for payment fitness; paragraph 58 describes official issuer/reference materials and rules №49 for foreign-currency payment fitness. The external acts №230 and №49 were not independently reviewed here.
- **Version status:** source wording verified in the official current revision served on 17 October 2026. Paragraph 19's note records the 2022 replacement and 2025 amendment. This does not establish compliance, an exhaustive interpretation of linked acts, or a video event type.
- **Visible evidence needed before even proposing a rule association:** establish that the transaction is cash acceptance and that the objects are cash; observe the relevant counting interval continuously, including any counting machine and cash path. Human review must distinguish counting from another sorting/stacking activity.
- **Excluded inferences:** sparse frames cannot prove counting was omitted. Repeated restacking does not establish a wrong total. Machine operation or a hand movement does not prove an authenticity check, currency value, counterfeit status or payment fitness. Offscreen/earlier checks and transaction stage remain unknown.
- **Activation requirement:** define a narrow observable predicate with a human; verify the linked external requirements before asserting an authenticity/payment-fitness rule. No automatic absence-based violation or type assignment is enabled.

## Integration boundary

Do not insert either candidate into current API descriptions as if it substantiates the returned event. The public source check is complete for the numbered clauses above; a valid future association still needs an independently reviewed observable predicate and matching clip evidence. A model's generic handoff, contact, relocation or restacking observation does not establish that association. The catalog creates no verified action labels, evaluates no footage and changes no model or decision threshold.

## Transaction context rechecked after the direct pretrained probe

On 17 October 2026 the current official API again returned `is_actual=true` and the same text checksum recorded above. The human specifically asks to reason about document/ID presentation, cash provision, cash counting and document completion. These are useful workflow stages; a visible stage is not proof that its procedural requirement was satisfied.

The current [official resolution №231](https://adilet.zan.kz/api/documents/by-ngr/V1900019680?language=rus) provides this context for covered second-tier banks:

| Operation | Public requirements relevant to context | What video can support |
|---|---|---|
| Cash acceptance | Incoming document (§17); identification, counting/authenticity checks, amount reconciliation and confirmations (§19); acceptance/counting in the client's presence (§20) | Document presentation/handoffs, direction of cash movement, visible counting or machine interaction, customer presence, document writing/stamping/return |
| Cash withdrawal | Outgoing document (§29); identification/signature checks, preparation/counting, payment and confirmation (§31) | Cash preparation/counting, outward handover and document return, with the transaction stage established |
| Cash storage during acceptance | Previously accepted cash is stored in key-lockable devices (§20) | Cash moved from the work surface into a storage device; ownership and actual lock state may remain unknown |
| Workstation presence | Cashier does not leave before completing customer service during acceptance (§22) | Departure across an established workstation boundary while service is demonstrably unfinished |
| Cash transfers between cash workers | Accounting documentation accompanies the process (§15) | The transfer itself; absent visible paper does not prove accounting records are missing |

**Withdrawal exception:** §32 permits payment using the stated amounts of intact packaged banknotes without note-by-note counting. Do not apply a blanket absent-counting rule to all cash payouts. Paragraph 12's personal-belonging restrictions likewise require the actual cash-unit boundary, item context and exceptions; phone appearance alone does not prove a violation.

Use a flexible stage model, not a universal four-step sequence. Documents can be prepared before cash is accepted, and the video may not reveal electronic processing. The observable event vocabulary should distinguish document presentation, document handover, cash handover/direction, manual counting, counting-machine interaction, writing/stamping, receipt/document return, cash storage and workstation presence. These are proposed observations, not yet trained or validated classes. No rule is activated by this contextual research.

Camera evidence alone generally cannot establish identity/signature matching, correct account posting, ownership, theft, exact amount correctness or banknote authenticity. No requirement to count with both hands was established. Verify the applicable transaction's full conditions before any omission-based finding.

## Additional sources and limits checked during all-supplied-video review

Checked 17 October 2026. These source checks do not turn any proposed visual annotation into a verified finding.

- **Collection bags:** current №231 §24 requires accompanying-document information, registration, bag integrity and sealing checks for the covered collection acceptance. §25 requires opening/recount in the collector's presence when bag integrity is compromised. Do not impose immediate note-by-note recount on every intact sealed collection bag or infer missing electronic records from absent visible paper.
- **Cash-room premises:** current [NBK resolution №14 of 24 February 2020](https://adilet.zan.kz/api/documents/by-ngr/V2000020075?language=rus), registration №20075, API status `upd`, `is_actual=true`, exact text SHA-256 `89b91567941acfa3d4fca1d374f04b1716cdf20fab43bd4d50ca7074b17ee16d`. §16(1) specifies a separate entrance with a lockable door for each cash office; §19 addresses the cash-counting room; §21 concerns the collection box/loading area and third-party access/viewing. These premises requirements do not by themselves say that every pictured glass office door must be closed continuously. Check the operative transitional wording and actual room scope before grounding an event. Old №250/2012 (`V1200008080`) is repealed, even though its API marks the latest stored revision `is_actual=true`; status/repeal text must also be checked.
- **Documents/clean desk:** [Halyk's public employee guidance](https://halykbank.kz/en/for-employees) instructs its employees to remove documents from the desk and lock them in a drawer when leaving. This undated, currently accessible page is attributable Kazakh-bank practice accepted as a type of public source by FAQ31, not proof of ForteBank's policy or a universal statutory obligation. A bound notebook's actual status/content remains uncertain.
- **Information security:** current [ARRFR resolution №53 of 3 April 2026](https://adilet.zan.kz/api/documents/by-ngr/V2600038336?language=rus), registration №38336, API status `new`, `is_actual=true`, exact text SHA-256 `8d3ffc326265de22e160bcee437c58d5aa954951ab7922db2cd596089b07898c`. §71 includes an automatic session-end/workstation-lock condition after more than 30 minutes of inactivity; it does not substantiate a violation immediately after a short departure. §76 addresses organizational/technical protection of paper information. An illuminated or image-like display is not proof of an authenticated, unlocked session.
- **Keys:** current №231 §80 delegates relevant key/stamp/storage-control arrangements to internal bank documents that prevent unauthorized use. No current categorical prohibition on a key remaining in any cabinet lock was verified. Do not reuse that categorical language from repealed 2001 rules (`V010001482_`, old §81).
- **Food/drink:** no explicit standalone eating/drinking prohibition was verified in №231. Record visible cup/snack handling separately from a conditional personal-item storage concern under §12; do not fabricate a food-specific clause.

The resulting catalogue must distinguish clear actions from unresolved legal conditions. A supplied positive verdict/anchor can guide interpretation, but does not establish a particular event type, an object's contents or the source's applicability.
