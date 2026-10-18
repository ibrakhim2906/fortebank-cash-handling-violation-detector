# Human annotations — what exactly is wrong in each video (from the team's labels file)

Source: team-annotated `open_labels.xlsx` (column 4), translated. Type keys refer to `docs/case/violations.md`.

## Violation videos
| Video | t (s) | Annotated violation (EN translation) | Type key(s) |
|---|---|---|---|
| X-1 | 22 | Cash movement without a confirming document: e.g. machine counted 50 000, client is given 10 000 change, but no receipt/document for the returned 10 000 | `no_receipt` |
| X-2 | 10 | KYC breach: the cashier starts the cash operation and accepts the payment **without first checking the client's identity document** | `no_id_check` |
| Х-4 | 57 | Outsiders systematically in the service (cash) room; no access control — **the cash-desk door is not locked between clients**, free passage; **transfer tray ignored** | `unauthorized_access`, `door_left_open`, `hand_to_hand_transfer` |
| Х-47 (= Х-4) | 68 | **Cash accepted without counting each banknote and checking authenticity/fitness** | `no_count_check` |
| Х-11 | 18 | Cashier left the workplace **without closing the door of the cash booth** → risk of unauthorised access | `door_left_open` |
| Х-13 | 47 | Valuables left unattended: cashier left the room, **cash left on the desk (not in the safe)**, **safe door left open** | `cash_left_out`, `safe_left_open` |
| Х-15 | 15 | Cashier left the workplace **leaving cash/valuables/collection bags outside the safe**; must put everything into the safe and lock it before leaving | `cash_left_out` |
| Х-17 | 7 | Access regime breach: **outsider admitted inside the cash unit**; collection bag handed **hand-to-hand, bypassing the transfer tray** | `unauthorized_access`, `hand_to_hand_transfer` |
| Х-18 | 20 | **Several outsiders at once in the cash room**; no door control; valuables/documents passed **hand-to-hand** | `unauthorized_access`, `hand_to_hand_transfer` |
| Х-20 | 5 | **Safe left open while working** (must be locked right after taking out valuables); handling open cash on the desk with the safe open | `safe_left_open` |
| Х-22 | 14 | **Keys left in the safe lock** unattended; after locking, the cashier must remove the keys and keep them — especially when leaving the booth | `keys_left_in_safe` |
| Х-24 | 8 | **Personal activities** at the workplace (reading personal literature / entertainment); also **leaves the room without locking the booth door** | `personal_activity`, `door_left_open` |
| Х-26 | 9 | **Cash left outside the safe** without staff control; before any exit the cashier must put all cash into the safe and lock it | `cash_left_out` |
| Х-28 | 7 | Information security: leaving the workplace **without locking the computer** (Win+L or closing the lid) | `workstation_unlocked` |
| Х-30 | 13 | **Cash-room door left ajar** | `door_left_open` |
| Х-32 | 5 | **Unauthorised access to the cashier's workstation and cash-handling equipment** | `unauthorized_access` |
| Х-33 | 6 | **Unauthorised access to the cash-handling room** | `unauthorized_access` |
| Х-34 | 13 | **Counting incoming cash in the customer's absence** | `counting_without_client` |
| Х-38 | 33 | **Previously accepted cash stored outside designated lockable storage** (left on the desk) | `cash_left_out` |
| Х-40 | 17 | **Personal belongings placed in the cash-handling area** | `personal_items` |
| Х-44 | 16 | **Previously accepted cash stored outside designated lockable storage** | `cash_left_out` |
| Х-46 | 11 | **Drinking a beverage at the workstation while handling cash** | `food_drink` |

## Clean videos (no notes in the file) — inferred correct behaviour
The violations define the expected protocol, so the clean look-alikes should show it (confirm by watching):
- Leaving the booth (Х-25, Х-27, Х-29): cash put into the safe, safe locked, **keys removed**, **laptop locked/closed**, **door closed** behind.
- Serving a client (Х-3, Х-5, Х-35): client stays on the client side, **ID document checked first**, cash passed **via the transfer tray**, **counted note by note (machine)** in the client's presence, **document/receipt handed** to the client, door closed.
- Cashier alone (Х-8, Х-10, Х-16, Х-23 …): cash kept in the safe/drawer, no drinks, no personal items, no reading.
