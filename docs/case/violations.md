# Violation catalogue (from team annotations — `docs/annotations.md`)

21 violation videos, 23 annotated violations, 14 types. The annotations reveal the **expected protocol**;
a violation = a visible deviation from it. This is exactly what the Facts → Rules engine encodes.

## 1. The expected protocol (what clean videos should show)
- **Leaving the booth:** cash into the safe → safe locked → keys removed → laptop locked/closed → door closed behind.
- **Serving a client:** client stays on the client side → ID document checked first → cash passed via the transfer tray/device → counted note by note (banknote counter) in the client's presence → confirming document/receipt handed over → door kept closed.
- **Always:** no outsiders inside; no personal items, drinks or reading at the cash desk; cash never lying on the desk outside an operation.

## 2. Types, cues, detection tier
Tier 1 = objects/people the detector sees directly · Tier 2 = state change vs. a per-video reference (door, safe, desk) · Tier 3 = procedure sequence (what *didn't* happen).

| Key | Name (RU, for the jury) | Videos | Visible cue | Tier | Legal |
|---|---|---|---|---|---|
| `cash_left_out` | Наличные деньги оставлены вне сейфа (хранилища) | Х-13, Х-15, Х-26, Х-38, Х-44 | banknotes/bag lying on the desk while the cashier leaves or outside an active operation | 2 | №231 п. 20; п. 22 if leaving during service |
| `door_left_open` | Дверь кассы оставлена открытой / незапертой | Х-11, Х-24, Х-30, Х-4 | door region stays open after someone passes / after the cashier leaves | 2 | №14 п. 16 пп. 1 |
| `unauthorized_access` | Доступ постороннего лица в кассовое помещение / к рабочему месту кассира | Х-4, Х-17, Х-18, Х-32, Х-33 | person other than the cashier inside the booth / at the cashier's side or equipment; 3+ people | 1 | №14 п. 16 пп. 1, п. 8; №231 п. 75 (safe room) |
| `hand_to_hand_transfer` | Передача ценностей из рук в руки, минуя передаточное устройство | Х-17, Х-18, Х-4 | wrists of cashier and another person meet over the desk; bag/cash passes hand-to-hand | 2 (pose) | №14 п. 16 пп. 4 |
| `safe_left_open` | Сейф оставлен открытым | Х-20, Х-13 | safe door open while cashier is away from it / works at the desk | 2 | №231 п. 20 («закрывающихся на ключ») |
| `keys_left_in_safe` | Ключи оставлены в замке сейфа | Х-22 | key visible in the safe lock when the cashier leaves | 3 (tiny object) | №231 п. 20 (interpretation) |
| `workstation_unlocked` | Рабочий компьютер не заблокирован при уходе | Х-28 | laptop open, screen lit while the cashier leaves | 1–2 | **no verified RK norm yet** — legal teammate: НБРК requirements on information security |
| `counting_without_client` | Пересчёт принятых наличных в отсутствие клиента | Х-34 | client leaves, then cashier counts/handles the cash (banknote counter active) | 3 | №231 п. 20 («пересчитываются … в присутствии клиента») |
| `no_count_check` | Приём наличных без полистного пересчёта и проверки подлинности | Х-47 (68 s) | cash taken and put away without passing through the banknote counter | 3 | №231 п. 19 пп. 2; п. 57 |
| `no_id_check` | Приём наличных без идентификации клиента по документу | X-2 | money accepted without any ID document being shown/handed first | 3 | №231 п. 19 пп. 1; п. 7 |
| `no_receipt` | Выдача наличных (сдачи) без документа, подтверждающего операцию | X-1 | cash handed to the client, no receipt/document handed | 3 | №231 п. 31 пп. 6 (п. 19 пп. 6 for acceptance) |
| `personal_items` | Личные вещи в кассовой зоне | Х-40 | bag/coat placed on the cash desk | 1 | №231 п. 12 |
| `food_drink` | Употребление напитков на рабочем месте при работе с наличными | Х-46 | cup/bottle in the cashier's hand / at the face, cash on the desk | 1 | no explicit norm found; «по смыслу п. 12 №231» (personal items in the cash unit) — say it's an interpretation |
| `personal_activity` | Посторонние занятия на рабочем месте (чтение) | Х-24 | book/magazine in the cashier's hands | 1 | no public norm found → report Х-24 via `door_left_open`; show this one only with a «внутренний регламент» caveat or not at all |

## 3. What this means for the verdict (criterion 1 = 25 pts, scored per video)
One correctly detected violation per video gives the verdict. Videos by the easiest available cue:
- **Tier 1 available:** Х-17, Х-18, Х-4 (outsiders), Х-40 (bag), Х-46 (cup), Х-24 (book), Х-28 (open laptop on leave).
- **Tier 2 needed:** Х-11, Х-30 (door), Х-13, Х-15, Х-26, Х-38, Х-44 (cash on desk), Х-20 (safe open).
- **Tier 3 only (hardest):** X-1 (no receipt), X-2 (no ID check), Х-22 (keys), Х-32, Х-33 (who is "unauthorised" without faces?), Х-34 (counting after client left).
Priority order: Tier 1 → cash on desk → door → safe → hand-to-hand → Tier 3. Tier 3 types are where most teams will fail; even 2–3 of them caught is a differentiator.

## 4. Hard questions and suggested approaches
In GPU mode the **Protocol Auditor** (`auditor.md`) is the main detector for Tier 3 and the second opinion for Tier 2; the approaches below are the CPU-mode fallbacks and the inputs (episodes, crops) the auditor needs.
- **Cash on the desk:** COCO has no "money". Options: (a) open-vocabulary detector OWLv2 / Grounding DINO (Apache-2.0) with prompts «banknotes», «stack of money», «cash», run only on keyframes; (b) desk-region change vs. the empty-desk reference + colour/texture of banknotes; (c) few-shot crop classifier (OpenCLIP features + logistic regression) trained on desk crops labelled from these videos.
- **Door state:** door = where people enter/exit (learned per video from track start/end points). State = difference of the door region from its "closed" reference (median of frames with no motion there / first frames). Rule: door differs from closed reference for ≥ N s after the last person passed.
- **Who is "unauthorised" (Х-32, Х-33) without face recognition?** Use role, not identity: the cashier = the person seated behind the desk at the start, or the first person who opens the safe/uses the counter; anyone else on the cashier side, at the equipment, or entering the booth = outsider. If the room starts empty, the first entrant's behaviour matters (goes to the client chair = client; goes to equipment without sitting/working protocol = suspicious). Clothing-colour histograms per track are allowed (not biometrics) to keep roles consistent within a video. Compare Х-31 (clean) with Х-32 frame by frame to find the difference.
- **Procedure checks (X-1, X-2, Х-47, Х-34):** segment a *transaction episode* (client arrives on client side → leaves). Within it, look for: document handed before cash (small flat object passing to the cashier / OWLv2 «passport», «ID card»), banknote-counter activity (motion in the counter box while cash is present), receipt handed back (OWLv2 «receipt», paper slip). Missing step → violation at the moment the step was due (cash accepted / cash handed out / client left). Only enable a check after it stays silent on the clean client videos Х-3, Х-5, Х-35.
- **Keys in the lock (Х-22):** probably invisible at 464×848; try OWLv2 «key in lock» on the safe crop at the moment the cashier leaves the safe; if unreliable, leave it — don't guess.

## 5. Timecode policy
Report the onset of the wrongful act or the moment the duty was due: leaving without locking → the moment the cashier walks away; outsider → entry moment; hand-to-hand → moment of the handover; drink/book/bag → first clear frame; procedure gaps → the moment the missing step was due. Labels confirm onset semantics (e.g. Х-22 = 14 s when the cashier walks to the door).
