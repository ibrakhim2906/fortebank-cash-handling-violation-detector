# Dataset facts (verified by running code on the files)

- Location: `data/raw/open/videos/*.mov` and `data/raw/open/open_labels.xlsx` (sheet `labels`, columns: `наименование файла`, `таймстемп`, `нарушение / без нарушения`).
- **42 label rows → 40 distinct videos.** Frame-identical pairs: **Х-4 == Х-47** (both violation: 57 s and 68 s → one video, two events) and **Х-5 == Х-48** (both clean). Merge them in eval, or the duplicates inflate scores.
- 22 violation rows (21 distinct videos, 22 events) / 20 clean rows (19 distinct).
- Video: H.264 .mov, **464×848 portrait**, 30 fps, no audio, 10–145 s (violation videos median ~21 s; clean median ~29 s; Х-14 = 142.5 s is the longest).
- File names: `X-1.mov`, `X-2.mov` (Latin X) and `#U0425-<n>.mov` = Cyrillic «Х-<n>» after unzip. Labels use Cyrillic «Х» except X-1, X-2. Map by the number.
- Timestamps: integers like «22 секунда», one as «1минута 8 секунда» (= 68 s). Parse both.
- Labels mark **the onset of the wrongful action** (e.g. the second the cashier walks away), not when its consequence becomes visible.
- Cameras: at least 3 setups already — (a) desk centre-left, safe right, glass door back-left (most videos); (b) tighter view (Х-30–Х-33); (c) different room with a blue client chair (Х-33, Х-40, Х-44, Х-46). Test adds another angle.
- Lighting: Х-11 and Х-12 turn dark/purple near the end (lights switched off) — not an event by itself.
- Background people behind the glass wall are visible and get detected as `person` (small boxes, conf 0.3–0.5).

## Label table (seconds; V = violation)
| Video | Label | Dur | Video | Label | Dur |
|---|---|---|---|---|---|
| X-1 | V 22 | 25.9 | Х-26 | V 9 | 19.8 |
| X-2 | V 10 | 17.1 | Х-27 | – | 26.4 |
| Х-3 | – | 20.0 | Х-28 | V 7 | 16.5 |
| Х-4 (=Х-47) | V 57 (+68) | 72.7 | Х-29 | – | 17.3 |
| Х-5 (=Х-48) | – | 71.6 | Х-30 | V 13 | 18.8 |
| Х-8 | – | 38.0 | Х-31 | – | 23.8 |
| Х-10 | – | 15.6 | Х-32 | V 5 | 21.3 |
| Х-11 | V 18 | 25.6 | Х-33 | V 6 | 14.3 |
| Х-12 | – | 42.3 | Х-34 | V 13 | 17.7 |
| Х-13 | V 47 | 65.3 | Х-35 | – | 56.8 |
| Х-14 | – | 142.5 | Х-37 | – | 15.9 |
| Х-15 | V 15 | 20.9 | Х-38 | V 33 | 59.0 |
| Х-16 | – | 22.7 | Х-39 | – | 78.7 |
| Х-17 | V 7 | 14.7 | Х-40 | V 17 | 25.3 |
| Х-18 | V 20 | 22.6 | Х-43 | – | 23.4 |
| Х-19 | – | 57.4 | Х-44 | V 16 | 20.4 |
| Х-20 | V 5 | 11.6 | Х-45 | – | 33.3 |
| Х-21 | – | 19.6 | Х-46 | V 11 | 19.1 |
| Х-22 | V 14 | 21.0 | Х-47 (=Х-4) | V 68 | 72.7 |
| Х-23 | – | 18.3 | Х-48 (=Х-5) | – | 71.6 |
| Х-24 | V 8 | 14.7 | | | |
| Х-25 | – | 31.7 | | | |
