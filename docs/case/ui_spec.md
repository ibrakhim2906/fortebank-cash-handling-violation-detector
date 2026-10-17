# UI spec (R1–R11) — product readiness = 10 points

One static page `service/static/index.html` (vanilla HTML/JS, no build step), Russian, served at `/`.

## Flow
1. Header: «Проверка кассовой дисциплины по видео» + one line: what it does, that models run locally.
2. Upload area: drag-and-drop + click to choose; shows file name and size; accepts .mov/.mp4.
3. Button «Проверить» → POST `/ui/analyze` → poll `/ui/jobs/{id}` every 2 s.
4. Status line: «Загрузка видео…» → «Идёт анализ… N с» → «Готово за N с» / «Ошибка: …».
5. Verdict banner, exact wording (R4):
   - red: **«Обнаружены признаки возможных нарушений»**
   - green: **«Нарушение не обнаружено»**
6. For each violation, a card: type (title) · timecode as `m:ss.s` and seconds · 1–3 evidence frames (≈300 px high, caption with time) · description · rule block(s): act, point, quote.
7. Footer disclaimer (R56): «Результат — подсказка для проверяющего, а не окончательное заключение. Лица не распознаются, звук не используется.»

## Quality bar
- Works on a laptop screen and a phone (frames wrap). No external CDNs/fonts (portability R51).
- Clicking a frame opens it full-size (simple overlay) — nice to have.
- Never freezes: errors shown in the status line, button re-enabled.
- Escape all text inserted into HTML.
