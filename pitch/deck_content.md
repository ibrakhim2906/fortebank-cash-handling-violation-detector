# CodeLint — pitch deck content (Team Seltik)

Live deck (Slides artifact, private until shared): https://claude.ai/artifact/LwnP6P6VqrWGmiEiahS9HU

7 slides in Russian, Montserrat, grey background, black text, bullet points, no icons. Focus: what CodeLint does and its advantages.
Team: Ибрахим Копжасар, Дмитрий Куттыбаев, Дидар Баймуратов.

## Slides

| # | Slide | Content |
|---|---|---|
| 1 | Обложка | CodeLint — автоматическая проверка кассовых процедур по видео. Команда Seltik и состав команды |
| 2 | Задача и решение | Задача кейса (тип, момент, объяснение, кадры, API до 5 минут) и ответ CodeLint (14 типов, таймкод, 1–3 кадра, норма НБРК, REST API) |
| 3 | Что находит CodeLint | 14 типов нарушений в 4 группах: хранение и безопасность (5), доступ и передача (2), этапы операции (4), правила рабочего места (3) |
| 4 | Как это работает | Загрузка → кадры 2 в секунду → анализ (Qwen3-VL-32B, одна A40) → повторная проверка → отчёт; без внешних ИИ-сервисов; обычная работа не считается нарушением |
| 5 | Пример результата | Нарушение · наличные оставлены вне сейфа · 44,0 с · кадры 44,0 и 45,5 с · НБРК №231, п.20 |
| 6 | Результаты | 27/40 верных вердиктов, −50% ложных срабатываний (6 → 3), 42/42 задачи API, 73 с максимум; версии на 30 видео: 12 → 16 → 20 → 23 |
| 7 | Преимущества | Конфиденциально, объяснимо, быстро, надёжно, проверяемо, ответственно; готовы к пилоту; спасибо за внимание |

Slide 5 shows an example of the real output format (fields, timing rule and norm link from `service/pipeline/analyze.py::build_result`).

## Where every number comes from

All figures are copied from results already recorded in the repo; nothing was re-run for the deck.

| Figure | Source |
|---|---|
| 14 violation types, norm references | `service/rules.json` |
| 2 fps sampling, 16 frames, model and GPU | `service/pipeline/frames.py`, `deploy/mission-best.env`, `docs/REVIEW_CONTEXT.md` |
| 30-video development set: 12/16/20/23 correct; 12 of 16 caught; false alarms 6 → 3 | `service/static/evaluation.json`, `experiments.md` |
| 40 cases: 27/40, TP13 FN8 FP5 TN14, 6/22 within ±2 s | `README.md`, `EXPORT_MANIFEST.json` → `actual_api_qualification` |
| 42/42 jobs, 0 failed; 72.98 s slowest | `EXPORT_MANIFEST.json` → `actual_api_qualification` |
| 1,196 tests passed | `EXPORT_MANIFEST.json` → `public_export_test_result` |

## Q&A backup

- The version comparison and the −50% false-alarm figure are measured on the 30-video development set; all 40 given videos were seen during development, so these are not an independent test score.
- Planned improvements (higher-resolution frames, native video input, more training work) are not yet measured.
