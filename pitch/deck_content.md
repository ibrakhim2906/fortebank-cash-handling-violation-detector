# CashLint — pitch deck content (Team Seltik)

Live deck (Slides artifact, private until shared): https://claude.ai/artifact/LwnP6P6VqrWGmiEiahS9HU

7 slides in Russian, Montserrat, grey background, black text, bullet points, no icons. Focus: what CashLint does and its advantages.
Team: Ибрахим Копжасар, Дмитрий Куттыбаев, Дидар Баймуратов.

Opening lines (in the cover slide’s speaker notes): «После подачи мы улучшили систему: 28 из 40 вместо 27 и проверили другой ракурс — сохраняется около 80% точности. Всё, что вы видите, — результаты реальных прогонов».

## Slides

| # | Slide | Content |
|---|---|---|
| 1 | Обложка | CashLint — автоматическая проверка кассовых процедур по видео. Команда Seltik и состав команды |
| 2 | Задача и решение | Задача кейса (тип, момент, объяснение, кадры, API до 5 минут) и ответ CashLint (14 типов, таймкод, 1–3 кадра, норма НБРК, точная цитата НБРК №231, REST API) |
| 3 | Что находит CashLint | 14 типов нарушений в 4 группах: хранение и безопасность (5), доступ и передача (2), этапы операции (4), правила рабочего места (3) |
| 4 | Как это работает | Загрузка → обзорные кадры с исходным временем → анализ (Qwen3-VL-32B, одна A40) → перепроверка на плотных кадрах и повторный проход → отчёт |
| 5 | Пример результата | Нарушение · наличные оставлены вне сейфа · 44,0 с · кадры 44,0 и 45,5 с · НБРК №231, п.20 с цитатой; без юридических утверждений там, где прямой нормы нет |
| 6 | Результаты | 28/40, −50% ложных срабатываний (6 → 3), 42/42 задачи API, ~80% точности на другом ракурсе; версии на 30 видео: 12 → 16 → 20 → 23 → 24 |
| 7 | Преимущества | Конфиденциально, объяснимо, быстро, надёжно (1 234 автотеста), проверяемо, ответственно; следующий шаг; спасибо за внимание |

Slide 5 shows an example of the output format (fields and timing rule from `service/pipeline/analyze.py::build_result`).

## Where every number comes from

| Figure | Source |
|---|---|
| 14 violation types, norm references | `service/rules.json` |
| Model and GPU | `deploy/mission-best.env`, `docs/REVIEW_CONTEXT.md` |
| 30-video development set: 12/16/20/23 correct; false alarms 6 → 3 | `service/static/evaluation.json`, `experiments.md` |
| 42/42 jobs, 0 failed; 72.98 s slowest | `EXPORT_MANIFEST.json` → `actual_api_qualification` |
| 28/40 on all 40 videos; 24/30 with the second pass; mirror 19/30 and 10° rotation 20/30; 1,234 tests; НБРК №231 quote; dense-frame re-check and second pass | Team-reported post-submission runs; not recorded in this repo snapshot (the snapshot has 27/40, 23/30 and 1,196 tests) |

## Q&A backup

- All 40 given videos were seen during development, so these are not independent test scores (also stated on slide 6).
- Next steps on slide 7 (more precise timecodes, fine-tuning on the new camera angle) are plans, not measured results.
