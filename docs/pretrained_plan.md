# Local pretrained representation experiment

The human explicitly requested this next improvement after reviewing the P2 flow. No additional package approval is needed: the experiment uses the installed OpenCV DNN and the existing sklearn/LightGBM stack. No clips, frames, model-derived descriptions or evidence leave the computer.

1. Verify the official OpenCV Zoo MobileNetV2 weights and preserve their Apache 2.0 license. Pin byte hash and preprocessing. Confirm the actual 1280-dimensional pooled tensor using synthetic input before any footage extraction.
2. Extract a single new feature family on the same 32 development groups: local pretrained frame embeddings at 2 FPS, fixed seed-42 128-dimensional projection, 2-second temporal summaries and clip pooling. Projection is label-independent; learned scaling/imputation/classifiers stay inside folds. Exact model and helper-source fingerprints isolate caches/artifacts.
3. Run `uv run python -m src.train --task video --encoder mobilenet` into a fresh directory, compare the same fixed linear/LightGBM recipes to the recorded numeric baseline, and run `uv run python -m src.sanity --task video` with 20 shuffled-label trials. Do not relax suspicious-feature/score stop conditions. The eight holdout clips remain unopened.
4. Assess ROC-AUC, balanced accuracy, recall/false positives, candidate timestamp error, and CPU latency. Before observing results, a meaningful promotion target is development AUC at least 0.60 with balanced accuracy above 0.50, no recall regression from 10/17, and passing sanity. This is an engineering gate, not an official jury metric or proof of generalization. Disclose any timestamp regression separately rather than claiming detection gains fix localization.
5. Add a fixed-allowlist inference dispatcher that preserves legacy artifacts. Update the local service only for an audited improvement, verify offline operation and the existing Windows loopback bridge, and keep the original run as a rollback. If this representation fails the gate, record/drop it and investigate a bounded temporal use of the same local embeddings; never choose from holdout results.

The encoder is pretrained for general images. This experiment does not establish action categories, legal facts, normative grounding or multi-event coverage. Those limitations stay visible in the UI and report.

## Bounded temporal follow-up, declared after the first result

The first MobileNet run selected LightGBM with AUC 0.588235, balanced accuracy 0.535294 and recall 8/17. It fails the original promotion gate (AUC below 0.60 and recall below 10/17). Its unchanged sanity gates passed: 20 shuffled-label trials mean AUC 0.464510; no single-feature AUC above 0.90. Do not promote this model.

Evaluate exactly one follow-up: the existing regularized linear window head on the same cached descriptors, with video probability equal to the maximum window probability and candidate time at the same maximum. Fit within each existing outer training fold, using all negative windows and only timestamp-selected eligible positive windows; no validation annotation enters fitting. Keep threshold 0.5 and the original promotion gate. Do not try alternative pooling rules or tune a threshold against these results.

For matched shuffled-target checks, permute complete supervision tuples (target, localization eligibility, annotated fraction of duration) across feature bags. Map a donor's annotated fraction to the recipient duration; rebuild the window head in each training fold. This severs the original labels and timestamps together. The null must never reuse a true-label window head or true-label timestamp selection.

This follow-up is exploratory after seeing the first development result. Maximum-window scores are uncalibrated and can increase false alarms on longer videos. If it also fails, retain the original live baseline, preserve the pretrained candidates for review, and report that a useful improvement has not yet been demonstrated. The eight holdout recordings remain untouched.
