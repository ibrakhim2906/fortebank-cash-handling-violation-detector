# Predeclared action-focused detector trial, after local runtime approval

This recipe and its promotion gate were fixed before detector scores were produced. The human approved the runtime/model and the bounded pilot succeeded operationally. The experiment is now complete and rejected: selected AUC 0.500000, all videos flagged, identical feature vectors across the 32 development recordings. Preserve the original plan below; actual results and decoder-recovery provenance are in `reports/qwen_action_detection.md`. No threshold tuning or wider model search followed the failed gate.

There are two different uses of local vision output:

1. **Action annotations:** proposals near organizer timestamps can help a human identify action labels. They stay unverified until independently reviewed. Do not train/evaluate an action classifier against its own model-generated labels.
2. **Video detector features:** fixed pretrained vision observations can be used as input features and evaluated against the independent, original binary labels. This does not require pretending that proposed action types are gold truth. It also does not establish action-type accuracy or legal grounding.

For the second use, extraction must never select intervals from organizer timestamps, video targets, human review status or validation predictions. Every clip uses the same label-independent selection rule and the frozen prompt/vocabulary/model. No official label or annotation metadata enters the vision prompt.

The three-task labeling pilot saved 3/3 proposals without failures. A development-only benchmark on the longest development video (78.734 seconds) returned valid features in 47.694127 seconds. Before seeing any new detector scores, the recipe is now frozen: four windows centered at 1/8, 3/8, 5/8 and 7/8 of clip duration, each with width min(4 seconds, duration/4); three chronological JPEG samples per window, 384-pixel maximum side. This gives 19 present/unknown/visibility indicators per window and 57 max/mean/std clip features. Each local request is capped at 45 seconds, with a 240-second global extraction budget. Four valid requests are required; entirely unobservable input and operational failures fail explicitly. Sparse samples do not establish continuous coverage. The benchmark cache is reused, without changing sources or descriptors, in the fresh training run.

Use numeric indicators for present/unknown primitive observations, visibility and their fixed temporal summaries. Treat model text as untrusted and keep all of it local; do not parse claims of theft, rule violations or invisible facts into labels. The extractor is fixed, label-independent and separate from fold-fitted supervised preprocessing.

Compare only the two established fixed clip-classifier recipes on the same three development folds, with threshold 0.5. Run the actual training and matched shuffled-target sanity checks, log every candidate and retain the holdout untouched. Promote only if development AUC is at least 0.60, balanced accuracy exceeds 0.50, recall does not regress from 10/17, and sanity/runtime checks pass. Report any localization regression separately. All development model selection remains uncertain on 32 clips.

Action-specific model training, action-type accuracy and source-to-rule claims still require independent local review and verified sources. Binary detection improvement can be investigated autonomously without waiting for all action annotations, while keeping this distinction explicit.
