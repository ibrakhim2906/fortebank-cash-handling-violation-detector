# Fast structured video recognition — fixed independent trial

The human asks to continue toward impressive case-specific video recognition while the slower local-language-model comparison runs. This independent route uses the already approved OpenCV dependency, not another runtime/package. The official Apache-2.0 OpenCV Zoo YOLOX-S asset is approximately 35.9 MB. No bank media or descriptions are uploaded. No outcomes from this representation have been inspected when this plan is written.

## Fixed model and visual scope

Pin Zoo revision `47534e27c9851bb1128ccc0102f1145e27f23f98`, weight SHA-256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`, and preserve the folder's Apache-2.0 license. Existing mismatched assets fail verification and remain preserved. Verify a synthetic blank-image forward before processing development footage; record output shape and CPU latency, not pixels.

Use the publisher preprocessing: RGB float32, 640-square top-left padding at 114, aspect-preserving resize, CHW batch, no division by 255. Decode the three stride grids and apply class-aware NMS. Freeze confidence 0.35 and IoU 0.50. Retain only COCO person, backpack, handbag, suitcase, cellphone and laptop classes (0, 24, 26, 28, 67, 63). These are pretrained object predictions, not verified action labels. There is no cash or hand class, person identification, ownership inference or automatic legal conclusion.

## Fixed temporal representation

Sample requested times 0, 0.5, 1.0 seconds and so on, strictly below duration, with at most 300 instants; fail unsupported inputs rather than silently reduce the sampling rate. Sequentially decode/grab-skip to these instants, validate chronological actual frame positions, and retain source/timing/object provenance locally. Use a 240-second extraction deadline, CPU execution and two OpenCV threads.

Use two-second windows. The frozen 57 window features are:

- For each of six classes: mean/max accepted count, maximum score and mean normalized box area (24).
- For each of five non-person classes: mean/min normalized nearest-person center distance and missing-pair fraction (15).
- For each class: anonymous within-video track count, mean within-window visible span and mean normalized center speed (18).

Associate same-class detections greedily by IoU >=0.3 with a maximum one-second gap and deterministic tie-breaking. Track IDs reset for every recording and do not identify people. A zero count means no accepted detector prediction, not proof that an object/action is absent. Clip features are fixed max/mean/std summaries, giving 171 values. No filename, official target, anchor, review status, clock metadata or model-generated action target enters the visual inputs.

## Measurement and decision

Use the unchanged 32 development content groups, three folds, linear/shallow-LightGBM recipes and threshold 0.5. Train preprocessing inside folds. Keep all eight reserved videos untouched. Preserve every previous run; use a fresh `object-tracks` run and cache. Do not combine this representation with other candidates or search thresholds/hyperparameters based on outcomes in this trial.

Run actual `src.train` and `src.sanity`, log keep/drop, and preserve the same promotion gate: AUC >=0.60, balanced accuracy >0.50, recall >=10/17, and passed sanity/runtime checks. Report localization separately and stop/report any suspicious feature or shuffled-target result. A small repeatedly used development set cannot establish generalization to unseen cameras.

Dense object timelines can support more useful evidence and subsequent local action checks. They do not by themselves establish cashier procedure compliance, handheld use, theft, omitted counting or any event-to-rule association. Actual event-type accuracy and rule grounding remain separate requirements; no unsupported explanation is activated just to make the demo look complete.
