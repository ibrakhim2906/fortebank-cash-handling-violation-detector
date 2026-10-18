# Approved direct pretrained action probe

The human approved testing what the installed pretrained model recognizes on development videos, entirely locally, on 17 October 2026. This is a separate inference experiment. The existing human-label-gated transfer training pilot remains unchanged.

## Frozen scope before inference

- Use eight unique recordings from the **training side of development fold 0**. Sort eligible group IDs by SHA-256 of `f"{config.SEED}:{group_id}"` and take the first eight. Do not use official targets, anchors, filenames or predictions for selection. All fold-0 validation recordings and all eight reserved recordings remain undecoded by this experiment.
- Use the installed, hash-pinned MoViNet-A0-Stream float16 TFLite v2 and LiteRT 2.3.0. Preserve `src/video_movinet.py`, the pretrained weights and the frozen evaluation protocol.
- Process each complete selected recording with the existing recipe: aspect-preserving 172-pixel RGB letterbox, five frames per second, eight-second windows, two-second stride and zero streaming state per window. No crops, prompt changes, training or threshold search.
- Pin the 600-class label map linked by the official MoViNet tutorial, including its source revision and local checksum. Inspect these five outputs: counting money (112), looking at phone (265), talking on cell phone (516), texting (525), writing (595). Indices are zero-based. No generic transfer class exists; award-giving is not a replacement.
- Preserve all 600 logits locally; compute numerically stable 600-way softmax and rank. These are uncalibrated action scores, not independent probabilities of each action, bank violations or action absence. Several actions may coexist.
- Freeze the manifest, source hashes, model provenance and selected source paths before decoding. Verify source hashes and the development-only path boundary before each extraction. Stop on provenance mismatch or invalid data; preserve partial results without presenting them as a completed run.
- Repeat the predetermined longest selected video with a newly constructed encoder in the same process. Compare logits, source times, frame indices and frame hashes. This checks fresh-instance repeatability, not an independent-process or independent-machine reproduction.

## Locality and evidence

Download only the public label map before inference. During model processing, block Python socket connection entry points and record attempted calls; this is an application-level check, not proof of OS-level isolation. No external model calls, bank pixels, per-video descriptions or per-video predictions may enter chat/tool output. Keep private manifests, predictions and sampling evidence under `data/processed/pretrained-action-probe-v1/` with restrictive permissions. Emit aggregate diagnostics only.

Show the selected training recordings and predictions in the existing authenticated local review page, in a clearly read-only pretrained-probe mode. Keep its ordinary annotation workflow and blind pilot separate. Prediction inspection cannot create independent gold labels. Do not publish the review page or change the jury service.

## What this experiment can establish

Report actual completion, runtime, finite/nonconstant outputs, target-class rank/score distributions, top-1/top-5 proposal counts and repeatability. Display all selected recordings locally so high-scoring examples are not the only evidence available. Separately identify target classes without a pretrained output, such as generic person-to-person transfer.

Action accuracy, event precision/recall, localization accuracy and bank-violation improvement remain **unmeasured** unless independently reviewed action truth already exists for these exact samples. The supplied binary violation labels are not counting/phone/writing labels. Do not label low-scoring actions absent or infer skipped counting. A constant or weak score pattern is a diagnostic finding, not proof that a particular action needs training. No automatic promotion or model fitting follows this probe.

## Deliverables and checks

Create the immutable manifest, private per-recording results, an aggregate summary, and `reports/pretrained_probe.md`; append the actual run and keep/drop decision to `experiments.md`. Test selection/exclusion, softmax/rank calculations, failed-run handling, provenance and authenticated read-only display. No existing feature/model recipe changes are planned, so fitting a classifier or running a shuffled-target experiment would answer a different question. Any such follow-up must be separately versioned and evaluated under the repository's train/sanity requirements.
