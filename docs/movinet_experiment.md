# First local temporal experiment

The human approved starting the first experiment on 17 October 2026, prioritizing recognition and backend quality. All video processing remains local. The only review-interface changes are the controls needed to obtain genuine action labels.

## Exact model and runtime

- Model: Google MoViNet-A0-Stream, Kinetics-600, float16 TFLite variant, version 2. This is the corrected float16 architecture; version 1 is not used.
- Official model metadata explicitly assigns **Apache 2.0** to this exact variant/version. Evidence is preserved in `licenses/movinet/kaggle-model-metadata.json`, with source URL and checksum in `licenses/movinet/provenance.json`.
- Download: 6,999,449-byte archive; single 7,992,000-byte `2.tflite` member. Installed to `models/movinet/movinet_a0_stream.tflite` under the approved experiment. Weight SHA-256: `2afdb24bc127104171df2d133245beb3b8828705cc8d93f1aeb1e24392838924`. Hashes were calculated from the official HTTPS download; no independent publisher checksum was available.
- Proposed runtime: **ai-edge-litert==2.3.0**, Apache 2.0, CPython 3.12 Linux wheel 21,781,762 bytes. Its Interpreter API runs the TFLite model without installing the complete TensorFlow training stack. Additional missing transitive packages: `backports.strenum`, `flatbuffers`, `ml_dtypes`; existing NumPy/protobuf/tqdm/typing-extensions must retain compatible versions.
- The human explicitly approved the pinned LiteRT runtime on 17 October. Installation command: `UV_CACHE_DIR=/tmp/datathon-uv-cache uv add 'ai-edge-litert==2.3.0'`. Review the resulting lockfile changes before inference. Do not install accelerator extras, upload video, or call hosted inference.

Sources: [exact Google model metadata](https://www.kaggle.com/api/v1/models/list?search=movinet), [corrected model documentation](https://raw.githubusercontent.com/tensorflow/tfhub.dev/master/assets/docs/tensorflow/models/movinet/a0/stream/kinetics-600/classification/tflite/float16/lite/2.md), [runtime release](https://pypi.org/project/ai-edge-litert/2.3.0/), [official Python Interpreter migration](https://developers.google.com/edge/litert/migration), [runtime license](https://raw.githubusercontent.com/google-ai-edge/LiteRT/v2.3.0/LICENSE).

## Fixed first experiment

Use aspect-preserving RGB 172 × 172 input, five sampled frames per second, eight-second windows with two-second stride, and zero streaming state at the start of every window. Use the final 600 classifier logits as the frozen representation; they are general-action scores, not cashier-action predictions or penultimate embeddings. A small StandardScaler plus logistic-regression classifier is fitted on independently verified local action labels only, using config.SEED and a fixed 0.5 decision threshold.

The initial observable primitive is person-to-person object transfer. Similar single-person relocation and contact without transfer are hard negatives; background and uncertainty remain separate. This primitive does not itself establish a bank-rule violation. The fixed fold-0 development partition separates eight training recordings (four positive/four hard negative) from eight blind pilot recordings (four/four). Missing support blocks semantic training. The eight reserved recordings remain untouched.

Before bank-media extraction, run a bounded synthetic runtime and repeatability benchmark. Then require reviewed labels and an immutable pilot manifest before training. Synthetic success proves operation and speed only. The fixed semantic gate and later full development evaluation remain those in `docs/recognition_recovery_plan.md`.
