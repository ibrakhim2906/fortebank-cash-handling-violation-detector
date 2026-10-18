# Installed local action-label model

Status: the human explicitly approved the additional runtime/model download. All pinned weights, projector, licenses and runtime archive are installed and verified; 60 runtime files were materialized from the checked archive. The CPU server is running on loopback port 8003. A schema-constrained synthetic request succeeded in 10.119785 seconds and the first three development-label proposals were saved locally with no failures. These are unverified proposals, not action truth. The frozen detector experiment completed with AUC 0.5 and identical action features across all 32 development clips; it is rejected, with bulk drafting paused for pilot review. See `reports/qwen_action_detection.md` and `reports/action_labels.md`.

The proposed model is publisher-official [Qwen3-VL-2B-Instruct GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF), revision `52d6c8ffea26cc873ac5ad116f8631268d7eb503`, with a standalone [llama.cpp b11707 Linux CPU release](https://github.com/ggml-org/llama.cpp/releases/tag/b11707). The model card declares Apache-2.0 and CPU compatibility; the [Qwen3-VL source license](https://github.com/QwenLM/Qwen3-VL/blob/main/LICENSE) is Apache-2.0. The [llama.cpp release license](https://github.com/ggml-org/llama.cpp/blob/b11707/LICENSE) is MIT. These weights differ from Qwen2.5-VL-3B, whose research license must not be assumed to be Apache-2.0.

| Pinned component | Bytes | SHA-256 |
|---|---:|---|
| Qwen3VL-2B-Instruct-Q4_K_M.gguf | 1,107,409,952 | `089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae` |
| mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf | 445,053,216 | `f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82` |
| llama-b11707-bin-ubuntu-x64.tar.gz | 17,914,097 | `88879871f765c3dec22260046e10a76e2ed38de5bdf4a7f68049245a8f4056c8` |

The three main artifacts total 1,570,377,265 bytes, approximately 1.57 GB; the pinned Apache and MIT license texts add 12,435 bytes. Runtime extraction needs additional disk space. One process snapshot measured 4,412,584 KiB RSS; this is not peak RAM. The four-window benchmark on the 78.734-second longest development clip took 47.694127 seconds. No five-minute production guarantee follows from a single benchmark.

Read-only inspection, including exact source URLs and verified existing-file status:

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.provision_action_model
```

Approval has been received; these are the provisioning and launch commands:

```bash
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.provision_action_model --install
UV_CACHE_DIR=/tmp/datathon-uv-cache uv run --offline python -m scripts.serve_action_model
```

Files stay in `models/local-action-model/` and `tools/local-action-model/`; mismatching existing files are preserved and cause failure. All downloads are bounded by exact byte counts and checked against pinned SHA-256 values. Archive paths are checked, special files rejected, and internal library links materialized as verified ordinary files. The installation manifest records each extracted file hash. The launcher rechecks the installation before execution, uses explicit local model paths, binds only `127.0.0.1:8003`, disables GPU offload, uses seed 42, eight CPU threads, one slot, a 4,096-token context and at most 256 tokens per image. It disables request/model logging and the runtime UI. It does not enable tools or automatic model downloads.

No suitable runtime was initially found in the project environment or targeted Windows checks. The approved project-local llama.cpp installation now supplies it, without adding Python packages. FFmpeg is absent; `src.video_action_proposals` decodes bounded frames with the existing OpenCV package and sends them only to the fixed local endpoint. It disables proxy use, rejects redirects and writes only unverified local proposals. The initial permissive JSON response failed vocabulary validation on a synthetic image; the client now requests a strict JSON schema with one entry per allowed action ID. No media-derived text is emitted. Do not send bank footage, frames, base64 or clip descriptions to cloud tools.

This model may propose action observations for development clips. Proposed labels remain unverified suggestions, never organizer ground truth or legal findings. Unknown/occluded actions must permit abstention. Keep the eight holdout videos untouched. Start with a bounded pilot and measure runtime and label usefulness before corpus-wide processing or retraining. Human confirmation is still required for ambiguous actions and rule interpretation.
