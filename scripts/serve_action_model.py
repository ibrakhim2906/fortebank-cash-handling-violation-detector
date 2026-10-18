"""Launch the verified CPU-only vision model on a fixed Linux loopback endpoint."""
import os
from pathlib import Path
import subprocess

from scripts.provision_action_model import ASSETS, safe_path, verify_ready
from src import config


def command(root: Path, manifest: dict) -> list[str]:
    candidates = [safe_path(root, path) for path in manifest["runtime_files"]
                  if Path(path).name == "llama-server"]
    if len(candidates) != 1:
        raise ValueError("Installation must contain exactly one verified llama-server")
    return [str(candidates[0]), "-m", str(safe_path(root, ASSETS[0].path)),
            "--mmproj", str(safe_path(root, ASSETS[1].path)),
            "--host", "127.0.0.1", "--port", "8003", "--n-gpu-layers", "0",
            "--no-mmproj-offload", "--threads", "8", "--ctx-size", "4096",
            "--parallel", "1", "--seed", str(config.SEED), "--image-max-tokens", "256",
            "--alias", "local-action-review", "--no-webui", "--log-disable"]


def main():
    root = config.ROOT
    manifest = verify_ready(root)
    args = command(root, manifest)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("LLAMA_", "MTMD_", "HF_", "HUGGING_FACE_", "TRANSFORMERS_"))}
    # Local file flags only; no HF identifiers, URLs, remote model loading or tools.
    runtime = Path(args[0]).parent
    environment["LD_LIBRARY_PATH"] = str(runtime) + ":" + str(runtime.parent)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["LLAMA_LOG_DISABLE"] = "1"
    print("Verified local CPU vision service: http://127.0.0.1:8003", flush=True)
    # Suppress model output, prompts, request bodies and any media-derived text.
    with open(os.devnull, "wb") as output:
        subprocess.run(args, cwd=root, env=environment, stdout=output, stderr=output, check=True)


if __name__ == "__main__":
    main()
