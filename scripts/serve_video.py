"""Start the local P2 video service without printing its generated API key."""
import argparse
import os
from pathlib import Path
import secrets

import uvicorn

from src import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine", choices=("artifact", "semantic-v1"), default="artifact")
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "video-baseline-comparison")
    parser.add_argument("--key-file", type=Path, default=config.DATA_PROC / "video-service/api-key.txt")
    args = parser.parse_args()
    model = args.run_dir.resolve() / "models/baseline.joblib"
    if args.engine == "artifact" and not model.is_file():
        raise SystemExit("The local video model is missing; run the documented training command first")
    key_path = args.key_file.resolve()
    if key_path.is_relative_to(config.DATA_RAW.resolve()):
        raise SystemExit("API key must not be stored in data/raw")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if not key_path.exists():
        descriptor = os.open(key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    key = key_path.read_text().strip()
    if len(key) < 24:
        raise SystemExit("The configured local API key is too short")
    os.chmod(key_path, 0o600)
    os.environ["DATATHON_API_KEY"] = key
    os.environ["DATATHON_VIDEO_MODEL"] = str(model)
    os.environ["DATATHON_VIDEO_ENGINE"] = args.engine
    print(f"Local video UI: http://{args.host}:{args.port}/", flush=True)
    print(f"API key stored locally in {key_path}; its value is not logged.", flush=True)
    uvicorn.run("src.video_api:app", host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
