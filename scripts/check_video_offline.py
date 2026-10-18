"""Check direct local inference with Python network connections denied.

Uses one development clip only. Emits timing/status, never pixels, evidence
payloads or clip-specific descriptions. This is an engineering runtime check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch

from src import config
from src.video_eval import load_protocol, protocol_groups
from src.video_inference import analyze_video


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    group = max(protocol_groups(protocol, "development"), key=lambda g: g["duration_sec"])
    if group["group_id"] in protocol["holdout_group_ids"]:
        raise ValueError("Offline check cannot use reserved holdout")
    path = config.ROOT / group["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != group["group_id"]:
        raise ValueError("Development video changed since the protocol was frozen")
    attempts = []
    loopback_connections = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def allowed(address):
        return isinstance(address, tuple) and len(address) >= 2 and address[:2] == ("127.0.0.1", 8003)

    def deny_connection(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("External network access forbidden during local inference")

    def connect(sock, address):
        if allowed(address):
            loopback_connections.append(1)
            return original_connect(sock, address)
        return deny_connection()

    def connect_ex(sock, address):
        if allowed(address):
            loopback_connections.append(1)
            return original_connect_ex(sock, address)
        return deny_connection()

    def create_connection(address, *args, **kwargs):
        if allowed(address):
            return original_create_connection(address, *args, **kwargs)
        return deny_connection()

    started = time.monotonic()
    with patch.object(socket.socket, "connect", connect), \
            patch.object(socket.socket, "connect_ex", connect_ex), \
            patch.object(socket, "create_connection", create_connection):
        result = analyze_video(path, model_path=args.run_dir / "models/baseline.joblib",
                               deadline=started + 240)
    elapsed = time.monotonic() - started
    assert not attempts and result["verdict"] in {"violation", "no_violation"}
    summary = {"duration_sec": group["duration_sec"], "elapsed_sec": elapsed,
               "verdict": result["verdict"], "events": len(result["violations"]),
               "python_connection_attempts": len(attempts), "holdout_videos_used": 0,
               "local_vision_connections": len(loopback_connections),
               "notes": "Single development example; Python connections denied except fixed local vision endpoint; not an accuracy or worst-case runtime estimate"}
    (args.run_dir / "offline_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
