"""Two synthetic native-video color-order controls; inspection is the default."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np

from scripts import check_native_visual_input as visual
from scripts import check_temporal_protocol as native
from scripts.provision_action_model import verify_ready as verify_model
from scripts.provision_video_decoder import verify_ready as verify_decoder
from src import config


ORDERS = (("red", "red", "blue", "blue"), ("blue", "blue", "red", "red"))
COLORS = ("red", "blue", "unclear")
PROMPT = ("The video contains four chronological frames. Identify the dominant color "
          "of the first frame and the last frame. Return only first_color and last_color; "
          "each must be red, blue, or unclear.")
SCHEMA = {"type": "object", "properties": {
    name: {"type": "string", "enum": list(COLORS)} for name in ("first_color", "last_color")},
    "required": ["first_color", "last_color"], "additionalProperties": False}


def encode_order(path, order):
    if tuple(order) not in ORDERS:
        raise ValueError("Unsupported fixed color order")
    frames = [np.full((256, 256, 3), (0, 0, 255) if color == "red" else (255, 0, 0), np.uint8)
              for color in order]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2., (256, 256))
    try:
        if not writer.isOpened():
            raise ValueError("Synthetic MJPEG encoding is unavailable")
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(path))
    try:
        if (not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) != 4
                or abs(cap.get(cv2.CAP_PROP_FPS) - 2.) > 1e-6):
            raise ValueError("Encoded order control differs from the fixed timing recipe")
        for expected in frames:
            ok, frame = cap.read()
            if not ok or frame.shape != expected.shape or np.max(np.abs(frame.astype(int) - expected)) > 5:
                raise ValueError("Encoded frame does not preserve its expected color/order")
        if cap.read()[0]:
            raise ValueError("Encoded order control has extra frames")
    finally:
        cap.release()
    payload = Path(path).read_bytes()
    if not payload or len(payload) > 2 * 1024 * 1024:
        raise ValueError("Synthetic payload has an unsupported bounded size")
    return payload, [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]


def request_body(alias, payload):
    return {"model": alias, "temperature": 0, "seed": config.SEED, "cache_prompt": False,
            "max_tokens": 32, "response_format": {"type": "json_object", "schema": SCHEMA},
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                {"type": "input_video", "input_video": {"data": base64.b64encode(payload).decode()}}]}]}


def parse_colors(response):
    try:
        choices = response["choices"]
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError
        value = json.loads(choices[0]["message"]["content"])
        if (not isinstance(value, dict) or set(value) != {"first_color", "last_color"}
                or any(color not in COLORS for color in value.values())):
            raise ValueError
        return value
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        raise ValueError("Synthetic response does not match the fixed first/last schema") from None


def run_probe(run_dir):
    helper_source, own_source = Path(native.__file__).read_bytes(), Path(__file__).read_bytes()
    if hashlib.sha256(helper_source).hexdigest() != visual.PINNED_HELPER_SHA256:
        raise ValueError("Native helper differs from the frozen baseline")
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(config.DATA_PROC.resolve()):
        raise ValueError("Synthetic output must be inside data/processed")
    if run_dir.exists():
        raise FileExistsError("Order diagnostic requires a fresh output directory")
    model, decoder = verify_model(), verify_decoder()
    provenance = {"helper": native.protocol_provenance(model, decoder),
                  "source_sha256": hashlib.sha256(own_source).hexdigest(),
                  "metrics_helper_sha256": hashlib.sha256(Path(visual.__file__).read_bytes()).hexdigest(),
                  "server_command": native.server_command(model, decoder)}
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "source_snapshot.py").write_bytes(own_source)
    (run_dir / "native_helper_source_snapshot.py").write_bytes(helper_source)
    records, failure, stage, started = [], None, "encoding", time.monotonic()
    try:
        with tempfile.TemporaryDirectory(dir=run_dir) as temporary:
            inputs = [encode_order(Path(temporary) / (str(i) + ".avi"), order)
                      for i, order in enumerate(ORDERS)]
            stage = "service_start"
            with native.native_server(model, decoder):
                for index, (order, (payload, hashes)) in enumerate(zip(ORDERS, inputs)):
                    stage = "order_" + str(index + 1)
                    body, call_started = request_body(model["alias"], payload), time.monotonic()
                    response = native.request_json("/v1/chat/completions", body, timeout=45)
                    observed = parse_colors(response)
                    expected = {"first_color": order[0], "last_color": order[-1]}
                    records.append({"source_order": list(order), "expected": expected, "observed": observed,
                                    "correct": observed == expected, "elapsed_sec": time.monotonic() - call_started,
                                    "source_frame_sha256": hashes, "payload_sha256": hashlib.sha256(payload).hexdigest(),
                                    "request_body_sha256": hashlib.sha256(json.dumps(body).encode()).hexdigest(),
                                    "reported_metrics": visual.reported_metrics(response)})
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        failure = {"stage": stage, "exception_type": type(error).__name__}
    result = {"purpose": "synthetic_native_first_last_only", "cases": records,
              "passed": failure is None and len(records) == 2 and all(r["correct"] for r in records),
              "complete": len(records) == 2, "operational_failure": failure, "provenance": provenance,
              "seed": config.SEED, "temperature": 0, "cache_prompt": False, "max_tokens": 32,
              "request_timeout_sec": 45, "video_fps": 2, "frames_per_case": 4,
              "bank_videos_used": 0, "holdout_videos_used": 0, "authorizes_bank_video_batch": False,
              "elapsed_sec": time.monotonic() - started,
              "limitations": ["First/last synthetic color recognition is not bank-action or motion accuracy",
                              "The failed temporal gate remains unchanged and blocked"]}
    (run_dir / "frame_order.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "native-frame-order")
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"will_execute": False, "synthetic_cases": 2, "cache_prompt": False,
                          "bank_videos_used": 0, "authorizes_bank_video_batch": False}))
        return
    result = run_probe(args.run_dir)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit("Synthetic first/last diagnostic did not pass; bank-video gate remains blocked")


if __name__ == "__main__":
    main()
