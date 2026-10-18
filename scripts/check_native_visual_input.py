"""Four synthetic color controls; inspection is default, never a bank/action gate."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np

from scripts import check_temporal_protocol as native
from scripts.provision_action_model import verify_ready as verify_model
from scripts.provision_video_decoder import verify_ready as verify_decoder
from src import config


PINNED_HELPER_SHA256 = "62b0c8d36b6db83e5ab070a7a6e4dcbef41a45581cc9364942ecb08a0b3fd669"
COLORS = ("red", "blue", "unclear")
CONTROLS = (("native_video", "red"), ("native_video", "blue"),
            ("single_jpeg", "red"), ("single_jpeg", "blue"))
PROMPT = ("Identify the dominant color across these images/video. "
          "Return only the color field: red, blue, or unclear.")
SCHEMA = {"type": "object", "properties": {"color": {"type": "string", "enum": list(COLORS)}},
          "required": ["color"], "additionalProperties": False}


def encode_color(directory, color):
    """Encode the same flat BGR source as four MJPEG frames and one JPEG."""
    if color not in COLORS[:2]:
        raise ValueError("Unsupported synthetic color")
    frame = np.full((256, 256, 3), (0, 0, 255) if color == "red" else (255, 0, 0), np.uint8)
    path = Path(directory) / (color + ".avi")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2., (256, 256))
    try:
        if not writer.isOpened():
            raise ValueError("Synthetic MJPEG encoding is unavailable")
        for _ in range(4):
            writer.write(frame)
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(path))
    try:
        if (not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) != 4
                or abs(cap.get(cv2.CAP_PROP_FPS) - 2.) > 1e-6):
            raise ValueError("Synthetic video differs from the fixed timing recipe")
        for _ in range(4):
            ok, decoded = cap.read()
            if not ok or decoded.shape != frame.shape or np.max(np.abs(decoded.astype(int) - frame)) > 5:
                raise ValueError("Synthetic video color or shape differs from its source")
        if cap.read()[0]:
            raise ValueError("Synthetic video has extra frames")
    finally:
        cap.release()
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    payloads = {"native_video": path.read_bytes(), "single_jpeg": jpeg.tobytes()}
    if not ok or any(not value or len(value) > 2 * 1024 * 1024 for value in payloads.values()):
        raise ValueError("Synthetic payload encoding failed or exceeded its bound")
    return payloads, hashlib.sha256(frame.tobytes()).hexdigest()


def request_body(alias, kind, payload):
    encoded = base64.b64encode(payload).decode()
    if kind == "native_video":
        media = {"type": "input_video", "input_video": {"data": encoded}}
    elif kind == "single_jpeg":
        media = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}
    else:
        raise ValueError("Unsupported synthetic input kind")
    return {"model": alias, "temperature": 0, "seed": config.SEED, "cache_prompt": False,
            "max_tokens": 32, "response_format": {"type": "json_object", "schema": SCHEMA},
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, media]}]}


def parse_color(response):
    try:
        choices = response["choices"]
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError
        value = json.loads(choices[0]["message"]["content"])
        if not isinstance(value, dict) or set(value) != {"color"} or value["color"] not in COLORS:
            raise ValueError
        return value["color"]
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        raise ValueError("Synthetic response does not match the fixed color schema") from None


def reported_metrics(response):
    """Missing counts stay missing; retain only finite nonnegative numeric fields."""
    keys = {"usage": ("prompt_tokens", "completion_tokens", "total_tokens", "image_tokens", "video_tokens"),
            "timings": ("cache_n", "prompt_n", "predicted_n", "prompt_ms", "predicted_ms")}
    metrics = {}
    for group, names in keys.items():
        values = response.get(group, {})
        if isinstance(values, dict):
            for name in names:
                value = values.get(name)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    metrics[group + "." + name] = value
    details = response.get("usage", {})
    if isinstance(details, dict) and isinstance(details.get("prompt_tokens_details"), dict):
        for name in ("image_tokens", "video_tokens", "text_tokens", "cached_tokens"):
            value = details["prompt_tokens_details"].get(name)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                metrics["usage.prompt_tokens_details." + name] = value
    return metrics


def run_probe(run_dir):
    helper_source, own_source = Path(native.__file__).read_bytes(), Path(__file__).read_bytes()
    if hashlib.sha256(helper_source).hexdigest() != PINNED_HELPER_SHA256:
        raise ValueError("Synthetic helper differs from the controlled baseline")
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(config.DATA_PROC.resolve()):
        raise ValueError("Synthetic output must be inside data/processed")
    if run_dir.exists():
        raise FileExistsError("Synthetic diagnostic requires a fresh output directory")
    model, decoder = verify_model(), verify_decoder()
    provenance = {"helper": native.protocol_provenance(model, decoder),
                  "source_sha256": hashlib.sha256(own_source).hexdigest(),
                  "server_command": native.server_command(model, decoder)}
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "native_helper_source_snapshot.py").write_bytes(helper_source)
    (run_dir / "source_snapshot.py").write_bytes(own_source)
    records, failure, stage, started = [], None, "encoding", time.monotonic()
    try:
        with tempfile.TemporaryDirectory(dir=run_dir) as temporary:
            inputs = {color: encode_color(temporary, color) for color in COLORS[:2]}
            stage = "service_start"
            with native.native_server(model, decoder):
                for kind, expected in CONTROLS:
                    stage = kind + "_" + expected
                    payloads, source_hash = inputs[expected]
                    payload = payloads[kind]
                    body = request_body(model["alias"], kind, payload)
                    call_started = time.monotonic()
                    response = native.request_json("/v1/chat/completions", body, timeout=45)
                    observed = parse_color(response)
                    records.append({"input_kind": kind, "expected": expected, "observed": observed,
                                    "correct": observed == expected, "elapsed_sec": time.monotonic() - call_started,
                                    "source_frame_sha256": source_hash, "payload_bytes": len(payload),
                                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                                    "request_body_sha256": hashlib.sha256(json.dumps(body).encode()).hexdigest(),
                                    "reported_metrics": reported_metrics(response)})
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        failure = {"stage": stage, "exception_type": type(error).__name__}
    passed = {kind: len(items := [r for r in records if r["input_kind"] == kind]) == 2
              and all(r["correct"] for r in items) for kind in ("native_video", "single_jpeg")}
    result = {"purpose": "synthetic_visual_input_only", "cases": records, "controls_passed": passed,
              "operational_failure": failure, "complete": len(records) == 4,
              "seed": config.SEED, "temperature": 0, "cache_prompt": False, "max_tokens": 32,
              "request_timeout_sec": 45, "video_fps": 2, "native_frames_per_case": 4,
              "bank_videos_used": 0, "holdout_videos_used": 0, "provenance": provenance,
              "elapsed_sec": time.monotonic() - started, "authorizes_bank_video_batch": False,
              "limitations": ["Color controls diagnose input delivery, not motion or bank-action accuracy",
                              "Missing image/video token counts do not establish whether native frames were ingested",
                              "The failed temporal gate remains unchanged and blocked"]}
    (run_dir / "visual_input.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "native-visual-input")
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"will_execute": False, "synthetic_cases": 4, "cache_prompt": False,
                          "bank_videos_used": 0, "authorizes_bank_video_batch": False}))
        return
    result = run_probe(args.run_dir)
    print(json.dumps(result), flush=True)
    if not result["complete"] or not all(result["controls_passed"].values()):
        raise SystemExit("Synthetic visual input diagnostic did not pass; bank-video gate remains blocked")


if __name__ == "__main__":
    main()
