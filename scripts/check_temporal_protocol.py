"""Bounded synthetic-only native-video diagnostic; inspection is the default.

No supplied recording, label or review record is read. Run only after the
decoder is approved/provisioned and the existing vision server has stopped.
The temporary server uses local assets, no media URLs and a loopback binding.
"""
import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import selectors
import socket
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import cv2
import numpy as np

from src import config


BASE_URL = "http://127.0.0.1:8003"
FRAME_TIMES = (0., .5, 1., 1.5)
MAX_RESPONSE_BYTES = 65536
MAX_WORKER_BYTES = 8 * MAX_RESPONSE_BYTES + 1024
MOTIONS = ("left_to_right", "right_to_left", "stationary", "unclear")
SCHEMA = {"type": "object", "properties": {"motion": {"type": "string", "enum": list(MOTIONS)}},
          "required": ["motion"], "additionalProperties": False}
PROMPT = (
    "This synthetic silent video contains one red square on a white background. "
    "The four chronological frames are at 0.0, 0.5, 1.0 and 1.5 seconds. "
    "Classify how the square moves across the whole sequence, using screen coordinates. "
    "Return only the motion field: left_to_right, right_to_left, stationary, or unclear."
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Synthetic local vision requests cannot redirect")


def _request_json_http(path, body=None, *, timeout=45):
    """Worker-only HTTP path; socket timeouts supplement the parent deadline."""
    if path not in {"/health", "/v1/models", "/v1/chat/completions"}:
        raise ValueError("Unsupported synthetic diagnostic endpoint")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    data = None if body is None else json.dumps(body).encode()
    request = Request(BASE_URL + path, data=data, headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError("Synthetic diagnostic received an unexpected status")
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError("Synthetic diagnostic response exceeds its bound")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Synthetic diagnostic requires a JSON object")
    return value


def _request_worker(connection, path, body, timeout):
    """Return bounded structured output without printing any request/response."""
    try:
        descriptor = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(descriptor, 1)
            os.dup2(descriptor, 2)
        finally:
            os.close(descriptor)
        value = _request_json_http(path, body, timeout=timeout)
        packet = json.dumps({"ok": True, "response": value}, ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8")
        if len(packet) > MAX_WORKER_BYTES:
            raise ValueError
    except BaseException:
        # No exception text, server body, media bytes or descriptions cross on
        # the error path. The owned worker also suppresses stderr/stdout.
        packet = b'{"ok":false}'
    try:
        remaining = memoryview(packet)
        while remaining:
            sent = os.write(connection.fileno(), remaining)
            if sent <= 0:
                return
            remaining = remaining[sent:]
    except (OSError, ValueError):
        pass
    finally:
        connection.close()


def _read_worker_result(connection, deadline):
    """Drain bytes nonblocking; partial pipe messages cannot bypass timeout."""
    descriptor = connection.fileno()
    os.set_blocking(descriptor, False)
    output = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Local video request exceeded its wall-clock timeout")
            if not selector.select(remaining):
                continue
            try:
                chunk = os.read(descriptor, min(65536, MAX_WORKER_BYTES - len(output) + 1))
            except BlockingIOError:
                continue
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > MAX_WORKER_BYTES:
                raise ValueError("Local video request worker output exceeds its bound")
    try:
        packet = json.loads(output)
        if (not isinstance(packet, dict) or set(packet) != {"ok", "response"}
                or packet["ok"] is not True or not isinstance(packet["response"], dict)):
            raise ValueError
        return packet["response"]
    except (ValueError, TypeError, UnicodeError):
        raise RuntimeError("Local video request failed or returned invalid structured output") from None


def _stop_request_worker(process):
    """Bound cleanup separately; never leave a timed-out transport worker."""
    if process.is_alive():
        process.terminate()
    process.join(timeout=.25)
    if process.is_alive():
        process.kill()
        process.join(timeout=.25)
    if process.is_alive():
        raise RuntimeError("Local video request worker could not be stopped")
    process.close()


def request_json(path, body=None, *, timeout=45):
    """Use one owned spawned worker for a true whole-request deadline.

    Timeout includes process startup, serialization and HTTP handling. Worker
    cleanup adds at most two bounded 0.25-second joins after the request budget.
    """
    if path not in {"/health", "/v1/models", "/v1/chat/completions"}:
        raise ValueError("Unsupported synthetic diagnostic endpoint")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Local video request timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_request_worker, args=(writer, path, body, timeout), daemon=True)
    started = False
    try:
        process.start()
        started = True
        writer.close()
        return _read_worker_result(reader, deadline)
    finally:
        reader.close()
        writer.close()
        if started:
            _stop_request_worker(process)
        else:
            process.close()


def motion_frames(motion):
    if motion not in MOTIONS[:3]:
        raise ValueError("Unsupported synthetic motion")
    positions = [24, 80, 136, 192]
    if motion == "right_to_left":
        positions.reverse()
    elif motion == "stationary":
        positions = [108] * 4
    frames = []
    for left in positions:
        frame = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.rectangle(frame, (left, 104), (left + 32, 136), (0, 0, 255), thickness=-1)
        frames.append(frame)
    return frames


def encode_video(path, motion):
    """Write four known frames at 2 FPS; verify the actual encoded metadata."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2., (256, 256))
    if not writer.isOpened():
        raise ValueError("Local OpenCV cannot encode the synthetic video")
    try:
        for frame in motion_frames(motion):
            writer.write(frame)
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != 4 or abs(cap.get(cv2.CAP_PROP_FPS) - 2.) > 1e-6:
            raise ValueError("Synthetic video timing metadata differs from the fixed recipe")
        for _ in FRAME_TIMES:
            ok, frame = cap.read()
            if not ok or frame.shape != (256, 256, 3):
                raise ValueError("Synthetic video could not be decoded completely")
        if cap.read()[0]:
            raise ValueError("Synthetic video has unexpected extra frames")
    finally:
        cap.release()
    payload = Path(path).read_bytes()
    if not payload or len(payload) > 2 * 1024 * 1024:
        raise ValueError("Synthetic video has an unsupported bounded size")
    return payload


def parse_motion(response):
    try:
        choices = response["choices"]
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError
        value = json.loads(choices[0]["message"]["content"])
        if (not isinstance(value, dict) or set(value) != {"motion"}
                or value["motion"] not in MOTIONS):
            raise ValueError
        return value["motion"]
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        raise ValueError("Synthetic response does not match the fixed motion schema") from None


def server_command(model, decoder):
    from scripts.serve_action_model import command
    return command(config.ROOT, model) + [
        "--offline", "--video-ffmpeg-dir", str(decoder["bin_dir"]),
        "--video-fps", "2", "--video-timestamp-interval", "0",
    ]


def stop_server(process):
    """Reap the owned server and decoder children in their own process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def protocol_provenance(model, decoder):
    sources = ("scripts/provision_action_model.py", "scripts/provision_video_decoder.py",
               "scripts/serve_action_model.py")
    return {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inherited_sources": {name: hashlib.sha256((config.ROOT / name).read_bytes()).hexdigest()
                                  for name in sources},
            "model_weights_sha256": model["weights_sha256"],
            "model_projector_sha256": model["projector_sha256"],
            "runtime_release": model["runtime_release"],
            "decoder_archive_sha256": decoder["archive_sha256"]}


def verify_temporal_gate(report_path, model, decoder):
    path = Path(report_path)
    if not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("Completed synthetic temporal gate is missing or oversized")
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("Synthetic temporal gate has an invalid report schema")
    cases = report.get("cases", [])
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("Synthetic temporal gate has invalid control records")
    if (report.get("passed") is not True or report.get("operational_failure") is not None
            or report.get("seed") != config.SEED or report.get("bank_videos_used") != 0
            or report.get("holdout_videos_used") != 0 or report.get("frames_per_case") != 4
            or report.get("video_fps") != 2 or report.get("video_timestamp_interval_ms") != 0
            or len(cases) != 3
            or [case.get("expected") for case in cases] != list(MOTIONS[:3])
            or any(case.get("correct") is not True for case in cases)
            or any(report.get(key) != value for key, value in protocol_provenance(model, decoder).items())):
        raise ValueError("Synthetic temporal gate did not pass for the exact local protocol/assets")
    return report


@contextmanager
def native_server(model, decoder):
    """Own one verified local server; callers verify assets before opening media."""
    # Never start a second large model or interrupt the frozen comparison.
    with socket.socket() as guard:
        guard.bind(("127.0.0.1", 8003))
    args = server_command(model, decoder)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("LLAMA_", "MTMD_", "HF_", "HUGGING_FACE_", "TRANSFORMERS_"))}
    binary_dir = Path(args[0]).parent
    environment.update(LD_LIBRARY_PATH=str(binary_dir) + ":" + str(binary_dir.parent),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LLAMA_LOG_DISABLE="1")
    with open(os.devnull, "wb") as output:
        process = subprocess.Popen(args, cwd=config.ROOT, env=environment, stdout=output, stderr=output,
                                   start_new_session=True)
        try:
            ready_by = time.monotonic() + 60
            while True:
                remaining = ready_by - time.monotonic()
                if process.poll() is not None or remaining <= 0:
                    raise ValueError("Temporary local vision service failed to become ready")
                try:
                    request_json("/health", timeout=min(2, remaining))
                    break
                except (OSError, URLError, RuntimeError):
                    # The transport worker deliberately sanitizes connection
                    # errors. Retry only readiness; capability/schema errors
                    # below remain terminal once the service answers health.
                    time.sleep(min(.25, max(0., ready_by - time.monotonic())))
            advertised = request_json("/v1/models", timeout=5).get("data", [])
            if (len(advertised) != 1 or advertised[0].get("id") != model["alias"]
                    or "video" not in advertised[0].get("architecture", {}).get("input_modalities", [])):
                raise ValueError("Temporary local model does not advertise the required video capability")
            yield process
        finally:
            stop_server(process)


def run_probe(run_dir):
    from scripts.provision_action_model import verify_ready as verify_model
    from scripts.provision_video_decoder import verify_ready as verify_decoder
    model, decoder = verify_model(), verify_decoder()
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(config.DATA_PROC.resolve()):
        raise ValueError("Temporal diagnostic output must be inside data/processed")
    run_dir.mkdir(parents=True, exist_ok=False)
    records, failure = [], None
    stage = "service_start"
    started = time.monotonic()
    try:
        with native_server(model, decoder):
            with tempfile.TemporaryDirectory(dir=run_dir) as temporary:
                for expected in MOTIONS[:3]:
                    stage = "synthetic_" + expected
                    payload = encode_video(Path(temporary) / (expected + ".avi"), expected)
                    body = {"model": model["alias"], "temperature": 0, "seed": config.SEED,
                            "max_tokens": 32, "response_format": {"type": "json_object", "schema": SCHEMA},
                            "messages": [{"role": "user", "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "input_video", "input_video": {"data": base64.b64encode(payload).decode()}},
                            ]}]}
                    call_started = time.monotonic()
                    observed = parse_motion(request_json("/v1/chat/completions", body, timeout=45))
                    records.append({"expected": expected, "correct": observed == expected,
                                    "elapsed_sec": time.monotonic() - call_started,
                                    "synthetic_payload_sha256": hashlib.sha256(payload).hexdigest()})
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        failure = {"stage": stage, "exception_type": type(error).__name__}
        status = getattr(error, "code", None)
        if type(status) is int:
            failure["http_status"] = status
    result = {"protocol": "input_video/native frame grouping; explicit frame times in prompt",
              "video_fps": 2, "video_timestamp_interval_ms": 0, "frames_per_case": 4,
              "seed": config.SEED, "cases": records, "passed": failure is None and len(records) == 3 and all(r["correct"] for r in records),
              "operational_failure": failure,
              "elapsed_sec": time.monotonic() - started, "bank_videos_used": 0,
              "holdout_videos_used": 0, "bank_action_accuracy_verified": False,
              **protocol_provenance(model, decoder),
              "limitations": ["Synthetic motion only; no bank-action, rule or event-timing accuracy claim",
                              "Native helper text timestamps disabled for this frame-grouping diagnostic; frame times given explicitly"]}
    (run_dir / "temporal_protocol.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute after decoder approval/install and stopping the existing model server")
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "temporal-protocol-native")
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"will_execute": False, "synthetic_cases": 3, "frames_per_case": 4,
                          "video_fps": 2, "bank_videos_used": 0, "requires_decoder_approval": True}))
        return
    try:
        result = run_probe(args.run_dir)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError):
        raise SystemExit("Temporal protocol diagnostic failed; no model response or media emitted") from None
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit("Fixed synthetic temporal gate failed; do not start a bank-video batch")


if __name__ == "__main__":
    main()
