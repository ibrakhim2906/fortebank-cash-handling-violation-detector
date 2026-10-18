"""Six fixed synthetic controls for the explicit no-prefix-cache hypothesis.

Inspection is the default. This separate mechanism diagnostic never reads bank
media or labels, never repairs a gate, and never authorizes a bank-video batch.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

from scripts import check_temporal_protocol as native
from scripts.provision_action_model import verify_ready as verify_model
from scripts.provision_video_decoder import verify_ready as verify_decoder
from src import config


PINNED_HELPER_SHA256 = "62b0c8d36b6db83e5ab070a7a6e4dcbef41a45581cc9364942ecb08a0b3fd669"
CONTROLS = native.MOTIONS[:3] * 2


def _request_body(alias, payload):
    # Keep the failed gate's recipe; cache_prompt is the sole policy change.
    return {"model": alias, "temperature": 0, "seed": config.SEED,
            "max_tokens": 32, "response_format": {"type": "json_object", "schema": native.SCHEMA},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": native.PROMPT},
                {"type": "input_video", "input_video": {"data": base64.b64encode(payload).decode()}},
            ]}], "cache_prompt": False}


def _cache_timing(response):
    timings = response.get("timings")
    if not isinstance(timings, dict) or "cache_n" not in timings:
        return None, "missing"
    count = timings["cache_n"]
    return (count, "reported") if type(count) is int and count >= 0 else (None, "invalid")


def run_probe(run_dir):
    helper_source = Path(native.__file__).read_bytes()
    if hashlib.sha256(helper_source).hexdigest() != PINNED_HELPER_SHA256:
        raise ValueError("Synthetic helper differs from the controlled baseline")
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(config.DATA_PROC.resolve()):
        raise ValueError("Cache diagnostic output must be inside data/processed")
    if run_dir.exists():
        raise FileExistsError("Cache diagnostic requires a fresh output directory")
    model, decoder = verify_model(), verify_decoder()
    provenance = native.protocol_provenance(model, decoder)
    own_source = Path(__file__).read_bytes()
    provenance = {"helper": provenance, "source_sha256": hashlib.sha256(own_source).hexdigest(),
                  "request_policy": {"cache_prompt": False},
                  "server_command": native.server_command(model, decoder)}
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "native_helper_source_snapshot.py").write_bytes(helper_source)
    (run_dir / "source_snapshot.py").write_bytes(own_source)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    records, failure = [], None
    stage, started = "service_start", time.monotonic()
    try:
        with native.native_server(model, decoder):
            with tempfile.TemporaryDirectory(dir=run_dir) as temporary:
                payloads = {}
                for index, expected in enumerate(CONTROLS):
                    stage = "synthetic_" + str(index + 1) + "_" + expected
                    if expected not in payloads:
                        payloads[expected] = native.encode_video(Path(temporary) / (expected + ".avi"), expected)
                    payload = payloads[expected]
                    call_started = time.monotonic()
                    response = native.request_json("/v1/chat/completions", _request_body(model["alias"], payload), timeout=45)
                    observed = native.parse_motion(response)
                    cache_n, cache_status = _cache_timing(response)
                    records.append({"pass_index": index // 3 + 1, "expected": expected,
                                    "observed": observed, "correct": observed == expected,
                                    "elapsed_sec": time.monotonic() - call_started,
                                    "synthetic_payload_sha256": hashlib.sha256(payload).hexdigest(),
                                    "cache_n": cache_n, "cache_n_status": cache_status})
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        failure = {"stage": stage, "exception_type": type(error).__name__}
        status = getattr(error, "code", None)
        if type(status) is int:
            failure["http_status"] = status
    complete = len(records) == len(CONTROLS)
    consistent = complete and all(records[i]["observed"] == records[i + 3]["observed"] for i in range(3))
    counts = [record["cache_n"] for record in records]
    result = {"purpose": "synthetic_cache_behavior_only", "cache_prompt_requested": False,
              "protocol": "input_video/native frame grouping; explicit frame times in prompt",
              "video_fps": 2, "video_timestamp_interval_ms": 0, "frames_per_case": 4,
              "seed": config.SEED, "cases": records, "repeat_consistent": consistent,
              "passed": failure is None and complete and consistent and all(r["correct"] for r in records),
              "cache_metadata_complete": complete and all(count is not None for count in counts),
              "reported_zero_cache_n": sum(count == 0 for count in counts),
              "reported_positive_cache_n": sum(count is not None and count > 0 for count in counts),
              "missing_or_invalid_cache_n": sum(count is None for count in counts),
              "operational_failure": failure, "elapsed_sec": time.monotonic() - started,
              "bank_videos_used": 0, "holdout_videos_used": 0,
              "bank_action_accuracy_verified": False, "authorizes_bank_video_batch": False,
              "provenance": provenance,
              "limitations": ["Synthetic mechanism diagnostic only; no bank-action accuracy or exclusive-cause claim",
                              "Missing cache timings cannot prove prefix reuse was disabled",
                              "First and repeat passes do not establish cold or warm internal cache state",
                              "The preserved failed temporal gate remains separate and unchanged"]}
    (run_dir / "cache_behavior.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute only with approved installed assets and the old model server stopped")
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "local-cache-behavior")
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"will_execute": False, "synthetic_cases": 6, "cache_prompt": False,
                          "bank_videos_used": 0, "authorizes_bank_video_batch": False}))
        return
    try:
        result = run_probe(args.run_dir)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError):
        raise SystemExit("Cache mechanism diagnostic failed; no response payload emitted") from None
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit("Fixed synthetic cache diagnostic did not pass; bank-video gate remains blocked")


if __name__ == "__main__":
    main()
