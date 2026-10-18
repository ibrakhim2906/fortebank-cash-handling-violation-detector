"""Bounded private spatial cache/history diagnostic; default inspection only."""
import argparse
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from unittest.mock import patch

import cv2
import numpy as np

from scripts import check_native_visual_input as metrics
from scripts import check_spatial_replay as replay
from scripts import check_temporal_protocol as transport
from scripts.provision_action_model import verify_ready
from scripts.serve_action_model import command
from src import config


PHASES = ("fresh_process", "after_conditioning", "restarted_process")
engine = replay.engine


@contextmanager
def original_server(model):
    """Own the original CPU launch profile, with bounded cleanup and no logs."""
    with socket.socket() as guard:
        guard.bind(("127.0.0.1", 8003))
    args = command(config.ROOT, model)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("LLAMA_", "MTMD_", "HF_", "HUGGING_FACE_", "TRANSFORMERS_"))}
    runtime = Path(args[0]).parent
    environment.update(LD_LIBRARY_PATH=str(runtime) + ":" + str(runtime.parent),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LLAMA_LOG_DISABLE="1")
    with open(os.devnull, "wb") as output:
        process = subprocess.Popen(args, cwd=config.ROOT, env=environment, stdout=output,
                                   stderr=output, start_new_session=True)
        try:
            ready_by = time.monotonic() + 60
            while True:
                remaining = ready_by - time.monotonic()
                if process.poll() is not None or remaining <= 0:
                    raise RuntimeError("Owned original-profile service did not become ready")
                try:
                    transport.request_json("/health", timeout=min(2, remaining))
                    break
                except (OSError, RuntimeError):
                    time.sleep(min(.2, max(0, ready_by - time.monotonic())))
            models = transport.request_json("/v1/models", timeout=5).get("data")
            replay._require(isinstance(models, list) and len(models) == 1
                            and models[0].get("id") == model["alias"],
                            "Owned original-profile service identity differs")
            yield
        finally:
            transport.stop_server(process)


@contextmanager
def policy_requests(policy, budget, *, conditioning=False):
    """Override only cache_prompt; frozen parsing receives the normal envelope."""
    captures = []

    class Opener:
        def open(self, request, *, timeout):
            replay._require(type(policy) is bool and request.full_url == engine._observations.RECIPE["endpoint"]
                            and 0 < timeout <= 45 and len(captures) < (1 if conditioning else 4),
                            "Spatial diagnostic request differs from its bound")
            category, maximum = ("conditioning", 2) if conditioning else ("target", 24)
            replay._require(budget[category] < maximum, "Spatial diagnostic request budget exceeded")
            body = json.loads(request.data)
            body["cache_prompt"] = policy
            encoded = json.dumps(body).encode()
            normalized = {key: value for key, value in body.items() if key != "cache_prompt"}
            row = {"body_sha256": replay.hashlib.sha256(encoded).hexdigest(),
                   "normalized_body_sha256": replay._fingerprint(normalized), "body_bytes": len(encoded),
                   "cache_prompt": policy}
            captures.append(row)
            budget[category] += 1
            started = time.monotonic()
            try:
                response = transport.request_json("/v1/chat/completions", body, timeout=timeout)
                row["reported_metrics"] = metrics.reported_metrics(response)
                row["completed"] = True
                return io.BytesIO(json.dumps(response).encode())
            finally:
                row["elapsed_sec"] = time.monotonic() - started

    with patch.object(engine._observations, "build_opener", lambda *handlers: Opener()):
        yield captures


def _conditioning_jpeg():
    frame = np.full((256, 256, 3), (255, 0, 0), np.uint8)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    replay._require(ok, "Synthetic blue conditioning JPEG could not be encoded")
    return jpeg.tobytes()


def _condition(jpeg):
    engine._observations.request_observations([jpeg] * 3, timeout=45)


def _cache_exposure(rows):
    values = [row.get("reported_metrics", {}).get("timings.cache_n") for row in rows]
    return {"zero_cache_n_requests": sum(value == 0 for value in values),
            "positive_cache_n_requests": sum(value is not None and value > 0 for value in values),
            "missing_cache_n_requests": sum(value is None for value in values)}


def _summary(records, captures, cached, model):
    comparisons = {"cache_vs_" + phase: replay._comparison(cached, record, model)
                   for phase, record in zip(PHASES, records, strict=True)}
    comparisons.update({"fresh_vs_after_conditioning": replay._comparison(records[0], records[1], model),
                        "fresh_vs_restarted": replay._comparison(records[0], records[2], model),
                        "conditioned_vs_restarted": replay._comparison(records[1], records[2], model)})
    identities = [replay.identity_difference(cached, record) for record in records]
    hashes = [[row["body_sha256"] for row in capture] for capture in captures]
    normalized = [[row["normalized_body_sha256"] for row in capture] for capture in captures]
    same_features = all(comparisons[key]["exact_clip_features_equal"]
                        and comparisons[key]["exact_window_features_equal"]
                        for key in ("fresh_vs_after_conditioning", "fresh_vs_restarted"))
    counters = [row.get("reported_metrics", {}).get("timings.cache_n") for rows in captures for row in rows]
    return {"comparisons": comparisons, "sampling_identity_equal": not any(any(row.values()) for row in identities),
            "actual_request_bodies_equal": hashes[0] == hashes[1] == hashes[2],
            "normalized_request_bodies_equal": normalized[0] == normalized[1] == normalized[2],
            "new_replays_numerically_stable": same_features,
            "zero_cache_n_requests": sum(value == 0 for value in counters),
            "positive_cache_n_requests": sum(value is not None and value > 0 for value in counters),
            "missing_cache_n_requests": sum(value is None for value in counters),
            "target_request_elapsed_sec": sum(row["elapsed_sec"] for rows in captures for row in rows)}


def diagnose(run_dir, protocol_path=None, *, execute=False, output_dir=None):
    run = Path(run_dir).resolve()
    prepared = replay._prepare(run, protocol_path or config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    source_hashes = {str(Path(module.__file__).relative_to(config.ROOT)): replay.digest(module.__file__)
                     for module in (replay, transport, metrics)}
    provenance = {**prepared["public_provenance"], "source_sha256": replay.digest(__file__),
                  "diagnostic_helpers": source_hashes}
    result = {"purpose": "spatial_cache_history_only", "execute": execute, "provenance": provenance,
              "target_requests_attempted": 0, "conditioning_requests_attempted": 0,
              "selected_development_groups": 1, "holdout_groups_used": 0, "selection_uses_labels": False,
              "promotion_authorized": False, "semantic_accuracy_verified": False,
              "limitations": ["One fixed development recording diagnoses numeric replay, not semantic accuracy",
                              "Fixed policy order and one conditioning request do not establish exclusive causality",
                              "Missing cache counters cannot establish cache state; original failed reports remain frozen"]}
    if not execute:
        return result
    output = Path(output_dir or run / "cache-policy-diagnostic").resolve()
    replay._require(output.is_relative_to(config.DATA_PROC.resolve())
                    and not output.is_relative_to(run / "cache") and not output.is_relative_to(run / "models"),
                    "Cache diagnostic requires a new derived output directory")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    descriptor = os.open(output / "source_snapshot.py", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(Path(__file__).read_bytes())
    replay._private(output / "selection.joblib", {key: prepared[key] for key in (
        "chosen", "cache_sha256", "original_failure_sha256")})
    budget, summaries, normalized_all = {"target": 0, "conditioning": 0}, {}, []
    conditioning_captures = []
    stage, started = "sampling_probe", time.monotonic()
    try:
        sample = replay._decode_probe(prepared["source"], prepared["cached"])
        replay._private(output / "sampling_probe.joblib", sample)
        replay._require(not any(replay.identity_difference(prepared["cached"], sample).values()),
                        "Fixed source sampling differs from historical cache")
        model = verify_ready()
        conditioning_jpeg = _conditioning_jpeg()
        launch = command(config.ROOT, model)
        replay._private(output / "server_command.joblib", launch)
        result["server_command_sha256"] = replay._fingerprint(launch)
        for policy in (True, False):
            label, records, captures = str(policy).lower(), [], []

            def extract(phase):
                nonlocal stage
                stage = label + "_" + phase
                try:
                    with policy_requests(policy, budget) as rows:
                        record = engine.extract_video(prepared["source"], deadline=time.monotonic() + 240)
                    replay._validate_record(record)
                    replay._require(len(rows) == 4 and all(row.get("completed") for row in rows),
                                    "Frozen target extraction did not make four completed requests")
                    replay._private(output / (stage + ".joblib"), record)
                    replay._require(not any(replay.identity_difference(prepared["cached"], record).values()),
                                    "Frozen target extraction sample identity changed")
                    records.append(record)
                    captures.append(rows)
                finally:
                    if "rows" in locals():
                        replay._private(output / (stage + "_requests.joblib"), rows)

            with original_server(model):
                extract(PHASES[0])
                stage = label + "_conditioning"
                try:
                    with policy_requests(policy, budget, conditioning=True) as rows:
                        _condition(conditioning_jpeg)
                    replay._require(len(rows) == 1 and rows[0].get("completed"), "Conditioning call did not complete")
                    conditioning_captures.append(rows[0])
                finally:
                    replay._private(output / (stage + "_requests.joblib"), rows)
                extract(PHASES[1])
            with original_server(model):
                extract(PHASES[2])
            summaries[label] = _summary(records, captures, prepared["cached"], prepared["model"])
            normalized_all.extend([[row["normalized_body_sha256"] for row in rows] for rows in captures])
        result["policies"] = summaries
        result["conditioning_cache_exposure"] = _cache_exposure(conditioning_captures)
        result["conditioning_normalized_requests_equal"] = (
            conditioning_captures[0]["normalized_body_sha256"] == conditioning_captures[1]["normalized_body_sha256"])
        replay._require(result["conditioning_normalized_requests_equal"], "Normalized conditioning request changed")
        result["normalized_requests_equal_across_all_replays"] = all(value == normalized_all[0] for value in normalized_all)
        replay._require(result["normalized_requests_equal_across_all_replays"], "Target normalized request bodies changed")
        stable_on, stable_off = (summaries[key]["new_replays_numerically_stable"] for key in ("true", "false"))
        result["interpretation"] = ("supports_history_sensitivity" if stable_off and not stable_on else
                                    "both_stable_inconclusive" if stable_off else "cache_off_drift_insufficient")
        result["diagnostic_complete"] = True
    except Exception:
        result.update(diagnostic_complete=False, failure_stage=stage, failure_details_suppressed=True)
    result.update(target_requests_attempted=budget["target"], conditioning_requests_attempted=budget["conditioning"],
                  conditioning_cache_exposure=_cache_exposure(conditioning_captures),
                  elapsed_sec=time.monotonic() - started)
    failure = prepared["original_failure"]
    preserved = ((replay.digest(failure) if failure.exists() else None) == prepared["original_failure_sha256"])
    result["original_failed_report_preserved"] = preserved
    sources_ok = (replay.digest(__file__) == provenance["source_sha256"] and all(
        replay.digest(config.ROOT / path) == sha for path, sha in source_hashes.items()))
    engine._check_inherited_sources()
    result["source_hashes_preserved"] = sources_ok
    artifacts_ok = (replay.digest(prepared["source"]) == prepared["chosen"]["group_id"]
                    and replay.digest(run / "cache" / (prepared["chosen"]["group_id"] + ".joblib"))
                    == prepared["cache_sha256"]
                    and replay.digest(run / "models/baseline.joblib")
                    == prepared["public_provenance"]["classifier_sha256"]
                    and engine.get_encoder_provenance() == {key: prepared["public_provenance"][key]
                                                           for key in engine.get_encoder_provenance()})
    result["frozen_artifact_hashes_preserved"] = artifacts_ok
    if not preserved or not sources_ok or not artifacts_ok:
        result["diagnostic_complete"] = False
    replay._private(output / "cache_policy_diagnostic.json", result, json_output=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "video-qwen-spatial-v3")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = diagnose(args.run_dir, args.protocol, execute=args.run, output_dir=args.output_dir)
    except Exception:
        raise SystemExit("Private cache/history diagnostic failed; no model text or identifiers emitted") from None
    print(json.dumps(result), flush=True)
    if result.get("diagnostic_complete") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
