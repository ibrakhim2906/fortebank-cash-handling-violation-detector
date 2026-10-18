"""Synthetic/mocked only; no supplied footage or actual model service."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import check_spatial_cache_policy as probe


def observation(state="none"):
    value = {key: states[0] for key, states in probe.engine.FIELDS.items()}
    value["people_count"] = state
    return value


def record(observations=None):
    observations = observations or [observation()] * 4
    windows, audit = [], []
    for index, value in enumerate(observations):
        vector, _ = probe.engine.observation_features(value)
        windows.append({"timestamp_sec": float(index), "features": vector})
        audit.append({"requested_sec": float(index), "actual_sec": float(index),
                      "source_frame_index": index, "timestamp_source": "synthetic", "source_shape": [64, 64, 3],
                      "source_frame_sha256": "private-frame-" + str(index),
                      "views": [{"sha256": "private-view-" + str(index)}], "observation": value})
    matrix = np.stack([row["features"] for row in windows])
    summaries = {"max": matrix.max(axis=0), "mean": matrix.mean(axis=0), "std": matrix.std(axis=0)}
    return {"clip_features": np.concatenate([summaries[key] for key in probe.engine.CLIP_STATISTICS]),
            "windows": windows, "observation_audit": audit, "sampling": {"feature_config": probe.engine.FEATURE_CONFIG}}


class Classifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, values):
        probability = .1 + .8 * values[:, 0]
        return np.column_stack([1 - probability, probability])


def envelope(value):
    return {"choices": [{"message": {"content": json.dumps(value)}}],
            "timings": {"cache_n": 0}, "usage": {"prompt_tokens": 7}}


def test_only_cache_key_changes_and_frozen_parsing_and_builder_restore(monkeypatch):
    client, calls, budget = probe.engine._observations, [], {"target": 0, "conditioning": 0}
    original = client.build_opener

    def request(path, body, timeout):
        assert path == "/v1/chat/completions" and timeout == 12
        calls.append(body)
        return envelope(observation())

    monkeypatch.setattr(probe.transport, "request_json", request)
    captures = []
    for policy in (True, False):
        with probe.policy_requests(policy, budget) as rows:
            assert client.request_observations([b"\xff\xd8synthetic JPEG"], timeout=12) == observation()
        captures.append(rows[0])
        assert client.build_opener is original
    assert calls[0]["cache_prompt"] is True and calls[1]["cache_prompt"] is False
    assert {k: v for k, v in calls[0].items() if k != "cache_prompt"} == {
        k: v for k, v in calls[1].items() if k != "cache_prompt"}
    assert captures[0]["normalized_body_sha256"] == captures[1]["normalized_body_sha256"]
    assert captures[0]["body_sha256"] != captures[1]["body_sha256"]
    assert captures[0]["reported_metrics"] == {"usage.prompt_tokens": 7, "timings.cache_n": 0}
    assert budget == {"target": 2, "conditioning": 0}


def test_budget_rejects_before_transport_and_exception_restores_builder(monkeypatch):
    client, budget = probe.engine._observations, {"target": 24, "conditioning": 2}
    original = client.build_opener
    monkeypatch.setattr(probe.transport, "request_json", lambda *a, **k: pytest.fail("Budget must stop request"))
    for conditioning in (False, True):
        with pytest.raises(RuntimeError, match="structured observations"):
            with probe.policy_requests(False, budget, conditioning=conditioning):
                client.request_observations([b"\xff\xd8synthetic JPEG"])
        assert client.build_opener is original
    assert budget == {"target": 24, "conditioning": 2}


def test_original_server_cleanup_and_profile_do_not_add_native_flags(monkeypatch):
    class Guard:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, endpoint):
            assert endpoint == ("127.0.0.1", 8003)

    class Process:
        def poll(self):
            return None

    process, launches, stopped = Process(), [], []
    monkeypatch.setattr(probe.socket, "socket", Guard)
    monkeypatch.setattr(probe, "command", lambda *args: ["/tmp/local/llama-server", "--parallel", "1"])
    monkeypatch.setattr(probe.subprocess, "Popen", lambda args, **kwargs: launches.append((args, kwargs)) or process)
    monkeypatch.setattr(probe.transport, "request_json", lambda path, **kwargs:
                        {"data": [{"id": "verified"}]} if path == "/v1/models" else {})
    monkeypatch.setattr(probe.transport, "stop_server", lambda proc: stopped.append(proc))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with probe.original_server({"alias": "verified"}):
            raise RuntimeError("synthetic failure")
    assert launches[0][0] == ["/tmp/local/llama-server", "--parallel", "1"] and stopped == [process]
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][1]["env"]["LLAMA_LOG_DISABLE"] == "1"


def test_aggregate_comparison_reports_only_numeric_changes_and_flips():
    same, changed = record(), record([observation("one")] * 4)
    captures = [[{"body_sha256": "same", "normalized_body_sha256": "same", "elapsed_sec": .1,
                  "reported_metrics": {"timings.cache_n": 0}}] * 4] * 3
    summary = probe._summary([same, changed, same], captures, same,
                             {"clip_model": Classifier(), "window_model": Classifier()})
    assert not summary["new_replays_numerically_stable"] and summary["sampling_identity_equal"]
    assert summary["comparisons"]["fresh_vs_after_conditioning"]["classification_threshold_flips"] == 1
    assert summary["zero_cache_n_requests"] == 12 and summary["missing_cache_n_requests"] == 0
    assert "private-frame" not in json.dumps(summary) and "probability" not in json.dumps(summary)


@pytest.fixture
def setup_run(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.config, "DATA_PROC", tmp_path / "processed")
    run = probe.config.DATA_PROC / "frozen"
    (run / "models").mkdir(parents=True)
    (run / "cache").mkdir()
    source = run / "PRIVATE-source.mov"
    source.write_bytes(b"synthetic placeholder, not media")
    group = probe.replay.digest(source)
    cache = run / "cache" / (group + ".joblib")
    cache.write_bytes(b"synthetic cache bytes")
    classifier = run / "models/baseline.joblib"
    classifier.write_bytes(b"synthetic classifier bytes")
    failed = run / "inference_replay_check.json"
    failed.write_text("original private failed report")
    cached = record()
    prepared = {"source": source, "chosen": {"group_id": group}, "cached": cached,
                "model": {"clip_model": Classifier(), "window_model": Classifier()},
                "cache_sha256": probe.replay.digest(cache), "original_failure": failed,
                "original_failure_sha256": probe.replay.digest(failed),
                "public_provenance": {**probe.engine.get_encoder_provenance(),
                    "classifier_sha256": probe.replay.digest(classifier)}}
    monkeypatch.setattr(probe.replay, "_prepare", lambda *args: prepared)
    monkeypatch.setattr(probe.replay, "_decode_probe", lambda *args: copy.deepcopy(cached))
    monkeypatch.setattr(probe, "verify_ready", lambda: {"alias": "verified"})
    monkeypatch.setattr(probe, "command", lambda *args: ["/tmp/original-profile"])
    return run, prepared


def test_inspection_never_decodes_starts_process_or_calls_model(setup_run, monkeypatch):
    run, _ = setup_run
    monkeypatch.setattr(probe.replay, "_decode_probe", lambda *a: pytest.fail("Inspection decode"))
    monkeypatch.setattr(probe, "original_server", lambda *a: pytest.fail("Inspection server"))
    result = probe.diagnose(run)
    assert not result["execute"] and result["target_requests_attempted"] == 0
    assert not (run / "cache-policy-diagnostic").exists()


def test_full_bounded_flow_supports_history_hypothesis_without_promotion(setup_run, monkeypatch):
    run, prepared = setup_run
    state, actual_calls, lifecycle = {"server": 0, "within_server": 0}, [], []

    @contextmanager
    def server(*args):
        state["server"] += 1
        state["within_server"] = 0
        lifecycle.append("start")
        yield
        lifecycle.append("stop")

    def request(path, body, timeout):
        assert lifecycle[-1] == "start" and path == "/v1/chat/completions" and 0 < timeout <= 45
        state["within_server"] += 1
        actual_calls.append(body)
        changed = body["cache_prompt"] and state["server"] == 1 and state["within_server"] > 5
        return envelope(observation("one" if changed else "none"))

    def extract(source, deadline):
        assert source == prepared["source"]
        values = [probe.engine._observations.request_observations([b"\xff\xd8synthetic JPEG"] * 3) for _ in range(4)]
        return record(values)

    monkeypatch.setattr(probe, "original_server", server)
    monkeypatch.setattr(probe.transport, "request_json", request)
    monkeypatch.setattr(probe.engine, "extract_video", extract)
    original_failure = prepared["original_failure"].read_bytes()
    result = probe.diagnose(run, execute=True, output_dir=run / "diagnostic")
    assert result["diagnostic_complete"] and result["interpretation"] == "supports_history_sensitivity"
    assert result["target_requests_attempted"] == 24 and result["conditioning_requests_attempted"] == 2
    assert len(actual_calls) == 26 and lifecycle == ["start", "stop"] * 4
    assert all(body["cache_prompt"] for body in actual_calls[:13])
    assert all(not body["cache_prompt"] for body in actual_calls[13:])
    assert result["normalized_requests_equal_across_all_replays"]
    assert result["conditioning_normalized_requests_equal"]
    assert result["conditioning_cache_exposure"] == {
        "zero_cache_n_requests": 2, "positive_cache_n_requests": 0, "missing_cache_n_requests": 0}
    assert result["original_failed_report_preserved"] and result["frozen_artifact_hashes_preserved"]
    assert not result["promotion_authorized"] and not result["semantic_accuracy_verified"]
    assert prepared["original_failure"].read_bytes() == original_failure
    output = run / "diagnostic"
    assert os.stat(output).st_mode & 0o777 == 0o700
    assert all(os.stat(path).st_mode & 0o777 == 0o600 for path in output.iterdir())
    assert "PRIVATE-source" not in json.dumps(result) and prepared["chosen"]["group_id"] not in json.dumps(result)
    with pytest.raises(FileExistsError):
        probe.diagnose(run, execute=True, output_dir=output)


def test_sampling_difference_blocks_all_model_calls(setup_run, monkeypatch):
    run, prepared = setup_run
    changed = copy.deepcopy(prepared["cached"])
    changed["observation_audit"][0]["source_frame_sha256"] = "changed"
    monkeypatch.setattr(probe.replay, "_decode_probe", lambda *args: changed)
    monkeypatch.setattr(probe, "original_server", lambda *args: pytest.fail("Sampling mismatch must stop server start"))
    result = probe.diagnose(run, execute=True, output_dir=run / "diagnostic")
    assert result["diagnostic_complete"] is False
    assert result["target_requests_attempted"] == result["conditioning_requests_attempted"] == 0
    assert result["failure_stage"] == "sampling_probe"
