"""Synthetic mocked transport checks; no model, media decoder or assets run."""
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import pytest

from scripts import check_local_cache_behavior as check


def response(motion, cache_n=0):
    value = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"motion": motion})}}]}
    if cache_n != "missing":
        value["timings"] = {"cache_n": cache_n}
    return value


@pytest.fixture
def environment(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    monkeypatch.setattr(check.config, "DATA_PROC", processed)
    model = {"alias": "local-action-review", "weights_sha256": "weights", "projector_sha256": "projector",
             "runtime_release": "runtime"}
    decoder = {"bin_dir": Path("/synthetic/decoder"), "archive_sha256": "decoder"}
    monkeypatch.setattr(check, "verify_model", lambda: model)
    monkeypatch.setattr(check, "verify_decoder", lambda: decoder)
    monkeypatch.setattr(check.native, "server_command", lambda *_: ["synthetic-server", "--offline"])
    state = {"requests": [], "encoded": [], "server_entries": 0, "server_exits": 0}

    @contextmanager
    def server(*_):
        state["server_entries"] += 1
        try:
            yield None
        finally:
            state["server_exits"] += 1

    def encode(_path, motion):
        state["encoded"].append(motion)
        return b"RIFFsynthetic-" + motion.encode()

    def request(path, body, *, timeout):
        state["requests"].append((path, body, timeout))
        return response(check.CONTROLS[len(state["requests"]) - 1])

    monkeypatch.setattr(check.native, "native_server", server)
    monkeypatch.setattr(check.native, "encode_video", encode)
    monkeypatch.setattr(check.native, "request_json", request)
    state["run_dir"] = processed / "cache-diagnostic"
    return state


def test_six_fixed_controls_only_policy_change_and_payload_reuse(environment):
    result = check.run_probe(environment["run_dir"])
    assert result["passed"] and result["repeat_consistent"]
    assert [case["expected"] for case in result["cases"]] == list(check.CONTROLS)
    assert [case["pass_index"] for case in result["cases"]] == [1, 1, 1, 2, 2, 2]
    assert environment["encoded"] == list(check.native.MOTIONS[:3])
    assert environment["server_entries"] == environment["server_exits"] == 1
    for index, (path, body, timeout) in enumerate(environment["requests"]):
        payload = b"RIFFsynthetic-" + check.CONTROLS[index].encode()
        original_body = {"model": "local-action-review", "temperature": 0, "seed": check.config.SEED,
                         "max_tokens": 32, "response_format": {"type": "json_object", "schema": check.native.SCHEMA},
                         "messages": [{"role": "user", "content": [
                             {"type": "text", "text": check.native.PROMPT},
                             {"type": "input_video", "input_video": {"data": base64.b64encode(payload).decode()}},
                         ]}]}
        assert path == "/v1/chat/completions" and timeout == 45
        assert body == {**original_body, "cache_prompt": False}
    assert environment["requests"][:3] == environment["requests"][3:]


def test_zero_reported_cache_count_is_separate_from_semantics(environment):
    result = check.run_probe(environment["run_dir"])
    assert result["cache_metadata_complete"]
    assert result["reported_zero_cache_n"] == 6 and result["reported_positive_cache_n"] == 0
    assert result["missing_or_invalid_cache_n"] == 0
    assert result["bank_videos_used"] == result["holdout_videos_used"] == 0
    assert result["authorizes_bank_video_batch"] is result["bank_action_accuracy_verified"] is False


def test_constant_valid_responses_fail(environment, monkeypatch):
    monkeypatch.setattr(check.native, "request_json", lambda *_args, **_kwargs: response("left_to_right"))
    result = check.run_probe(environment["run_dir"])
    assert not result["passed"] and result["repeat_consistent"]
    assert result["operational_failure"] is None and sum(c["correct"] for c in result["cases"]) == 2


@pytest.mark.parametrize("count", ["missing", -1, True, 1.5])
def test_missing_or_invalid_timings_do_not_establish_cache_state(environment, monkeypatch, count):
    calls = iter(check.CONTROLS)
    monkeypatch.setattr(check.native, "request_json", lambda *_args, **_kwargs: response(next(calls), count))
    result = check.run_probe(environment["run_dir"])
    assert result["passed"] and not result["cache_metadata_complete"]
    assert result["missing_or_invalid_cache_n"] == 6
    assert all(case["cache_n"] is None for case in result["cases"])


def test_positive_cache_count_recorded_without_masking_behavior(environment, monkeypatch):
    calls = iter(check.CONTROLS)
    monkeypatch.setattr(check.native, "request_json", lambda *_args, **_kwargs: response(next(calls), 10))
    result = check.run_probe(environment["run_dir"])
    assert result["passed"] and result["cache_metadata_complete"]
    assert result["reported_positive_cache_n"] == 6


def test_repeat_disagreement_fails(environment, monkeypatch):
    calls = iter((*check.CONTROLS[:3], "unclear", *check.CONTROLS[4:]))
    monkeypatch.setattr(check.native, "request_json", lambda *_args, **_kwargs: response(next(calls)))
    result = check.run_probe(environment["run_dir"])
    assert not result["passed"] and not result["repeat_consistent"]


def test_bad_response_stops_without_payload_logging_and_reaps_owned_server(environment, monkeypatch):
    monkeypatch.setattr(check.native, "request_json", lambda *_args, **_kwargs: {"secret": "raw-response-never-logged"})
    result = check.run_probe(environment["run_dir"])
    assert not result["passed"] and not result["cases"]
    assert result["operational_failure"] == {"stage": "synthetic_1_left_to_right", "exception_type": "ValueError"}
    assert "raw-response-never-logged" not in json.dumps(result)
    assert environment["server_exits"] == 1


def test_helper_drift_refused_before_server_or_output(environment, monkeypatch):
    monkeypatch.setattr(check, "PINNED_HELPER_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="controlled baseline"):
        check.run_probe(environment["run_dir"])
    assert environment["server_entries"] == 0 and not environment["run_dir"].exists()


def test_existing_output_is_preserved(environment):
    directory = environment["run_dir"]
    directory.mkdir(parents=True)
    preserved = directory / "cache_behavior.json"
    preserved.write_text("preserve")
    with pytest.raises(FileExistsError):
        check.run_probe(directory)
    assert preserved.read_text() == "preserve" and environment["server_entries"] == 0


def test_outside_processed_and_escaping_symlink_rejected(environment, tmp_path):
    with pytest.raises(ValueError, match="data/processed"):
        check.run_probe(tmp_path / "raw")
    environment["run_dir"].parent.mkdir()
    (environment["run_dir"].parent / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="data/processed"):
        check.run_probe(environment["run_dir"].parent / "escape" / "new")
    assert environment["server_entries"] == 0


def test_source_and_provenance_snapshots_match_report(environment):
    result = check.run_probe(environment["run_dir"])
    directory = environment["run_dir"]
    assert hashlib.sha256((directory / "native_helper_source_snapshot.py").read_bytes()).hexdigest() == check.PINNED_HELPER_SHA256
    assert hashlib.sha256((directory / "source_snapshot.py").read_bytes()).hexdigest() == result["provenance"]["source_sha256"]
    assert json.loads((directory / "provenance.json").read_text()) == result["provenance"]
    assert json.loads((directory / "cache_behavior.json").read_text()) == result
    assert not list(directory.glob("*.avi"))


def test_default_main_is_inspection_only(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Inspection must not execute a probe")
    monkeypatch.setattr(check, "run_probe", forbidden)
    check.main([])
    assert json.loads(capsys.readouterr().out) == {"will_execute": False, "synthetic_cases": 6,
                                                "cache_prompt": False, "bank_videos_used": 0,
                                                "authorizes_bank_video_batch": False}
