import base64
from contextlib import contextmanager
import hashlib
import json

import cv2
import numpy as np
import pytest

from scripts import check_native_visual_input as probe


def response(color, **metadata):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"color": color})}}],
            **metadata}


@pytest.mark.parametrize("color,channel", [("red", 2), ("blue", 0)])
def test_encoded_controls_have_matching_flat_color_and_four_frames(tmp_path, color, channel):
    payloads, source_hash = probe.encode_color(tmp_path, color)
    assert payloads["native_video"].startswith(b"RIFF")
    jpeg = cv2.imdecode(np.frombuffer(payloads["single_jpeg"], np.uint8), cv2.IMREAD_COLOR)
    source = np.zeros((256, 256, 3), np.uint8)
    source[:, :, channel] = 255
    assert source_hash == hashlib.sha256(source.tobytes()).hexdigest()
    assert np.max(np.abs(jpeg.astype(int) - source)) <= 5
    cap = cv2.VideoCapture(str(tmp_path / (color + ".avi")))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == 2
        for _ in range(4):
            ok, frame = cap.read()
            assert ok and np.max(np.abs(frame.astype(int) - source)) <= 5
        assert not cap.read()[0]
    finally:
        cap.release()


@pytest.mark.parametrize("kind", ["native_video", "single_jpeg"])
def test_request_uses_fixed_color_schema_and_cache_off(kind):
    body = probe.request_body("verified-alias", kind, b"synthetic")
    assert body["model"] == "verified-alias"
    assert (body["temperature"], body["seed"], body["max_tokens"], body["cache_prompt"]) == (0, 42, 32, False)
    assert body["response_format"]["schema"] == probe.SCHEMA
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": probe.PROMPT}
    media = content[1]
    if kind == "native_video":
        assert media == {"type": "input_video", "input_video": {"data": base64.b64encode(b"synthetic").decode()}}
    else:
        assert media["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"synthetic").decode()


def test_unknown_input_kind_fails():
    with pytest.raises(ValueError, match="input kind"):
        probe.request_body("alias", "remote_url", b"synthetic")


@pytest.mark.parametrize("color", probe.COLORS)
def test_exact_color_enum_parses(color):
    assert probe.parse_color(response(color)) == color


@pytest.mark.parametrize("bad", [response("green"), {"choices": []},
    {"choices": [{"finish_reason": "length", "message": {"content": '{"color":"red"}'}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": '{"color":"red","extra":1}'}}]}])
def test_malformed_and_truncated_responses_fail(bad):
    with pytest.raises(ValueError, match="fixed color schema"):
        probe.parse_color(bad)


def test_optional_metrics_are_numeric_only_and_missing_stay_missing():
    assert probe.reported_metrics({}) == {}
    metrics = probe.reported_metrics({"usage": {"prompt_tokens": 12, "image_tokens": True,
        "video_tokens": -1, "completion_tokens": float("inf"), "unrelated": 99,
        "prompt_tokens_details": {"video_tokens": 8, "cached_tokens": 0, "blob": "secret"}},
        "timings": {"cache_n": 0, "prompt_ms": 123.5, "predicted_ms": float("nan")}})
    assert metrics == {"usage.prompt_tokens": 12, "usage.prompt_tokens_details.video_tokens": 8,
        "usage.prompt_tokens_details.cached_tokens": 0, "timings.cache_n": 0, "timings.prompt_ms": 123.5}


def mocked_probe(monkeypatch, tmp_path, answers):
    monkeypatch.setattr(probe.config, "DATA_PROC", tmp_path)
    monkeypatch.setattr(probe, "verify_model", lambda: {"alias": "verified-alias"})
    monkeypatch.setattr(probe, "verify_decoder", lambda: {"bin_dir": "verified"})
    monkeypatch.setattr(probe.native, "protocol_provenance", lambda *args: {"runtime_release": "pinned"})
    monkeypatch.setattr(probe.native, "server_command", lambda *args: ["local-server"])
    calls, ownership = [], []

    @contextmanager
    def server(*args):
        ownership.append("start")
        try:
            yield
        finally:
            ownership.append("stop")

    def request(path, body, timeout):
        assert ownership == ["start"] and path == "/v1/chat/completions" and timeout == 45
        calls.append(body)
        return response(answers[len(calls) - 1], timings={"cache_n": 0})

    monkeypatch.setattr(probe.native, "native_server", server)
    monkeypatch.setattr(probe.native, "request_json", request)
    return calls, ownership


def test_four_calls_share_one_verified_server_and_preserve_provenance(monkeypatch, tmp_path):
    calls, ownership = mocked_probe(monkeypatch, tmp_path, ["red", "blue", "red", "blue"])
    result = probe.run_probe(tmp_path / "fresh")
    assert len(calls) == 4 and ownership == ["start", "stop"]
    assert result["controls_passed"] == {"native_video": True, "single_jpeg": True}
    assert result["complete"] and result["operational_failure"] is None
    assert result["bank_videos_used"] == result["holdout_videos_used"] == 0
    assert result["authorizes_bank_video_batch"] is False
    assert [(row["input_kind"], row["expected"]) for row in result["cases"]] == list(probe.CONTROLS)
    assert result["cases"][0]["source_frame_sha256"] == result["cases"][2]["source_frame_sha256"]
    assert result["cases"][0]["payload_sha256"] != result["cases"][2]["payload_sha256"]
    assert result["cases"][0]["request_body_sha256"] == hashlib.sha256(json.dumps(calls[0]).encode()).hexdigest()
    assert json.loads((tmp_path / "fresh/visual_input.json").read_text()) == result


def test_native_and_image_color_passes_are_separate(monkeypatch, tmp_path):
    mocked_probe(monkeypatch, tmp_path, ["unclear", "unclear", "red", "blue"])
    result = probe.run_probe(tmp_path / "fresh")
    assert result["complete"] and result["operational_failure"] is None
    assert result["controls_passed"] == {"native_video": False, "single_jpeg": True}


def test_existing_output_is_preserved_and_assets_not_touched(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.config, "DATA_PROC", tmp_path)
    monkeypatch.setattr(probe, "verify_model", lambda: pytest.fail("Existing output must fail first"))
    sentinel = tmp_path / "existing.json"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError, match="fresh"):
        probe.run_probe(tmp_path)
    assert sentinel.read_text() == "preserve"


def test_default_inspection_never_verifies_assets_or_calls_model(monkeypatch, capsys):
    monkeypatch.setattr(probe, "run_probe", lambda *args: pytest.fail("Inspection must not execute"))
    probe.main([])
    assert json.loads(capsys.readouterr().out)["will_execute"] is False


def test_request_failure_stops_owned_server_and_emits_no_error_payload(monkeypatch, tmp_path):
    calls, ownership = mocked_probe(monkeypatch, tmp_path, [])

    def fail(*args, **kwargs):
        raise TimeoutError("untrusted response payload must not be recorded")

    monkeypatch.setattr(probe.native, "request_json", fail)
    result = probe.run_probe(tmp_path / "fresh")
    assert not calls and ownership == ["start", "stop"]
    assert not result["complete"] and not any(result["controls_passed"].values())
    assert result["operational_failure"] == {"stage": "native_video_red", "exception_type": "TimeoutError"}
    assert "untrusted response" not in json.dumps(result)
