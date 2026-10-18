import base64
from contextlib import contextmanager
import json

import cv2
import numpy as np
import pytest

from scripts import check_native_frame_order as probe


@pytest.mark.parametrize("order", probe.ORDERS)
def test_encoded_four_frames_preserve_every_color_and_timing(tmp_path, order):
    path = tmp_path / "synthetic.avi"
    payload, hashes = probe.encode_order(path, order)
    assert payload.startswith(b"RIFF") and len(hashes) == 4
    assert hashes[0] == hashes[1] and hashes[2] == hashes[3] and hashes[0] != hashes[-1]
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == 2 and cap.get(cv2.CAP_PROP_FRAME_COUNT) == 4
        for color in order:
            ok, frame = cap.read()
            assert ok and frame.shape == (256, 256, 3)
            means = frame.mean(axis=(0, 1))
            assert means[2 if color == "red" else 0] > 250
            assert means[0 if color == "red" else 2] < 5 and means[1] < 5
        assert not cap.read()[0]
    finally:
        cap.release()


def test_fixed_recipe_and_exact_schema():
    body = probe.request_body("verified", b"synthetic")
    assert (body["temperature"], body["seed"], body["cache_prompt"], body["max_tokens"]) == (0, 42, False, 32)
    assert body["response_format"]["schema"] == probe.SCHEMA
    assert body["messages"][0]["content"] == [{"type": "text", "text": probe.PROMPT},
        {"type": "input_video", "input_video": {"data": base64.b64encode(b"synthetic").decode()}}]
    for value in ({"first_color": "red", "last_color": "blue"},
                  {"first_color": "unclear", "last_color": "unclear"}):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}
        assert probe.parse_colors(response) == value
    for bad in ({"first_color": "red"}, {"first_color": "red", "last_color": "green"},
                {"first_color": "red", "last_color": "blue", "extra": 1}):
        with pytest.raises(ValueError, match="first/last schema"):
            probe.parse_colors({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(bad)}}]})


def test_exactly_two_sequential_calls_on_one_owned_server(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.config, "DATA_PROC", tmp_path)
    monkeypatch.setattr(probe, "verify_model", lambda: {"alias": "verified"})
    monkeypatch.setattr(probe, "verify_decoder", lambda: {"bin_dir": "verified"})
    monkeypatch.setattr(probe.native, "protocol_provenance", lambda *args: {"runtime": "pinned"})
    monkeypatch.setattr(probe.native, "server_command", lambda *args: ["verified-server"])
    ownership, calls = [], []

    @contextmanager
    def server(*args):
        ownership.append("start")
        yield
        ownership.append("stop")

    def request(path, body, timeout):
        assert ownership == ["start"] and timeout == 45 and path == "/v1/chat/completions"
        order = probe.ORDERS[len(calls)]
        calls.append(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
            {"first_color": order[0], "last_color": order[-1]})}}], "timings": {"cache_n": 0}}

    monkeypatch.setattr(probe.native, "native_server", server)
    monkeypatch.setattr(probe.native, "request_json", request)
    result = probe.run_probe(tmp_path / "fresh")
    assert ownership == ["start", "stop"] and len(calls) == 2 and result["passed"]
    assert result["bank_videos_used"] == result["holdout_videos_used"] == 0
    assert result["authorizes_bank_video_batch"] is False
    assert result["cases"][0]["payload_sha256"] != result["cases"][1]["payload_sha256"]
    assert all(case["reported_metrics"] == {"timings.cache_n": 0} for case in result["cases"])
    assert json.loads((tmp_path / "fresh/frame_order.json").read_text()) == result
    with pytest.raises(FileExistsError):
        probe.run_probe(tmp_path / "fresh")


def test_default_inspection_does_not_start_model(monkeypatch, capsys):
    monkeypatch.setattr(probe, "run_probe", lambda *args: pytest.fail("Unexpected inference"))
    probe.main([])
    assert json.loads(capsys.readouterr().out)["will_execute"] is False
