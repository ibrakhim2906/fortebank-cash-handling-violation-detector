import json
import os
import signal
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest

from scripts import check_temporal_protocol as probe


@pytest.mark.parametrize("motion,direction", [("left_to_right", 1), ("right_to_left", -1), ("stationary", 0)])
def test_encoded_controls_preserve_known_motion_and_timing(tmp_path, motion, direction):
    path = tmp_path / "control.avi"
    payload = probe.encode_video(path, motion)
    assert payload.startswith(b"RIFF")
    cap = cv2.VideoCapture(str(path))
    centers = []
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == 2
        for _ in range(4):
            ok, frame = cap.read()
            assert ok
            mask = (frame[:, :, 2] > 150) & (frame[:, :, 1] < 100) & (frame[:, :, 0] < 100)
            centers.append(float(np.nonzero(mask)[1].mean()))
        assert not cap.read()[0]
    finally:
        cap.release()
    differences = np.diff(centers)
    assert np.all(differences > 40) if direction == 1 else np.all(differences < -40) if direction == -1 else np.all(abs(differences) < 1)


@pytest.mark.parametrize("motion", probe.MOTIONS)
def test_response_requires_one_complete_schema(motion):
    response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"motion": motion})}}]}
    assert probe.parse_motion(response) == motion


@pytest.mark.parametrize("response", [
    {}, {"choices": []}, {"choices": [1]},
    {"choices": [{"finish_reason": "length", "message": {"content": '{"motion":"stationary"}'}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": 'not JSON'}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": '{"motion":"jump"}'}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": '{"motion":"stationary","extra":1}'}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": 'null'}}]},
])
def test_invalid_or_truncated_outputs_fail(response):
    with pytest.raises(ValueError, match="fixed motion schema"):
        probe.parse_motion(response)


def test_unknown_endpoint_never_opens_network(monkeypatch):
    monkeypatch.setattr(probe, "build_opener", lambda *a: pytest.fail("Unexpected network setup"))
    with pytest.raises(ValueError, match="endpoint"):
        probe.request_json("http://example.invalid/video")


def test_redirects_are_rejected():
    with pytest.raises(ValueError, match="cannot redirect"):
        probe.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.invalid")


class HttpResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        assert limit == probe.MAX_RESPONSE_BYTES + 1
        return self.payload[:limit]


def test_worker_http_path_is_literal_loopback_without_proxies_and_returns_dict(monkeypatch):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            return HttpResponse(b'{"valid":true}')

    def opener(*handlers):
        assert handlers[0].proxies == {}
        assert isinstance(handlers[1], probe.NoRedirect)
        return Opener()

    monkeypatch.setattr(probe, "build_opener", opener)
    assert probe._request_json_http("/v1/chat/completions", {"synthetic": "request"}, timeout=2) == {"valid": True}
    request, timeout = calls[0]
    assert request.full_url == "http://127.0.0.1:8003/v1/chat/completions" and timeout == 2
    assert json.loads(request.data) == {"synthetic": "request"}


@pytest.mark.parametrize("payload", [b"x" * 65537, b"not-json", b"[]"])
def test_worker_http_response_bound_and_schema_are_enforced(monkeypatch, payload):
    class Opener:
        def open(self, *args, **kwargs):
            return HttpResponse(payload)

    monkeypatch.setattr(probe, "build_opener", lambda *args: Opener())
    with pytest.raises(ValueError):
        probe._request_json_http("/health")


def test_worker_suppresses_output_and_sanitizes_exception(monkeypatch):
    writes, redirects = [], []

    class Connection:
        closed = False

        def fileno(self):
            return 123

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(probe.os, "open", lambda *args: 9)
    monkeypatch.setattr(probe.os, "dup2", lambda source, target: redirects.append((source, target)))
    monkeypatch.setattr(probe.os, "close", lambda *args: None)
    monkeypatch.setattr(probe.os, "write", lambda descriptor, data: writes.append(bytes(data)) or len(data))
    monkeypatch.setattr(probe, "_request_json_http", lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("private synthetic response content")))
    probe._request_worker(connection, "/health", None, 1)
    assert redirects == [(9, 1), (9, 2)] and connection.closed is True
    assert b"".join(writes) == b'{"ok":false}'


@pytest.fixture
def fake_worker(monkeypatch):
    """Exercise actual nonblocking pipes with an owned, stoppable fake child."""
    original_context = probe.multiprocessing.get_context("spawn")
    state = {"packet": b'{"ok":true,"response":{"value":42}}', "trickle": False, "processes": []}

    class Process:
        def __init__(self, *, target, args, daemon):
            assert target is probe._request_worker and daemon is True
            self.args, self.stop = args, threading.Event()
            self.terminated = self.killed = self.closed = False
            state["processes"].append(self)

        def start(self):
            descriptor = os.dup(self.args[0].fileno())

            def run():
                try:
                    if state["trickle"]:
                        for value in state["packet"]:
                            if self.stop.wait(.02):
                                break
                            os.write(descriptor, bytes([value]))
                    else:
                        os.write(descriptor, state["packet"])
                except OSError:
                    pass
                finally:
                    os.close(descriptor)

            self.thread = threading.Thread(target=run, daemon=True)
            self.thread.start()

        def is_alive(self):
            return self.thread.is_alive()

        def terminate(self):
            self.terminated = True
            self.stop.set()

        def kill(self):
            self.killed = True
            self.stop.set()

        def join(self, timeout):
            self.thread.join(timeout)

        def close(self):
            self.closed = True

    class Context:
        def Pipe(self, duplex):
            assert duplex is False
            return original_context.Pipe(duplex=False)

    Context.Process = Process

    def context(name):
        assert name == "spawn"
        return Context()

    monkeypatch.setattr(probe.multiprocessing, "get_context", context)
    return state


def test_watchdog_success_returns_same_dict_and_closes_owned_worker(fake_worker):
    assert probe.request_json("/health", {"synthetic": True}, timeout=1) == {"value": 42}
    process = fake_worker["processes"][0]
    assert process.closed and not process.is_alive()
    assert process.args[1:] == ("/health", {"synthetic": True}, 1)


def synthetic_spawn_worker(connection, path, body, timeout):
    """A real spawned process verifies IPC wiring without any network/media."""
    try:
        assert path == "/health" and body == {"synthetic": True}
        os.write(connection.fileno(), b'{"ok":true,"response":{"synthetic_spawn":true}}')
    finally:
        connection.close()


def test_real_spawned_worker_pipe_roundtrip_without_network_or_media(monkeypatch):
    monkeypatch.setattr(probe, "_request_worker", synthetic_spawn_worker)
    assert probe.request_json("/health", {"synthetic": True}, timeout=5) == {"synthetic_spawn": True}


def test_watchdog_error_is_sanitized_and_worker_closed(fake_worker):
    fake_worker["packet"] = b'{"ok":false,"private":"synthetic secret"}'
    with pytest.raises(RuntimeError, match="failed or returned") as error:
        probe.request_json("/health", timeout=1)
    assert "synthetic secret" not in str(error.value)
    assert fake_worker["processes"][0].closed


def test_trickling_partial_worker_message_cannot_extend_wall_clock(fake_worker):
    fake_worker["trickle"] = True
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="wall-clock"):
        probe.request_json("/health", timeout=.08)
    elapsed = time.monotonic() - started
    process = fake_worker["processes"][0]
    assert elapsed < .7 and process.terminated and process.closed and not process.is_alive()


def test_worker_pipe_bound_does_not_wait_for_oversized_output(fake_worker, monkeypatch):
    monkeypatch.setattr(probe, "MAX_WORKER_BYTES", 16)
    with pytest.raises(ValueError, match="output exceeds"):
        probe.request_json("/health", timeout=1)
    assert fake_worker["processes"][0].closed


def test_cleanup_escalates_to_kill_and_join_if_terminate_is_ignored():
    events = []

    class Process:
        alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append("terminate")

        def join(self, timeout):
            events.append(("join", timeout))

        def kill(self):
            events.append("kill")
            self.alive = False

        def close(self):
            events.append("close")

    probe._stop_request_worker(Process())
    assert events == ["terminate", ("join", .25), "kill", ("join", .25), "close"]


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), "1"])
def test_invalid_watchdog_timeout_never_starts_worker(timeout, monkeypatch):
    monkeypatch.setattr(probe.multiprocessing, "get_context", lambda *args: pytest.fail("Invalid timeout must not spawn"))
    with pytest.raises(ValueError, match="finite and positive"):
        probe.request_json("/health", timeout=timeout)


def test_default_is_inspection_and_does_not_start_model(monkeypatch, capsys):
    monkeypatch.setattr(probe, "run_probe", lambda *a: pytest.fail("Inspection must not run inference"))
    probe.main([])
    result = json.loads(capsys.readouterr().out)
    assert result["will_execute"] is False
    assert result["bank_videos_used"] == 0


def test_native_server_is_local_offline_and_preserves_temporal_pairs():
    model = {"runtime_files": {"tools/local-action-model/runtime/example/llama-server": "unused"}}
    command = probe.server_command(model, {"bin_dir": "/tmp/approved-video-decoder/bin"})
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8003"
    assert command[command.index("--video-fps") + 1] == "2"
    assert command[command.index("--video-timestamp-interval") + 1] == "0"
    assert "--offline" in command and "--log-disable" in command
    assert not any(value.startswith(("http://", "https://")) for value in command)


def test_owned_server_group_is_reaped_after_timeout(monkeypatch):
    signals = []
    monkeypatch.setattr(probe.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    class Process:
        pid = 12345
        calls = 0
        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("synthetic-test", timeout)
    process = Process()
    probe.stop_server(process)
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert process.calls == 2


def test_readiness_retries_sanitized_connection_failure_then_checks_capability(monkeypatch):
    events = []

    class Guard:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def bind(self, address):
            assert address == ("127.0.0.1", 8003)

    class Process:
        def poll(self):
            return None

    process = Process()
    health_attempts = 0

    def request(path, *, timeout):
        nonlocal health_attempts
        events.append(path)
        if path == "/health":
            health_attempts += 1
            assert 0 < timeout <= 2
            if health_attempts == 1:
                raise RuntimeError("sanitized transient transport failure")
            return {"status": "ok"}
        assert path == "/v1/models"
        return {"data": [{"id": "local-action-review", "architecture": {"input_modalities": ["video"]}}]}

    monkeypatch.setattr(probe.socket, "socket", Guard)
    monkeypatch.setattr(probe, "server_command", lambda *args: ["/tmp/synthetic-runtime/llama-server"])
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(probe, "stop_server", lambda owned: events.append("stopped") if owned is process else pytest.fail("Wrong owner"))
    monkeypatch.setattr(probe, "request_json", request)
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: events.append("retry"))
    with probe.native_server({"alias": "local-action-review"}, {}):
        events.append("ready")
    assert events == ["/health", "retry", "/health", "/v1/models", "ready", "stopped"]


def test_capability_failure_after_health_is_terminal_and_server_is_cleaned(monkeypatch):
    class Guard:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def bind(self, address):
            pass

    class Process:
        def poll(self):
            return None

    events = []
    monkeypatch.setattr(probe.socket, "socket", Guard)
    monkeypatch.setattr(probe, "server_command", lambda *args: ["/tmp/synthetic-runtime/llama-server"])
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(probe, "request_json", lambda path, **kwargs: {} if path == "/health" else {"data": []})
    monkeypatch.setattr(probe, "stop_server", lambda owned: events.append("stopped"))
    monkeypatch.setattr(probe.time, "sleep", lambda *args: pytest.fail("Capability failures must not retry readiness"))
    with pytest.raises(ValueError, match="video capability"):
        with probe.native_server({"alias": "local-action-review"}, {}):
            pytest.fail("Invalid capability cannot become ready")
    assert events == ["stopped"]


def gate_report():
    return {"passed": True, "operational_failure": None, "seed": 42,
            "bank_videos_used": 0, "holdout_videos_used": 0, "frames_per_case": 4,
            "video_fps": 2, "video_timestamp_interval_ms": 0, "source_sha256": "fixed-source",
            "cases": [{"expected": motion, "correct": True} for motion in probe.MOTIONS[:3]]}


def test_gate_requires_exact_completed_controls_and_source(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "protocol_provenance", lambda *a: {"source_sha256": "fixed-source"})
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate_report()))
    assert probe.verify_temporal_gate(path, {}, {})["passed"] is True


@pytest.mark.parametrize("change", [
    {"passed": False}, {"source_sha256": "changed"}, {"seed": 13},
    {"bank_videos_used": 1}, {"video_timestamp_interval_ms": 1000},
    {"cases": [{"expected": "left_to_right", "correct": True}]},
    {"cases": ["not-a-record"]},
])
def test_failed_stale_or_malformed_gate_blocks_bank_pilot(tmp_path, monkeypatch, change):
    monkeypatch.setattr(probe, "protocol_provenance", lambda *a: {"source_sha256": "fixed-source"})
    report = {**gate_report(), **change}
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="temporal gate"):
        probe.verify_temporal_gate(path, {}, {})
