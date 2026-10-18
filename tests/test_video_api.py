"""API contract tests use dummy bytes only; no training footage or cloud tools."""

import asyncio
import base64
from io import BytesIO
import json
import math
from pathlib import Path
import time

import pytest

from src.video_api import create_app
from PIL import Image


KEY = "synthetic-test-key"
_jpeg = BytesIO()
Image.new("RGB", (4, 4), color="white").save(_jpeg, format="JPEG")
JPEG = base64.b64encode(_jpeg.getvalue()).decode()


class Client:
    """Small ASGI driver avoids adding an HTTP-client dependency for tests."""

    def __init__(self, app):
        self.app = app

    async def __aenter__(self):
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self.lifespan.__aexit__(*args)

    async def request(self, method, path, body=b"", headers=None):
        headers = {name.lower(): value for name, value in (headers or {}).items()}
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
                 "query_string": b"", "root_path": "", "headers": [
                     (name.encode(), value.encode()) for name, value in headers.items()],
                 "client": ("127.0.0.1", 12345), "server": ("test", 80)}
        messages = []
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        await self.app(scope, receive, send)
        start = next(message for message in messages if message["type"] == "http.response.start")
        payload = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
        content_type = dict(start["headers"]).get(b"content-type", b"")
        return start["status"], json.loads(payload) if b"application/json" in content_type else payload.decode()

    async def upload(self, payload=b"dummy local video bytes", name="clip.mov", key=KEY):
        body = (b"--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\""
                + name.encode() + b"\"\r\nContent-Type: video/quicktime\r\n\r\n"
                + payload + b"\r\n--test-boundary--\r\n")
        return await self.request("POST", "/api/v1/analyze", body, {
            "X-API-Key": key, "Content-Type": "multipart/form-data; boundary=test-boundary",
            "Content-Length": str(len(body)),
        })

    async def status(self, identifier, key=KEY):
        return await self.request("GET", f"/api/v1/jobs/{identifier}", headers={"X-API-Key": key})

    async def done(self, identifier, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code, result = await self.status(identifier)
            assert code == 200
            if result["status"] != "processing":
                return result
            await asyncio.sleep(0.005)
        raise AssertionError("Synthetic analysis did not finish")


def run(coroutine):
    return asyncio.run(coroutine)


async def negative(path, model, deadline):
    assert path.suffix == ".mov" and path.exists()
    assert deadline > time.monotonic()
    return {"verdict": "no_violation", "violations": [], "secret_metadata": "must not escape"}


def positive_result(*, timestamp=1.2, image=JPEG):
    return {"verdict": "violation", "violations": [{
        "type": "Synthetic event", "timecode_sec": timestamp, "description": "Synthetic observation",
        "evidence_frames": [{"timestamp_sec": timestamp, "image_base64": image}],
    }]}


def test_auth_required_and_key_never_in_ui(tmp_path):
    async def scenario():
        app = create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=negative)
        async with Client(app) as client:
            assert (await client.upload(key=""))[0] == 401
            assert (await client.upload(key="wrong"))[0] == 401
            assert (await client.status("unknown", key=""))[0] == 401
            assert (await client.status("unknown"))[0] == 404
            code, html = await client.request("GET", "/")
            assert code == 200 and KEY not in html
            assert "localStorage" not in html and "innerHTML" not in html
            assert (await client.request("GET", "/docs"))[0] == 404
            assert (await client.request("GET", "/openapi.json"))[0] == 404
            assert not list(tmp_path.iterdir())
    run(scenario())


def test_unconfigured_key_rejects_work(tmp_path):
    async def scenario():
        async with Client(create_app(api_key="", jobs_dir=tmp_path, analysis_runner=negative)) as client:
            assert (await client.upload())[0] == 503
    run(scenario())


def test_negative_exact_schema_and_upload_cleanup(tmp_path):
    async def scenario():
        app = create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=negative)
        async with Client(app) as client:
            code, accepted = await client.upload(name="../../violation-47.mov")
            assert code == 202 and set(accepted) == {"job_id"}
            identifier = accepted["job_id"]
            result = await client.done(identifier)
            assert result == {"job_id": identifier, "status": "done", "verdict": "no_violation", "violations": []}
            assert not list(tmp_path.iterdir())
    run(scenario())


def test_positive_exact_schema_strips_internal_fields_and_preserves_text(tmp_path):
    async def positive(path, model, deadline):
        assert path.name != "positive.mov"
        return {"verdict": "violation", "violations": [{
            "type": "Synthetic event", "timecode_sec": 1.2, "description": "<script>literal text</script>",
            "internal_score": 0.9, "evidence_frames": [{"timestamp_sec": 1.2, "image_base64": JPEG, "path": "private"}],
        }], "model_path": "private"}

    async def scenario():
        async with Client(create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=positive)) as client:
            _, accepted = await client.upload(name="positive.mov")
            result = await client.done(accepted["job_id"])
            assert set(result) == {"job_id", "status", "verdict", "violations"}
            event = result["violations"][0]
            assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
            assert event["description"] == "<script>literal text</script>"
            assert set(event["evidence_frames"][0]) == {"timestamp_sec", "image_base64"}
            decoded = Image.open(BytesIO(base64.b64decode(event["evidence_frames"][0]["image_base64"])))
            decoded.verify()
            assert decoded.format == "JPEG"
    run(scenario())


@pytest.mark.parametrize("bad_result", [
    {"verdict": "no_violation", "violations": [{"type": "x"}]},
    {"verdict": "violation", "violations": []},
    {"verdict": "violation", "violations": [{"type": "x", "timecode_sec": -1, "description": "x", "evidence_frames": []}]},
    {"verdict": "uncertain", "violations": []},
])
def test_invalid_analysis_result_is_failed_never_negative(tmp_path, bad_result):
    async def invalid(*_):
        return bad_result

    async def scenario():
        async with Client(create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=invalid)) as client:
            _, accepted = await client.upload()
            result = await client.done(accepted["job_id"])
            assert result == {"job_id": accepted["job_id"], "status": "failed"}
    run(scenario())


@pytest.mark.parametrize("payload", [
    b"\xff\xd8not-a-jpeg\xff\xd9",
    _jpeg.getvalue()[:100] + b"\xff\xd9",
])
def test_jpeg_markers_without_decodable_image_fail_analysis(tmp_path, payload):
    async def invalid(*_):
        return positive_result(image=base64.b64encode(payload).decode())

    async def scenario():
        async with Client(create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=invalid)) as client:
            _, accepted = await client.upload()
            result = await client.done(accepted["job_id"])
            assert result == {"job_id": accepted["job_id"], "status": "failed"}
            assert not list(tmp_path.iterdir())
    run(scenario())


def test_jpeg_with_openable_headers_but_invalid_pixel_encoding_is_rejected(tmp_path):
    payload = bytearray(_jpeg.getvalue())
    frame_header = payload.index(b"\xff\xc0")
    # Refer to an absent quantization table: image headers still open, but the
    # pixels cannot decode. A JPEG marker/header-only check would accept this.
    payload[frame_header + 12] = 3
    with Image.open(BytesIO(payload)) as headers:
        assert headers.format == "JPEG"

    async def invalid(*_):
        return positive_result(image=base64.b64encode(payload).decode())

    async def scenario():
        async with Client(create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=invalid)) as client:
            _, accepted = await client.upload()
            assert (await client.done(accepted["job_id"]))["status"] == "failed"
    run(scenario())


def test_compressed_evidence_cannot_trigger_unbounded_pixel_allocation(tmp_path, monkeypatch):
    from PIL import JpegImagePlugin

    payload = bytearray(_jpeg.getvalue())
    frame_header = payload.index(b"\xff\xc0")
    payload[frame_header + 5:frame_header + 7] = (5000).to_bytes(2, "big")
    payload[frame_header + 7:frame_header + 9] = (5000).to_bytes(2, "big")

    def unexpected_decode(*_args, **_kwargs):
        pytest.fail("Oversized evidence must be rejected before allocating decoded pixels")

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "load", unexpected_decode)

    async def oversized(*_):
        return positive_result(image=base64.b64encode(payload).decode())

    async def scenario():
        async with Client(create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=oversized)) as client:
            _, accepted = await client.upload()
            assert (await client.done(accepted["job_id"]))["status"] == "failed"
    run(scenario())


@pytest.mark.parametrize("field", ["event", "evidence"])
@pytest.mark.parametrize("timestamp", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_nonfinite_or_negative_timestamps_fail_analysis(tmp_path, field, timestamp):
    async def invalid(*_):
        result = positive_result()
        event = result["violations"][0]
        if field == "event":
            event["timecode_sec"] = timestamp
        else:
            event["evidence_frames"][0]["timestamp_sec"] = timestamp
        return result

    async def scenario():
        async with Client(create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=invalid)) as client:
            _, accepted = await client.upload()
            assert (await client.done(accepted["job_id"]))["status"] == "failed"
    run(scenario())


@pytest.mark.parametrize("field", ["event", "evidence"])
def test_billion_second_timestamp_rejected_with_known_source_duration(tmp_path, field):
    async def invalid(*_):
        result = positive_result()
        result["_source_timing"] = {"duration_sec": 4.0, "tolerance_sec": 0.25}
        event = result["violations"][0]
        if field == "event":
            event["timecode_sec"] = 1_000_000_000.0
        else:
            event["evidence_frames"][0]["timestamp_sec"] = 1_000_000_000.0
        return result

    async def scenario():
        async with Client(create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=invalid)) as client:
            _, accepted = await client.upload()
            assert (await client.done(accepted["job_id"]))["status"] == "failed"
    run(scenario())


@pytest.mark.parametrize("duration,timestamp,tolerance", [(4.0, 4.2, 0.25), (3600.0, 3599.0, 0.0)])
def test_private_source_duration_allows_rounding_and_long_videos(tmp_path, duration, timestamp, tolerance):
    async def positive(*_):
        result = positive_result(timestamp=timestamp)
        result["_source_timing"] = {"duration_sec": duration, "tolerance_sec": tolerance}
        return result

    async def scenario():
        async with Client(create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=positive)) as client:
            _, accepted = await client.upload()
            result = await client.done(accepted["job_id"])
            assert set(result) == {"job_id", "status", "verdict", "violations"}
            assert result["status"] == "done"
            assert result["violations"][0]["timecode_sec"] == timestamp
    run(scenario())


def test_result_validation_cannot_publish_done_after_deadline(tmp_path, monkeypatch):
    from src import video_api

    original = video_api.Result.model_validate
    validation_calls = []

    def delayed_validation(*args, **kwargs):
        validation_calls.append(True)
        result = original(*args, **kwargs)
        time.sleep(0.08)
        return result

    monkeypatch.setattr(video_api.Result, "model_validate", delayed_validation)

    async def scenario():
        app = create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=negative, deadline_seconds=0.05)
        async with Client(app) as client:
            _, accepted = await client.upload()
            # Waiting for the worker matters: an expired processing view alone
            # would conceal an incorrectly published late result.
            await asyncio.wait_for(app.state.jobs.queue.join(), timeout=2)
            assert validation_calls == [True]
            assert (await client.status(accepted["job_id"]))[1] == {
                "job_id": accepted["job_id"], "status": "failed",
            }
            assert not list(tmp_path.iterdir())
    run(scenario())


@pytest.mark.parametrize("fps,count", [(29.97, 300), (float("nan"), 300), (30, 0), (0, 300)])
def test_source_timing_probe_uses_metadata_without_decoding_frames(monkeypatch, fps, count):
    import cv2
    from src import video_api

    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, key):
            return {cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_COUNT: count}[key]

        def read(self):
            pytest.fail("The API timing probe must not duplicate video frame decoding")

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    context = video_api._source_timing_context("synthetic.mov", time.monotonic() + 5)
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        assert context is None  # Missing metadata must not create a guessed cap.
    else:
        assert context["duration_sec"] == pytest.approx(count / fps)
        assert context["tolerance_sec"] == 0.25
    assert capture.released


def test_decode_failure_never_returns_negative_or_exception_details(tmp_path):
    async def failure(*_):
        raise RuntimeError("PRIVATE VIDEO DESCRIPTION OR FILE PATH")

    async def scenario():
        async with Client(create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=failure)) as client:
            _, accepted = await client.upload()
            assert await client.done(accepted["job_id"]) == {"job_id": accepted["job_id"], "status": "failed"}
            assert not list(tmp_path.iterdir())
    run(scenario())


def test_file_and_body_limits_are_enforced(tmp_path):
    async def scenario():
        app = create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=negative, max_upload_bytes=32)
        async with Client(app) as client:
            assert (await client.upload(name="clip.mp4"))[0] == 415
            assert (await client.upload(payload=b""))[0] == 422
            assert (await client.upload(payload=b"x" * 33))[0] == 413
            assert (await client.upload(payload=b"x" * (129 * 1024)))[0] == 413
            assert not list(tmp_path.iterdir())
    run(scenario())


def test_streaming_upload_without_content_length_is_bounded(tmp_path):
    async def scenario():
        app = create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=negative, max_upload_bytes=32)
        async with Client(app) as client:
            code, response = await client.request("POST", "/api/v1/analyze", b"x" * (129 * 1024), {
                "X-API-Key": KEY, "Content-Type": "multipart/form-data; boundary=test-boundary",
            })
            assert code == 413
            assert response == {"detail": "Upload is too large"}
            assert not list(tmp_path.iterdir())
    run(scenario())


def test_queue_bounded_polling_responsive_and_queue_time_counts(tmp_path):
    calls = []

    async def slow(path, model, deadline):
        calls.append(path.name)
        await asyncio.sleep(1)
        return {"verdict": "no_violation", "violations": []}

    async def scenario():
        app = create_app(api_key=KEY, jobs_dir=tmp_path, analysis_runner=slow, queue_limit=1, deadline_seconds=0.08)
        async with Client(app) as client:
            _, first = await client.upload()
            _, second = await client.upload()
            assert (await client.upload())[0] == 503
            started = time.monotonic()
            assert (await client.status(first["job_id"]))[1]["status"] == "processing"
            assert time.monotonic() - started < 0.05
            assert (await client.done(first["job_id"]))["status"] == "failed"
            assert (await client.done(second["job_id"]))["status"] == "failed"
            await app.state.jobs.queue.join()
            assert len(calls) <= 2
            assert time.monotonic() - started < 0.25  # Queue wait consumed the same overall budget.
            assert not list(tmp_path.iterdir())
    run(scenario())


def test_job_storage_cannot_be_raw():
    from src.video_api import ROOT

    with pytest.raises(ValueError, match="outside data/raw"):
        create_app(api_key=KEY, jobs_dir=ROOT / "data/raw/forbidden")


def make_synthetic_mov_and_model(directory, verdict):
    """Only generated pixels and a fitted dummy estimator enter integration tests."""
    import hashlib

    import cv2
    import joblib
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from src import video_analysis as analysis

    video = directory / "synthetic.mov"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    if not writer.isOpened():
        pytest.skip("Local OpenCV cannot encode the synthetic MOV fixture")
    try:
        for index in range(40):
            frame = np.full((64, 96, 3), 20, dtype=np.uint8)
            cv2.rectangle(frame, ((index * 2) % 72, 18), ((index * 2) % 72 + 16, 38), (30, 180, 90), -1)
            writer.write(frame)
    finally:
        writer.release()
    model = directory / "synthetic.joblib"
    clip = DummyClassifier(strategy="constant", constant=verdict).fit(
        np.zeros((2, len(analysis.CLIP_FEATURE_NAMES))), [0, 1])
    window = DummyClassifier(strategy="constant", constant=1).fit(
        np.zeros((2, len(analysis.WINDOW_FEATURE_NAMES))), [0, 1])
    joblib.dump({
        "schema_version": analysis.SCHEMA_VERSION, "clip_model": clip, "window_model": window,
        "threshold": 0.5, "feature_config": dict(analysis.FEATURE_CONFIG),
        "feature_names": {"clip": analysis.CLIP_FEATURE_NAMES, "window": analysis.WINDOW_FEATURE_NAMES},
        "source_sha256": hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest(),
    }, model)
    return video, model


@pytest.fixture
def clean_spawn_main(monkeypatch):
    """Streamlit AppTest can leave its script installed as ``__main__``.

    Restore a guarded test-runner context while multiprocessing prepares its
    child. This only isolates tests; the actual uvicorn CLI already has a safe
    main module, as verified by the real-server smoke test.
    """
    import importlib.util
    import sys

    specification = importlib.util.find_spec("pytest.__main__")
    monkeypatch.setattr(sys.modules["__main__"], "__file__", specification.origin, raising=False)
    monkeypatch.setattr(sys.modules["__main__"], "__spec__", specification, raising=False)


@pytest.mark.parametrize("verdict", [0, 1])
def test_actual_spawned_analysis_on_generated_video(tmp_path, monkeypatch, clean_spawn_main, verdict):
    video, model = make_synthetic_mov_and_model(tmp_path, verdict)
    monkeypatch.setenv("DATATHON_VIDEO_MODEL", str(model))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

    async def scenario():
        app = create_app(api_key=KEY, jobs_dir=tmp_path / "jobs")
        async with Client(app) as client:
            code, accepted = await client.upload(payload=video.read_bytes())
            assert code == 202
            result = await client.done(accepted["job_id"], timeout=15)
            assert set(result) == {"job_id", "status", "verdict", "violations"}
            assert result["status"] == "done"
            assert result["verdict"] == ("violation" if verdict else "no_violation")
            assert bool(result["violations"]) == bool(verdict)
            if verdict:
                event = result["violations"][0]
                assert 0 <= event["timecode_sec"] < 4
                frame = event["evidence_frames"][0]
                evidence = Image.open(BytesIO(base64.b64decode(frame["image_base64"])))
                evidence.verify()
                assert evidence.format == "JPEG" and evidence.size == (96, 64)
            await app.state.jobs.queue.join()
            assert not list((tmp_path / "jobs").iterdir())
    run(scenario())


def test_actual_malformed_mov_is_failed_not_negative(tmp_path, monkeypatch, clean_spawn_main):
    _, model = make_synthetic_mov_and_model(tmp_path, verdict=0)
    monkeypatch.setenv("DATATHON_VIDEO_MODEL", str(model))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

    async def scenario():
        app = create_app(api_key=KEY, jobs_dir=tmp_path / "jobs")
        async with Client(app) as client:
            _, accepted = await client.upload(payload=b"synthetic invalid movie bytes")
            result = await client.done(accepted["job_id"], timeout=15)
            assert result == {"job_id": accepted["job_id"], "status": "failed"}
            await app.state.jobs.queue.join()
            assert not list((tmp_path / "jobs").iterdir())
    run(scenario())


def _sleeping_analysis_child(path, model, deadline, result_path):
    """Synthetic unresponsive native work for the hard-kill deadline test."""
    time.sleep(30)


def test_unresponsive_child_is_killed_and_polling_remains_responsive(tmp_path, monkeypatch, clean_spawn_main):
    import multiprocessing
    from src import video_api

    monkeypatch.setattr(video_api, "_child_analysis", _sleeping_analysis_child)
    children_before = {child.pid for child in multiprocessing.active_children()}

    async def scenario():
        app = create_app(api_key=KEY, jobs_dir=tmp_path, deadline_seconds=0.2)
        async with Client(app) as client:
            started = time.monotonic()
            _, accepted = await client.upload()
            await asyncio.sleep(0.05)
            poll_started = time.monotonic()
            assert (await client.status(accepted["job_id"]))[0] == 200
            assert time.monotonic() - poll_started < 0.1
            assert (await client.done(accepted["job_id"]))["status"] == "failed"
            await asyncio.wait_for(app.state.jobs.queue.join(), timeout=2)
            assert time.monotonic() - started < 2
            assert not list(tmp_path.iterdir())
    run(scenario())
    assert {child.pid for child in multiprocessing.active_children()} <= children_before
