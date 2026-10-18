"""Local, bounded video jobs and the exact jury API (one server worker only).

Run ``uv run uvicorn src.video_api:app --workers 1`` with DATATHON_API_KEY set.
The key is read from the environment and is never put into HTML or responses.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import hmac
from io import BytesIO
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Awaitable, Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, ValidationInfo, model_validator
from starlette.datastructures import UploadFile


ROOT = Path(__file__).resolve().parents[1]
MAX_RESULT_BYTES = 8 * 1024 * 1024
# Bound allocation during JPEG validation; current engines emit <= 960 px sides.
MAX_EVIDENCE_PIXELS = 4096 * 4096
AnalysisRunner = Callable[[Path, str | None, float], Awaitable[dict]]


class Evidence(BaseModel):
    timestamp_sec: float = Field(ge=0, allow_inf_nan=False)
    image_base64: str = Field(min_length=8, max_length=2 * 1024 * 1024)

    @model_validator(mode="after")
    def jpeg_payload(self):
        try:
            payload = base64.b64decode(self.image_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid evidence encoding") from exc
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise ValueError("Evidence must be a JPEG")
        try:
            with Image.open(BytesIO(payload)) as decoded:
                if decoded.format != "JPEG":
                    raise ValueError("Evidence must be a JPEG")
                if decoded.width * decoded.height > MAX_EVIDENCE_PIXELS:
                    raise ValueError("Evidence image exceeds decoded size limit")
                # Opening headers or checking markers does not decode JPEG pixels.
                decoded.load()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError("Evidence must be a decodable JPEG") from exc
        return self


class Violation(BaseModel):
    type: str = Field(min_length=1, max_length=500)
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str = Field(min_length=1, max_length=10000)
    evidence_frames: list[Evidence] = Field(min_length=1, max_length=3)


class Result(BaseModel):
    verdict: Literal["violation", "no_violation"]
    violations: list[Violation] = Field(max_length=64)

    @model_validator(mode="after")
    def consistent_verdict(self, info: ValidationInfo):
        if bool(self.violations) != (self.verdict == "violation"):
            raise ValueError("Verdict and events disagree")
        timing = info.context
        if timing is not None:
            if not isinstance(timing, dict):
                raise ValueError("Source duration context is invalid")
            duration, tolerance = timing.get("duration_sec"), timing.get("tolerance_sec", 0.0)
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in (duration, tolerance)):
                raise ValueError("Source duration context is invalid")
            if duration <= 0 or tolerance < 0:
                raise ValueError("Source duration context is invalid")
            for event in self.violations:
                times = [event.timecode_sec, *(frame.timestamp_sec for frame in event.evidence_frames)]
                if any(timestamp > duration + tolerance for timestamp in times):
                    raise ValueError("Event and evidence timestamps must be within source duration")
        return self


def _source_timing_context(path: str, deadline: float) -> dict | None:
    """Read matching decoder metadata only in the killable analysis child.

    Frame count / average FPS is an estimate, not a decoded presentation-time
    audit. Allow header rounding and two source frames; never invent a maximum
    video duration when trustworthy duration metadata is unavailable.
    """
    import cv2

    if time.monotonic() >= deadline:
        raise TimeoutError("Video processing deadline exceeded")
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("Source timing metadata is unavailable")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if time.monotonic() >= deadline:
        raise TimeoutError("Video processing deadline exceeded")
    if not math.isfinite(fps) or not math.isfinite(count) or fps <= 0 or count <= 0:
        return None
    duration = count / fps
    if not math.isfinite(duration) or duration <= 0:
        return None
    return {"duration_sec": duration, "tolerance_sec": max(0.25, 2.0 / fps),
            "source": "opencv_header_frame_count_over_fps"}


def _child_analysis(path: str, model: str | None, deadline: float, result_path: str):
    """Only this local process touches a video decoder/model; never cloud APIs."""
    temporary = Path(result_path + ".tmp")
    try:
        from src.video_inference import analyze_video

        result = analyze_video(Path(path), model_path=model, deadline=deadline)
        timing = _source_timing_context(path, deadline)
        validated = Result.model_validate(result, context=timing).model_dump()
        # Private transport metadata is removed by the worker's Result filter.
        if timing is not None:
            validated["_source_timing"] = timing
        payload = json.dumps(validated, allow_nan=False).encode("utf-8")
        if len(payload) > MAX_RESULT_BYTES:
            raise ValueError("Result exceeds bounded size")
        with temporary.open("xb") as target:
            target.write(payload)
        temporary.replace(result_path)
    except Exception:
        # No exception text: paths, model internals and observations stay private.
        try:
            temporary.write_bytes(b'{"failed":true}')
            temporary.replace(result_path)
        except OSError:
            pass


async def _run_analysis(path: Path, model: str | None, deadline: float) -> dict:
    """A hung native decoder is killable; async status polling stays responsive."""
    context = mp.get_context("spawn")
    result_path = path.with_suffix(".result.json")
    process = context.Process(
        target=_child_analysis, args=(str(path), model, deadline, str(result_path)), daemon=True
    )
    try:
        process.start()
        while time.monotonic() < deadline:
            if result_path.exists():
                if result_path.stat().st_size > MAX_RESULT_BYTES:
                    raise ValueError("Result exceeds bounded size")
                result = json.loads(result_path.read_bytes())
                if result.get("failed"):
                    raise RuntimeError("Local analysis failed")
                return result
            if not process.is_alive():
                raise RuntimeError("Local analysis exited without a result")
            await asyncio.sleep(0.04)
        raise TimeoutError("Video processing deadline exceeded")
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            # Poll without blocking the event loop or creating a thread pool.
            cleanup_deadline = time.monotonic() + 0.25
            while process.is_alive() and time.monotonic() < cleanup_deadline:
                await asyncio.sleep(0.01)
            if process.is_alive():
                process.kill()
                cleanup_deadline = time.monotonic() + 0.25
                while process.is_alive() and time.monotonic() < cleanup_deadline:
                    await asyncio.sleep(0.01)
            if not process.is_alive():
                process.join(timeout=0)
                process.close()
        result_path.unlink(missing_ok=True)
        Path(str(result_path) + ".tmp").unlink(missing_ok=True)


class Jobs:
    def __init__(self, directory: Path, runner: AnalysisRunner, model: str | None,
                 deadline_seconds: float, queue_limit: int, retained_jobs: int):
        self.directory = directory
        self.runner = runner
        self.model = model
        self.deadline_seconds = deadline_seconds
        self.queue_limit = queue_limit
        self.retained_jobs = retained_jobs
        self.records: dict[str, dict] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_limit + 1)
        self.worker: asyncio.Task | None = None

    def reserve(self, started: float) -> str:
        active = sum(record["status"] == "processing" for record in self.records.values())
        if active >= self.queue_limit + 1:
            raise HTTPException(503, "Analysis capacity is busy. Try again shortly.",
                                headers={"Retry-After": "5"})
        for identifier, record in list(self.records.items()):
            if record["status"] != "processing" and (
                len(self.records) >= self.retained_jobs or started - record["started"] > 1800
            ):
                del self.records[identifier]
        identifier = uuid4().hex
        self.records[identifier] = {
            "job_id": identifier, "status": "processing", "started": started,
            "deadline": started + self.deadline_seconds,
            "path": self.directory / f"{identifier}.mov",
        }
        return identifier

    def remove(self, identifier: str):
        record = self.records.pop(identifier, None)
        if record is not None:
            record["path"].unlink(missing_ok=True)

    async def work(self):
        while True:
            identifier = await self.queue.get()
            record = self.records[identifier]
            try:
                remaining = record["deadline"] - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Deadline expired in queue")
                result = await asyncio.wait_for(
                    self.runner(record["path"], self.model, record["deadline"]),
                    timeout=remaining,
                )
                if time.monotonic() >= record["deadline"]:
                    raise TimeoutError("Analysis finished after deadline")
                filtered = Result.model_validate(result, context=result.get("_source_timing")).model_dump()
                if len(json.dumps(filtered).encode("utf-8")) > MAX_RESULT_BYTES:
                    raise ValueError("Result too large")
                if time.monotonic() >= record["deadline"]:
                    raise TimeoutError("Result validation finished after deadline")
                record.update(status="done", result=filtered)
            except asyncio.CancelledError:
                record["status"] = "failed"
                raise
            except Exception:
                record["status"] = "failed"
            finally:
                record["path"].unlink(missing_ok=True)
                self.queue.task_done()

    def public(self, identifier: str) -> dict:
        record = self.records.get(identifier)
        if record is None:
            raise HTTPException(404, "Job not found")
        # Expired queued work must not appear to still be within its deadline.
        if record["status"] == "processing" and time.monotonic() >= record["deadline"]:
            return {"job_id": identifier, "status": "failed"}
        response = {"job_id": identifier, "status": record["status"]}
        if record["status"] == "done":
            response.update(record["result"])
        return response


class UploadBodyLimit:
    """Bound multipart bytes before form parsing spools them to disk."""

    def __init__(self, app, maximum: int):
        self.app = app
        self.maximum = maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        length = dict(scope.get("headers", [])).get(b"content-length")
        if length is not None:
            try:
                if int(length) > self.maximum:
                    response = JSONResponse({"detail": "Upload is too large"}, status_code=413)
                    return await response(scope, receive, send)
            except ValueError:
                pass
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.maximum:
                raise HTTPException(413, "Upload is too large")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(*, api_key: str | None = None, jobs_dir: Path | None = None,
               analysis_runner: AnalysisRunner | None = None,
               max_upload_bytes: int | None = None, deadline_seconds: float = 300,
               queue_limit: int = 2, retained_jobs: int = 32) -> FastAPI:
    """Factory supports isolated contract tests without loading real footage."""
    secret = os.environ.get("DATATHON_API_KEY", "") if api_key is None else api_key
    maximum = max_upload_bytes or int(os.environ.get("DATATHON_MAX_UPLOAD_BYTES", 64 * 1024 * 1024))
    directory = Path(jobs_dir or os.environ.get("DATATHON_VIDEO_JOBS", ROOT / "data/processed/video-jobs")).resolve()
    raw = (ROOT / "data/raw").resolve()
    if directory == raw or raw in directory.parents:
        raise ValueError("Job storage must be outside data/raw")
    if maximum <= 0 or not 0 < deadline_seconds <= 300 or queue_limit < 0 or retained_jobs < queue_limit + 1:
        raise ValueError("Invalid job limits")
    manager = Jobs(directory, analysis_runner or _run_analysis,
                   os.environ.get("DATATHON_VIDEO_MODEL"), deadline_seconds, queue_limit, retained_jobs)

    @asynccontextmanager
    async def lifespan(application):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        manager.worker = asyncio.create_task(manager.work())
        try:
            yield
        finally:
            manager.worker.cancel()
            try:
                await manager.worker
            except asyncio.CancelledError:
                pass
            for identifier in list(manager.records):
                manager.remove(identifier)

    application = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    application.state.jobs = manager
    application.add_middleware(UploadBodyLimit, maximum=maximum + 128 * 1024)

    def authorize(request: Request):
        if not secret:
            raise HTTPException(503, "Service API key has not been configured")
        presented = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(presented.encode("utf-8"), secret.encode("utf-8")):
            raise HTTPException(401, "A valid X-API-Key is required")

    @application.get("/", response_class=HTMLResponse)
    async def interface():
        return HTMLResponse((ROOT / "app/video_ui.html").read_text(encoding="utf-8"), headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'self'; form-action 'self'; frame-ancestors 'none'",
            "Referrer-Policy": "no-referrer",
        })

    @application.post("/api/v1/analyze", status_code=202)
    async def upload(request: Request):
        authorize(request)
        started = time.monotonic()
        identifier = manager.reserve(started)
        record = manager.records[identifier]
        try:
            async with asyncio.timeout(deadline_seconds):
                async with request.form(max_files=1, max_fields=1) as form:
                    file = form.get("file")
                    if not isinstance(file, UploadFile) or len(form.multi_items()) != 1 or set(form.keys()) != {"file"}:
                        raise HTTPException(422, "Provide one multipart video in the file field")
                    if not (file.filename or "").lower().endswith(".mov"):
                        raise HTTPException(415, "Only .mov video files are accepted")
                    length = 0
                    with record["path"].open("xb") as target:
                        while chunk := await file.read(256 * 1024):
                            length += len(chunk)
                            if length > maximum:
                                raise HTTPException(413, "Upload is too large")
                            target.write(chunk)
                    if length == 0:
                        raise HTTPException(422, "Video is empty")
            manager.queue.put_nowait(identifier)
            return {"job_id": identifier}
        except BaseException as exc:
            manager.remove(identifier)
            if isinstance(exc, TimeoutError):
                raise HTTPException(408, "Upload exceeded processing deadline") from exc
            raise

    @application.get("/api/v1/jobs/{job_id}")
    async def status(job_id: str, request: Request):
        authorize(request)
        return JSONResponse(manager.public(job_id), headers={"Cache-Control": "no-store"})

    return application


app = create_app()
