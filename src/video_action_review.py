"""Loopback-only operator review: footage and annotations stay on this computer.

Run ``uv run python -m src.video_action_review`` after preparing the development
queue. This is a separate local labeling tool, not the jury analysis endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from src import config


ROOT = Path(__file__).resolve().parents[1]
MAX_REVIEW_BYTES = 32 * 1024
MAX_MEDIA_BYTES = 64 * 1024 * 1024
DEFAULT_MEDIA_ROOT = config.DATA_PROC / "fortebank-20261017/open/videos"
DEFAULT_KEY_FILE = config.DATA_PROC / "video-service/api-key.txt"
DEFAULT_PRETRAINED_DIR = config.DATA_PROC / "pretrained-action-probe-v1"
PRETRAINED_CLASSES = {"counting money": 112, "looking at phone": 265,
                      "talking on cell phone": 516, "texting": 525, "writing": 595}


def _pretrained_tasks(directory: Path, protocol_path: Path) -> list[dict]:
    """Validate private predictions against exactly eight frozen fold-zero train members."""
    from src.video_eval import load_protocol

    protocol = load_protocol(protocol_path)
    manifest_path, records_path, summary_path = (directory / name for name in ("manifest.json", "records.json", "summary.json"))
    for path in (manifest_path, records_path, summary_path):
        if path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Invalid pretrained probe artifact")
    manifest_bytes, records_bytes = manifest_path.read_bytes(), records_path.read_bytes()
    manifest, result, summary = json.loads(manifest_bytes), json.loads(records_bytes), json.loads(summary_path.read_bytes())
    if (not isinstance(manifest, dict) or not isinstance(result, dict)
            or manifest.get("schema_version") != 1 or result.get("schema_version") != 1
            or manifest.get("protocol_fingerprint") != protocol["fingerprint"]
            or result.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
            or not isinstance(summary, dict) or summary.get("schema_version") != 1
            or summary.get("status") != "completed" or summary.get("repeatability_passed") is not True
            or summary.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
            or summary.get("records_sha256") != hashlib.sha256(records_bytes).hexdigest()):
        raise ValueError("Pretrained probe provenance differs")
    selected = sorted(protocol["folds"][0]["train_group_ids"], key=lambda identifier:
                      hashlib.sha256(f"{config.SEED}:{identifier}".encode()).hexdigest())[:8]
    excluded = set(protocol["holdout_group_ids"]) | set(protocol["folds"][0]["valid_group_ids"])
    if (len(selected) != 8 or set(selected) & excluded
            or not set(selected) <= set(protocol["development_group_ids"])):
        raise ValueError("Pretrained probe requires development training members only")
    selections, records = manifest["selections"], result["records"]
    if ([item["group_id"] for item in selections] != selected
            or [item["group_id"] for item in records] != selected):
        raise ValueError("Pretrained probe selection differs from frozen train-only recipe")
    groups, tasks = {group["group_id"]: group for group in protocol["groups"]}, []
    for selection, record in zip(selections, records, strict=True):
        group = groups[selection["group_id"]]
        expected = {"group_id": group["group_id"], "path": group["path"],
                    "source_sha256": group["sha256"], "duration_sec": group["duration_sec"]}
        if any(selection.get(key) != value or record.get(key) != value for key, value in expected.items()):
            raise ValueError("Pretrained probe source differs from protocol")
        decoded_duration = record["decoded_duration_sec"]
        if (isinstance(decoded_duration, bool) or not isinstance(decoded_duration, (int, float))
                or not math.isfinite(decoded_duration) or decoded_duration <= 0
                or abs(decoded_duration - group["duration_sec"]) > 0.25):
            raise ValueError("Pretrained probe decoded duration differs from protocol")
        windows = record["windows"]
        if not isinstance(windows, list) or not 1 <= len(windows) <= 256:
            raise ValueError("Pretrained probe requires bounded window observations")
        public_windows = []
        previous_start = -1.0
        for window in windows:
            start, end, timestamp = (window[key] for key in ("start_sec", "end_sec", "timestamp_sec"))
            if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in (start, end, timestamp))
                    or not 0 <= start < end <= decoded_duration
                    or not start <= timestamp <= end or start < previous_start):
                raise ValueError("Pretrained probe has an invalid window")
            previous_start = start
            predictions = window["predictions"]
            if (not isinstance(predictions, list) or len(predictions) != len(PRETRAINED_CLASSES)
                    or {item["label"] for item in predictions} != set(PRETRAINED_CLASSES)):
                raise ValueError("Pretrained probe classes differ")
            for prediction in predictions:
                score, rank = prediction["score"], prediction["rank_1based"]
                if (prediction["class_index"] != PRETRAINED_CLASSES[prediction["label"]]
                        or isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(score) or not 0 <= score <= 1
                        or isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 600):
                    raise ValueError("Pretrained probe has an invalid class score")
            if sum(item["score"] for item in predictions) > 1.000001:
                raise ValueError("Pretrained probe class scores are not softmax probabilities")
            public_windows.append({"start_sec": start, "end_sec": end, "timestamp_sec": timestamp,
                                   "predictions": [{key: item[key] for key in ("label", "score", "rank_1based")}
                                                   for item in predictions]})
        tasks.append({**expected, "duration_sec": decoded_duration,
                      "task_id": hashlib.sha256(group["group_id"].encode()).hexdigest(),
                      "anchor_sec": None, "interval": {"start_sec": 0.0, "end_sec": decoded_duration},
                      "ambiguous_anchor": False, "review_status": "unverified", "proposal": None,
                      "windows": public_windows})
    return tasks


def _media_response(path: Path, range_header: str | None):
    """Stream bounded local reads without a thread-pool/socket wakeup dependency."""
    size = path.stat().st_size
    start, end, partial = 0, size - 1, False
    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header) if len(range_header) <= 128 else None
        if not match or size == 0:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        first, last = match.groups()
        if not first and not last:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        else:
            start = max(0, size - int(last))
        if start > end or start >= size:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        partial = True

    async def chunks():
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {"Content-Length": str(max(0, end - start + 1)), "Accept-Ranges": "bytes",
               "Content-Disposition": "inline; filename=review.mov"}
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(chunks(), status_code=206 if partial else 200,
                             media_type="video/quicktime", headers=headers)


def _source_path(task: dict, media_root: Path) -> Path:
    """Only serve the hashed queue member within the extracted development tree."""
    candidate = Path(task["path"])
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve(strict=True)
    allowed = media_root.resolve(strict=True)
    if (not candidate.is_relative_to(allowed) or not candidate.is_file()
            or candidate.suffix.lower() != ".mov"
            or candidate.is_relative_to(config.DATA_RAW.resolve())
            or candidate.stat().st_size > MAX_MEDIA_BYTES):
        raise ValueError("Media is outside the permitted development directory")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if not hmac.compare_digest(digest.hexdigest(), task["source_sha256"]):
        raise ValueError("Development source bytes no longer match the queue")
    return candidate


def create_app(*, api_key: str | None = None, out_dir: Path | None = None,
               protocol_path: Path | None = None,
               media_root: Path = DEFAULT_MEDIA_ROOT,
               pretrained_dir: Path = DEFAULT_PRETRAINED_DIR) -> FastAPI:
    # Import lazily so the HTML and an actionable unprepared-queue failure remain
    # available without opening or decoding any clip during application import.
    from src import video_actions as actions
    from scripts import prepare_transfer_review as transfer

    directory = Path(out_dir or actions.DEFAULT_ACTION_DIR)
    transfer_directory = directory / "transfer-pilot-v1"
    protocol = Path(protocol_path or actions.DEFAULT_PROTOCOL)
    configured_key = api_key if api_key is not None else os.getenv("DATATHON_API_KEY", "")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def private_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "media-src blob: 'self'; img-src 'self' data:; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
        )
        return response

    def authenticate(request: Request):
        if not configured_key:
            raise HTTPException(503, "The local review API key is not configured")
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied.encode(), configured_key.encode()):
            raise HTTPException(401, "A valid local API key is required")

    def queue():
        try:
            return actions.load_queue(out_dir=directory, protocol_path=protocol)
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "Prepare or repair the development review queue first") from None

    def transfer_queue():
        try:
            return transfer.load_transfer_queue(transfer_directory, protocol)
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "Prepare the focused transfer review queue first") from None

    def pretrained_queue():
        try:
            return _pretrained_tasks(Path(pretrained_dir), protocol)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            raise HTTPException(409, "The completed train-only pretrained probe is not ready or failed verification") from None

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse((ROOT / "app/action_review.html").read_text(encoding="utf-8"))

    @app.get("/api/review/tasks")
    async def tasks(request: Request):
        authenticate(request)
        prepared = queue()
        try:
            records = actions.load_reviews(out_dir=directory, protocol_path=protocol)
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "The local review log requires inspection") from None
        latest_human, latest_proposal = {}, {}
        for record in records:
            destination = latest_human if record["provenance"]["kind"] == "human_review" else latest_proposal
            destination[record["task_id"]] = record
        public = []
        for task in prepared["tasks"]:
            item = {name: task[name] for name in (
                "task_id", "duration_sec", "anchor_sec", "interval", "ambiguous_anchor"
            )}
            previous = latest_human.get(task["task_id"])
            item["review_status"] = previous["status"] if previous else "unknown"
            proposal = latest_proposal.get(task["task_id"])
            item["proposal"] = ({
                "actions": proposal["actions"], "visibility": proposal["visibility"],
                "interval": proposal["interval"], "model_id": proposal["provenance"]["model_id"],
            } if proposal else None)
            # Official binary targets, paths and hashes are intentionally hidden
            # from this labeling form to reduce confirmation bias.
            public.append(item)
        return {"tasks": public, "actions": [
            {"action_id": identifier, "label": label}
            for identifier, label in actions.ACTION_VOCABULARY.items()
        ]}

    @app.get("/api/review/media/{task_id}")
    async def media(task_id: str, request: Request):
        authenticate(request)
        task = next((item for item in queue()["tasks"] if item["task_id"] == task_id), None)
        if task is None:
            raise HTTPException(404, "Development review task not found")
        try:
            path = _source_path(task, Path(media_root))
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "Development media verification failed") from None
        # Authenticate one blob fetch, then play the in-memory local object URL.
        return _media_response(path, request.headers.get("Range"))

    @app.post("/api/review/labels")
    async def save(request: Request):
        authenticate(request)
        queue()
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_REVIEW_BYTES:
                raise HTTPException(413, "Review is too large")
        try:
            supplied = json.loads(body)
            if not isinstance(supplied, dict):
                raise ValueError("Expected one review object")
            record = actions.append_review(supplied, out_dir=directory, protocol_path=protocol)
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            raise HTTPException(422, "Invalid review: check interval, actions, visibility and reviewer") from None
        except OSError:
            raise HTTPException(409, "The local review log could not be updated") from None
        # Do not echo observations, notes, paths or source metadata.
        return JSONResponse({"saved": True, "task_id": record["task_id"],
                             "revision": record["revision"], "status": record["status"]}, status_code=201)

    @app.get("/api/review/summary")
    async def summary(request: Request):
        authenticate(request)
        queue()
        try:
            return actions.aggregate_summary(out_dir=directory, protocol_path=protocol)
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "The local review log requires inspection") from None

    @app.get("/api/review/transfer/tasks")
    async def transfer_tasks(request: Request):
        authenticate(request)
        prepared = transfer_queue()
        try:
            latest = {record["task_id"]: record for record in
                      transfer.load_transfer_reviews(transfer_directory, protocol)}
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "The local transfer review log requires inspection") from None
        items = []
        for task in prepared["tasks"]:
            item = {name: task[name] for name in ("task_id", "duration_sec", "anchor_sec", "interval", "ambiguous_anchor", "pool")}
            previous = latest.get(task["task_id"])
            item.update(review_status=previous["status"] if previous else "unknown", proposal=None)
            # Prior human choices are local form state only; never expose model
            # proposals, official targets, source paths or hashes in this mode.
            item["human_review"] = ({key: previous[key] for key in (
                "choice", "interval", "coverage", "visibility", "evidence_sec", "revision")}
                if previous else None)
            items.append(item)
        return {"mode": "transfer", "tasks": items, "actions": [],
                "definition": prepared["definition"],
                "choices": [{"value": key, "label": label} for key, label in transfer.CHOICES.items()]}

    @app.get("/api/review/transfer/media/{task_id}")
    async def transfer_media(task_id: str, request: Request):
        authenticate(request)
        task = next((item for item in transfer_queue()["tasks"] if item["task_id"] == task_id), None)
        if task is None:
            raise HTTPException(404, "Development transfer task not found")
        try:
            path = _source_path(task, Path(media_root))
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "Development media verification failed") from None
        return _media_response(path, request.headers.get("Range"))

    @app.post("/api/review/transfer/labels")
    async def transfer_save(request: Request):
        authenticate(request)
        transfer_queue()
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_REVIEW_BYTES:
                raise HTTPException(413, "Review is too large")
        try:
            record = transfer.append_transfer_review(json.loads(body), transfer_directory, protocol)
        except (ValueError, KeyError, TypeError, UnicodeDecodeError) as error:
            # Validation messages contain only fixed operator guidance.
            raise HTTPException(422, "Invalid transfer review: " + str(error)) from None
        except OSError:
            raise HTTPException(409, "The local transfer review log could not be updated") from None
        return JSONResponse({"saved": True, "task_id": record["task_id"],
                             "revision": record["revision"], "status": record["status"]}, status_code=201)

    @app.get("/api/review/transfer/summary")
    async def transfer_progress(request: Request):
        authenticate(request)
        transfer_queue()
        try:
            return transfer.transfer_summary(transfer_directory, protocol)
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "The local transfer review log requires inspection") from None

    @app.get("/api/review/pretrained/tasks")
    async def pretrained_tasks(request: Request):
        authenticate(request)
        items = [{key: task[key] for key in ("task_id", "duration_sec", "anchor_sec", "interval",
                  "ambiguous_anchor", "review_status", "proposal", "windows")} for task in pretrained_queue()]
        return {"mode": "pretrained", "read_only": True, "tasks": items, "actions": []}

    @app.get("/api/review/pretrained/summary")
    async def pretrained_summary(request: Request):
        authenticate(request)
        items = pretrained_queue()
        return {"train_recordings": len(items), "windows": sum(len(task["windows"]) for task in items),
                "blind_recordings": 0, "holdout_recordings": 0, "verified_labels_created": 0,
                "read_only": True, "scores_are_uncalibrated": True}

    @app.get("/api/review/pretrained/media/{task_id}")
    async def pretrained_media(task_id: str, request: Request):
        authenticate(request)
        task = next((item for item in pretrained_queue() if item["task_id"] == task_id), None)
        if task is None:
            raise HTTPException(404, "Pretrained probe task not found")
        try:
            path = _source_path(task, Path(media_root))
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "Pretrained probe media verification failed") from None
        return _media_response(path, request.headers.get("Range"))

    return app


def main():
    from src.video_actions import DEFAULT_ACTION_DIR, DEFAULT_PROTOCOL
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ACTION_DIR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("Port must be between 1 and 65535")
    try:
        key = args.key_file.read_text(encoding="utf-8").strip()
    except OSError:
        raise SystemExit("Start the local video service first to create its private API key") from None
    if len(key) < 24:
        raise SystemExit("The configured local API key is too short")
    # There is deliberately no --host argument: this operator tool binds only
    # loopback. It must never become the publicly deployed jury application.
    print(f"Local action review: http://127.0.0.1:{args.port}/", flush=True)
    uvicorn.run(create_app(api_key=key, out_dir=args.out_dir, protocol_path=args.protocol),
                host="127.0.0.1", port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
