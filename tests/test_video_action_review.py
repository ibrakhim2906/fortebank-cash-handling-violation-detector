"""Operator review checks use synthetic bytes only, never supplied footage."""
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from src import video_actions
from src.video_action_review import MAX_REVIEW_BYTES, _source_path, create_app


KEY = "synthetic-review-key-not-for-real-use"


class Client:
    """ASGI request driver avoids installing a new HTTP client for tests."""

    def __init__(self, app):
        self.app = app

    async def request(self, method, path, body=b"", key=KEY, headers=None):
        pairs = {"x-api-key": key, **(headers or {})}
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": method, "scheme": "http", "path": path,
                 "raw_path": path.encode(), "query_string": b"", "root_path": "",
                 "headers": [(name.lower().encode(), value.encode()) for name, value in pairs.items()],
                 "client": ("127.0.0.1", 12345), "server": ("test", 80)}
        messages, sent = [], False
        finished = asyncio.Event()

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await finished.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finished.set()

        await self.app(scope, receive, send)
        start = next(message for message in messages if message["type"] == "http.response.start")
        raw = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
        response_headers = dict(start["headers"])
        content_type = response_headers.get(b"content-type", b"")
        payload = json.loads(raw) if b"application/json" in content_type else raw
        return start["status"], payload, response_headers


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    media_root = tmp_path / "extracted"
    media_root.mkdir()
    path = media_root / "synthetic.mov"
    path.write_bytes(b"SYNTHETIC-VIDEO-BYTES-ONLY")
    task = {"task_id": "a" * 64, "group_id": "b" * 64, "path": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "duration_sec": 10.0, "official_target": 1, "anchor_sec": 4.0,
            "interval": {"start_sec": 1.0, "end_sec": 8.0}, "ambiguous_anchor": False}
    queue = {"tasks": [task], "protocol_fingerprint": "c" * 64}
    monkeypatch.setattr(video_actions, "load_queue", lambda *args, **kwargs: queue)
    out_dir = tmp_path / "reviews"
    out_dir.mkdir()
    app = create_app(api_key=KEY, out_dir=out_dir, protocol_path=tmp_path / "unused.json", media_root=media_root)
    return Client(app), task, media_root, out_dir


def human_review(task):
    return {"task_id": task["task_id"], "reviewer_id": "TEST-REVIEWER", "status": "verified",
            "interval": {"start_sec": 1.0, "end_sec": 8.0}, "coverage": "interval",
            "visibility": "clear", "actions": [{"action_id": "object_relocation", "presence": "present"}],
            "provenance": {"kind": "human_review"}}


def test_html_has_no_key_no_external_assets_and_apis_require_auth(prepared):
    client, task, _, out_dir = prepared

    async def scenario():
        status, html, headers = await client.request("GET", "/", key="")
        assert status == 200 and KEY.encode() not in html
        assert b"localStorage" not in html and b"innerHTML" not in html
        assert b"https://" not in html and b"http://" not in html
        assert headers[b"cache-control"] == b"no-store"
        assert b"frame-ancestors 'none'" in headers[b"content-security-policy"]
        for method, path in [("GET", "/api/review/tasks"), ("GET", "/api/review/summary"),
                             ("GET", "/api/review/media/" + task["task_id"]),
                             ("POST", "/api/review/labels")]:
            assert (await client.request(method, path, key="wrong"))[0] == 401
        assert (await client.request("GET", "/docs"))[0] == 404
        assert not list(out_dir.iterdir())

    asyncio.run(scenario())


def test_unconfigured_key_cannot_access_queue(prepared, monkeypatch):
    monkeypatch.delenv("DATATHON_API_KEY", raising=False)
    client = Client(create_app(api_key=""))
    assert asyncio.run(client.request("GET", "/api/review/tasks"))[0] == 503


def test_task_metadata_hides_binary_target_and_paths(prepared):
    client, task, _, _ = prepared
    status, result, _ = asyncio.run(client.request("GET", "/api/review/tasks"))
    assert status == 200
    item = result["tasks"][0]
    assert item["task_id"] == task["task_id"]
    assert not {"official_target", "path", "source_sha256", "group_id"} & item.keys()
    assert item["review_status"] == "unknown" and item["proposal"] is None
    assert len(result["actions"]) == len(video_actions.ACTION_VOCABULARY)


def test_media_is_exact_verified_source_and_supports_range(prepared):
    client, task, _, _ = prepared

    async def scenario():
        status, payload, headers = await client.request("GET", "/api/review/media/" + task["task_id"])
        assert status == 200 and payload == b"SYNTHETIC-VIDEO-BYTES-ONLY"
        assert headers[b"content-type"] == b"video/quicktime"
        assert headers[b"accept-ranges"] == b"bytes"
        status, payload, _ = await client.request("GET", "/api/review/media/" + task["task_id"], headers={"Range": "bytes=0-3"})
        assert status == 206 and payload == b"SYNT"
        status, payload, _ = await client.request("GET", "/api/review/media/" + task["task_id"], headers={"Range": "bytes=-4"})
        assert status == 206 and payload == b"ONLY"
        for invalid in ["bytes=10000-", "bytes=9-2", "bytes=-0", "bytes=0-2,5-8", "bytes=" + "9" * 500 + "-"]:
            assert (await client.request("GET", "/api/review/media/" + task["task_id"], headers={"Range": invalid}))[0] == 416
        assert (await client.request("GET", "/api/review/media/" + "f" * 64))[0] == 404

    asyncio.run(scenario())


def test_media_rejects_changed_source_and_outside_path(prepared, tmp_path):
    client, task, media_root, _ = prepared
    Path(task["path"]).write_bytes(b"CHANGED-SYNTHETIC-BYTES")
    assert asyncio.run(client.request("GET", "/api/review/media/" + task["task_id"]))[0] == 409
    outside = tmp_path / "outside.mov"
    outside.write_bytes(b"SYNTHETIC")
    task.update(path=str(outside), source_sha256=hashlib.sha256(outside.read_bytes()).hexdigest())
    assert asyncio.run(client.request("GET", "/api/review/media/" + task["task_id"]))[0] == 409
    with pytest.raises(ValueError, match="outside"):
        _source_path(task, media_root)


def test_verified_review_appends_without_echoing_sensitive_observations(prepared):
    client, task, _, out_dir = prepared
    review = human_review(task)

    async def scenario():
        first = await client.request("POST", "/api/review/labels", json.dumps(review).encode())
        second = await client.request("POST", "/api/review/labels", json.dumps(review).encode())
        assert first[0] == second[0] == 201
        assert first[1] == {"saved": True, "task_id": task["task_id"], "revision": 1, "status": "verified"}
        assert second[1]["revision"] == 2
        assert "actions" not in first[1] and "note" not in first[1]
        status, result, _ = await client.request("GET", "/api/review/summary")
        assert status == 200 and result["verified_tasks"] == 1
        assert result["verified_action_observations"] == 1
        assert result["holdout_groups"] == 0
        assert (await client.request("GET", "/api/review/tasks"))[1]["tasks"][0]["review_status"] == "verified"

    asyncio.run(scenario())
    assert len((out_dir / "reviews.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("change", [
    {"task_id": "f" * 64}, {"interval": {"start_sec": 0, "end_sec": 11}},
    {"reviewer_id": ""}, {"provenance": {"kind": "local_model_proposal"}},
    {"actions": [{"action_id": "object_relocation", "presence": "unknown"}]},
    {"actions": [{"action_id": "made_up_action", "presence": "present"}]},
])
def test_invalid_reviews_never_append(prepared, change):
    client, task, _, out_dir = prepared
    payload = {**human_review(task), **change}
    status, result, _ = asyncio.run(client.request("POST", "/api/review/labels", json.dumps(payload).encode()))
    assert status == 422 and "Invalid review" in result["detail"]
    assert not (out_dir / "reviews.jsonl").exists()


def test_review_body_is_bounded(prepared):
    client, _, _, out_dir = prepared
    assert asyncio.run(client.request("POST", "/api/review/labels", b" " * (MAX_REVIEW_BYTES + 1)))[0] == 413
    assert not (out_dir / "reviews.jsonl").exists()


def test_proposals_do_not_override_human_review_status(prepared):
    client, task, _, out_dir = prepared

    async def scenario():
        assert (await client.request("POST", "/api/review/labels", json.dumps(human_review(task)).encode()))[0] == 201
        proposal = {**human_review(task), "reviewer_id": None, "status": "proposed",
                    "provenance": {"kind": "local_model_proposal", "model_id": "synthetic-model",
                                   "weights_sha256": "d" * 64, "recipe_sha256": "e" * 64}}
        assert (await client.request("POST", "/api/review/labels", json.dumps(proposal).encode()))[0] == 201
        status, result, _ = await client.request("GET", "/api/review/tasks")
        assert status == 200
        assert result["tasks"][0]["review_status"] == "verified"
        assert result["tasks"][0]["proposal"]["model_id"] == "synthetic-model"
        assert result["tasks"][0]["proposal"]["actions"] == proposal["actions"]

    asyncio.run(scenario())
