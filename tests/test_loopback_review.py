"""Fixed-destination relaying, using synthetic streams and no sockets/media."""
import io
import json
from types import SimpleNamespace

import pytest

from scripts import loopback_request as relay


def invoke(monkeypatch, service, method, path):
    metadata = {"method": method, "path": path, "content_length": 0,
                "headers": {"X-API-Key": "synthetic-key"}}
    source = io.BytesIO(json.dumps(metadata).encode() + b"\n")
    output = io.BytesIO()
    destinations = []

    class Connection:
        def __init__(self, host, port, timeout):
            destinations.append((host, port))

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return SimpleNamespace(status=200, read=lambda maximum: b"synthetic-local-response",
                                   getheaders=lambda: [("Content-Type", "application/json")])

        def close(self):
            pass

    monkeypatch.setattr(relay.sys, "stdin", SimpleNamespace(buffer=source))
    monkeypatch.setattr(relay.sys, "stdout", SimpleNamespace(buffer=output))
    monkeypatch.setattr(relay.http.client, "HTTPConnection", Connection)
    relay.main(["--service", service])
    header, body = output.getvalue().split(b"\n", 1)
    return json.loads(header), body, destinations


@pytest.mark.parametrize("service,path,port", [
    ("video", "/", 8000),
    ("review", "/api/review/tasks", 8002),
    ("review", "/api/review/summary", 8002),
    ("review", "/api/review/media/" + "a" * 64, 8002),
    ("review", "/api/review/transfer/tasks", 8002),
    ("review", "/api/review/transfer/summary", 8002),
    ("review", "/api/review/transfer/media/" + "a" * 64, 8002),
    ("review", "/api/review/pretrained/tasks", 8002),
    ("review", "/api/review/pretrained/summary", 8002),
    ("review", "/api/review/pretrained/media/" + "a" * 64, 8002),
])
def test_only_fixed_service_destinations(monkeypatch, service, path, port):
    header, body, destinations = invoke(monkeypatch, service, "GET", path)
    assert header["status"] == 200
    assert body == b"synthetic-local-response"
    assert destinations == [("127.0.0.1", port)]


def test_transfer_label_posts_stay_on_local_review_service(monkeypatch):
    header, _, destinations = invoke(monkeypatch, "review", "POST", "/api/review/transfer/labels")
    assert header["status"] == 200
    assert destinations == [("127.0.0.1", 8002)]


def test_pretrained_predictions_cannot_be_saved_as_labels(monkeypatch):
    header, _, destinations = invoke(monkeypatch, "review", "POST", "/api/review/pretrained/labels")
    assert header["status"] == 502
    assert not destinations


@pytest.mark.parametrize("service,path", [
    ("video", "/api/review/tasks"),
    ("review", "/api/v1/analyze"),
    ("review", "http://example.com/private"),
    ("review", "/api/review/media/../../data/raw/file"),
    ("review", "/api/review/transfer/media/../../data/raw/file"),
    ("video", "/api/review/transfer/tasks"),
    ("review", "/api/review/pretrained/media/../../data/raw/file"),
    ("video", "/api/review/pretrained/tasks"),
    ("unknown", "/"),
])
def test_no_cross_service_arbitrary_path_or_external_target(monkeypatch, service, path):
    header, _, destinations = invoke(monkeypatch, service, "GET", path)
    assert header["status"] == 502
    assert not destinations
