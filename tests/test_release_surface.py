"""Actual HTTP probes: failed guards must never earn a readiness claim."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

import pytest

from scripts.audit_release_surface import audit, main, validate_origin


@contextmanager
def endpoint(*, health=None, reject_code=401, redirect=False):
    calls = []
    secret = "private-test-credential"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            assert self.headers.get("User-Agent") == "datathon-release-audit/1.0"
            calls.append((self.command, self.path))
            if self.path == "/health":
                if redirect is True:
                    self.send_response(302)
                    self.send_header("Location", "/redirect-target")
                    self.end_headers()
                    return
                self.reply(200, json.dumps(health if health is not None else {"ok": True, "queue": 0}))
            elif self.path == "/":
                self.reply(200, "<!doctype html><html></html>", "text/html")
            else:
                if redirect == "auth" and self.headers.get("X-API-Key") == secret:
                    self.send_response(302)
                    self.send_header("Location", "/redirect-target")
                    self.end_headers()
                    return
                code = 404 if self.headers.get("X-API-Key") == secret else reject_code
                self.reply(code, json.dumps({"detail": "unknown or unauthorized"}))

        def do_POST(self):
            assert self.headers.get("User-Agent") == "datathon-release-audit/1.0"
            calls.append((self.command, self.path))
            assert self.rfile.read(int(self.headers.get("Content-Length", 0))) == b""
            self.reply(reject_code, '{}')

        def reply(self, code, body, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(body.encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls, secret
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_live_http_auth_checks_and_no_media_or_secret_in_report():
    with endpoint() as (url, calls, secret):
        result = audit(url, key=secret)
    assert result["surface_checks_passed"]
    assert all(c["status"] == "pass" for c in result["checks"])
    assert result["release_qualified"] is False
    assert secret not in json.dumps(result)
    assert len(calls) == 7
    assert [path for method, path in calls if method == "POST"] == ["/api/v1/analyze"] * 2
    assert set(path for method, path in calls if method == "GET") == {
        "/", "/health", "/api/v1/jobs/release-audit-unknown"}


def test_missing_private_key_stays_explicitly_unverified():
    with endpoint() as (url, _, _):
        result = audit(url)
    assert result["surface_checks_passed"]
    assert result["checks"][-1]["status"] == "not_checked"
    assert result["release_qualified"] is False


@pytest.mark.parametrize("health", [{"ok": False, "queue": 0}, {"ok": True, "queue": True}, {"ok": True}, []])
def test_http_200_alone_does_not_prove_health(health):
    with endpoint(health=health) as (url, _, _):
        assert not audit(url)["surface_checks_passed"]


def test_auth_bypass_is_a_failure():
    with endpoint(reject_code=200) as (url, _, _):
        result = audit(url)
    assert not result["surface_checks_passed"]
    assert sum(c["status"] == "fail" for c in result["checks"]) == 4


def test_redirect_is_not_followed():
    with endpoint(redirect=True) as (url, calls, _):
        result = audit(url)
    assert not result["surface_checks_passed"]
    assert all(path != "/redirect-target" for _, path in calls)


def test_authenticated_redirect_does_not_forward_private_key():
    with endpoint(redirect="auth") as (url, calls, secret):
        result = audit(url, key=secret)
    assert not result["surface_checks_passed"]
    assert result["checks"][-1]["http_status"] == 302
    assert all(path != "/redirect-target" for _, path in calls)
    assert secret not in json.dumps(result)


@pytest.mark.parametrize("value", ["http://example.com", "https://user:secret@example.com", "https://example.com/path", "https://example.com?token=secret", "https://example.com/#fragment"])
def test_rejects_unsafe_or_ambiguous_origin(value):
    with pytest.raises(ValueError):
        validate_origin(value)


def test_cli_preserves_evidence_and_does_not_print_key(tmp_path, capsys):
    dest = tmp_path / "audit.json"
    key_file = tmp_path / "key"
    with endpoint() as (url, _, secret):
        key_file.write_text(secret)
        assert main(["--api", url, "--key-file", str(key_file), "--json", str(dest)]) == 0
        original = dest.read_bytes()
        with pytest.raises(SystemExit):
            main(["--api", url, "--json", str(dest)])
    assert dest.read_bytes() == original
    output = capsys.readouterr()
    assert secret not in output.out + output.err + dest.read_text()


def test_cli_rejects_raw_data_destination_before_network(tmp_path, monkeypatch):
    from scripts import audit_release_surface as module
    monkeypatch.setattr(module, "ROOT", tmp_path)
    destination = tmp_path / "data/raw/audit.json"
    with pytest.raises(SystemExit):
        main(["--api", "https://example.com", "--json", str(destination)])
    assert not destination.exists()
