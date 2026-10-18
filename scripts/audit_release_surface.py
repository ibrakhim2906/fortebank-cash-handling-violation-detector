"""Probe public health/UI/auth only; never upload media or invoke recognition.

Example: uv run --offline python -m scripts.audit_release_surface \
  --api https://HOST --json data/processed/release-audit.json
Optional --key-file reads a private key for an unknown-job GET, never logs it.
Passing these checks does not qualify recognition, offline inference or release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ipaddress
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
WRONG_KEY = "release-audit-deliberately-invalid"
UNKNOWN_JOB = "/api/v1/jobs/release-audit-unknown"
MAX_BODY = 128 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_origin(value):
    parsed = urlsplit(value)
    if (not parsed.hostname or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in ("", "/")
            or any(c.isspace() or ord(c) < 32 for c in value)):
        raise ValueError("Provide an origin without credentials, path, query or fragment")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("HTTPS required except for a literal loopback host or localhost")
    parsed.port  # Reject invalid ports before issuing requests.
    return value.rstrip("/")


def request(origin, method, path, headers, timeout):
    """Bound responses; never persist response bodies or exception messages."""
    started = time.monotonic()
    req = Request(origin + path, method=method,
                  headers={"User-Agent": "datathon-release-audit/1.0", **headers},
                  data=b"" if method == "POST" else None)
    try:
        try:
            response = build_opener(NoRedirect()).open(req, timeout=timeout)
        except HTTPError as exc:
            response = exc
        with response:
            body = response.read(MAX_BODY + 1)
            observation = {
                "http_status": response.code,
                "content_type": response.headers.get_content_type(),
                "body_within_limit": len(body) <= MAX_BODY,
            }
    except (OSError, URLError, ValueError) as exc:
        body = b""
        observation = {"http_status": None, "error_code": type(exc).__name__}
    observation["elapsed_sec"] = round(time.monotonic() - started, 6)
    return observation, body


def audit(origin, *, key=None, timeout=15):
    origin = validate_origin(origin)
    if key is not None and (not key or key == WRONG_KEY or any(c.isspace() for c in key)):
        raise ValueError("Key must be a nonempty, single token distinct from the invalid probe")
    if not 0 < timeout <= 60:
        raise ValueError("Timeout must be greater than zero and at most 60 seconds")
    checks = []

    def check(name, method, path, headers, expected, predicate=None):
        observed, body = request(origin, method, path, headers, timeout)
        passed = observed["http_status"] == expected and observed.get("body_within_limit", False)
        if passed and predicate is not None:
            try:
                passed = bool(predicate(observed, body))
            except (ValueError, KeyError, TypeError, UnicodeError):
                passed = False
        checks.append(dict(name=name, method=method, path=path, expected_http_status=expected,
                           status="pass" if passed else "fail", **observed))

    def healthy(observed, body):
        value = json.loads(body)
        return (isinstance(value, dict) and observed["content_type"] == "application/json" and value.get("ok") is True
                and type(value.get("queue")) is int and value["queue"] >= 0)

    check("health", "GET", "/health", {}, 200, healthy)
    check("ui_document", "GET", "/", {}, 200,
          lambda obs, body: obs["content_type"] == "text/html" and b"<html" in body.lower())
    for label, headers in [("missing_key", {}), ("wrong_key", {"X-API-Key": WRONG_KEY})]:
        check("upload_" + label, "POST", "/api/v1/analyze", headers, 401)
        check("poll_" + label, "GET", UNKNOWN_JOB, headers, 401)
    if key is None:
        checks.append(dict(name="valid_key_unknown_job", status="not_checked",
                           reason="No private key supplied; valid-key acceptance is unverified"))
    else:
        check("valid_key_unknown_job", "GET", UNKNOWN_JOB, {"X-API-Key": key}, 404)
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_origin": origin,
        "scope": "health_ui_document_and_auth_only",
        "surface_checks_passed": all(c["status"] != "fail" for c in checks),
        "release_qualified": False,
        "checks": checks,
        "not_verified": ["video_processing", "recognition_quality", "gpu_offline_inference",
                         "browser_rendering_and_assets", "deployed_recipe_identity",
                         "legal_applicability", "license_qualification", "jury_key_handoff"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args(argv)
    destination = args.json.resolve()
    if destination.is_relative_to((ROOT / "data/raw").resolve()):
        parser.error("Audit output cannot be written in data/raw")
    if destination.exists():
        parser.error("Use a new JSON output path to preserve previous observations")
    key = args.key_file.read_text().strip() if args.key_file else None
    result = audit(args.api, key=key, timeout=args.timeout)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["surface_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
