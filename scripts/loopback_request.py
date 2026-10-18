"""One private stdin/stdout HTTP exchange to the WSL loopback video service.

Used by the Windows loopback bridge. No headers, keys or media are logged.
Each supported service has a fixed loopback destination. Callers cannot choose
an arbitrary host or port. The optional review service stays on the computer.
"""
import http.client
import json
import re
import sys


MAX_REQUEST = 64 * 1024 * 1024 + 128 * 1024
MAX_RESPONSE = 8 * 1024 * 1024


def main(argv=None):
    connection = None
    try:
        arguments = list(sys.argv[1:] if argv is None else argv)
        service = "video"
        if arguments:
            if len(arguments) != 2 or arguments[0] != "--service" or arguments[1] not in {"video", "review"}:
                raise ValueError("Unsupported relay service")
            service = arguments[1]
        line = sys.stdin.buffer.readline(16 * 1024 + 1)
        if len(line) > 16 * 1024 or not line.endswith(b"\n"):
            raise ValueError("Invalid relay metadata")
        metadata = json.loads(line)
        method, path = metadata["method"], metadata["path"]
        if service == "review":
            allowed = ((method == "GET" and path in {"/", "/api/review/tasks", "/api/review/summary",
                                                    "/api/review/transfer/tasks", "/api/review/transfer/summary",
                                                    "/api/review/pretrained/tasks", "/api/review/pretrained/summary"})
                       or (method == "GET" and re.fullmatch(r"/api/review/(?:(?:transfer|pretrained)/)?media/[0-9a-f]{64}", path))
                       or (method == "POST" and path in {"/api/review/labels", "/api/review/transfer/labels"}))
        else:
            allowed = ((method == "GET" and path == "/")
                       or (method == "POST" and path == "/api/v1/analyze")
                       or (method == "GET" and re.fullmatch(r"/api/v1/jobs/[0-9a-f]{32}", path)))
        length = metadata["content_length"]
        if not allowed or type(length) is not int or not 0 <= length <= MAX_REQUEST:
            raise ValueError("Unsupported relay request")
        if method == "GET" and length:
            raise ValueError("GET request bodies are not supported")
        body = sys.stdin.buffer.read(length)
        if len(body) != length:
            raise ValueError("Incomplete relay input")
        headers = {name: str(value) for name, value in metadata.get("headers", {}).items()
                   if name.lower() in {"content-type", "x-api-key"}}
        connection = http.client.HTTPConnection("127.0.0.1", 8002 if service == "review" else 8000, timeout=55)
        connection.request(method, path, body=body if method == "POST" else None, headers=headers)
        response = connection.getresponse()
        maximum = MAX_REQUEST if service == "review" else MAX_RESPONSE
        payload = response.read(maximum + 1)
        if len(payload) > maximum:
            raise ValueError("Upstream response exceeds relay limit")
        allowed_headers = {"content-type", "cache-control", "content-security-policy",
                           "x-content-type-options", "referrer-policy", "retry-after"}
        output_headers = {name.lower(): value for name, value in response.getheaders()
                          if name.lower() in allowed_headers}
        status = response.status
    except Exception:
        status = 502
        output_headers = {"content-type": "application/json", "cache-control": "no-store"}
        payload = b'{"detail":"Local WSL service is unavailable"}'
    finally:
        if connection is not None:
            connection.close()
    sys.stdout.buffer.write(json.dumps({"status": status, "headers": output_headers}).encode() + b"\n")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
