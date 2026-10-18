"""Forward one localhost OAuth callback from stdin without logging its URL/code."""
import argparse
import base64
import http.client
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    request = json.loads(sys.stdin.buffer.readline(16385))
    target = request.get("path", "")
    if not 1024 <= args.port <= 65535 or not args.path.startswith("/callback/") or target.split("?", 1)[0] != args.path or any(c in target for c in "\r\n\0"):
        raise SystemExit("Invalid local callback route")
    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=15)
    try:
        connection.request("GET", target, headers={"Connection": "close"})
        response = connection.getresponse()
        body = response.read(65537)
        if len(body) > 65536:
            raise ValueError("Local callback response exceeds bound")
        print(json.dumps({"status": response.status, "type": response.getheader("Content-Type", "text/html"),
                          "body": base64.b64encode(body).decode()}))
    except Exception:
        raise SystemExit("Local callback forwarding failed") from None
    finally:
        connection.close()


if __name__ == "__main__":
    main()
