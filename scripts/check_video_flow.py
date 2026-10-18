"""Verify the running local API; never print video pixels or descriptions."""
import argparse
import base64
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import joblib
import numpy as np

from src import config
from src.video_eval import load_protocol, protocol_groups


def request_json(url, key, body=None, boundary=None):
    headers = {"X-API-Key": key}
    if boundary:
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    with urlopen(Request(url, data=body, headers=headers), timeout=60) as response:
        return response.status, json.load(response)


def upload_and_wait(base_url, key, payload, filename):
    boundary = "datathon-local-verification-boundary"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            'Content-Type: video/quicktime\r\n\r\n').encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    started = time.monotonic()
    status, accepted = request_json(base_url + "/api/v1/analyze", key, body, boundary)
    assert status == 202 and set(accepted) == {"job_id"}
    while time.monotonic() - started < 300:
        status, result = request_json(base_url + "/api/v1/jobs/" + accepted["job_id"], key)
        assert status == 200
        if result["status"] != "processing":
            return result, time.monotonic() - started
        time.sleep(0.10)
    raise AssertionError("Local end-to-end job exceeded five minutes")


def validate_result(result, duration, expected):
    assert set(result) == {"job_id", "status", "verdict", "violations"}
    assert result["status"] == "done" and result["verdict"] == expected
    assert bool(result["violations"]) == (expected == "violation")
    frames = 0
    for event in result["violations"]:
        assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
        assert 0 <= event["timecode_sec"] < duration
        assert 1 <= len(event["evidence_frames"]) <= 3
        for evidence in event["evidence_frames"]:
            assert set(evidence) == {"timestamp_sec", "image_base64"}
            assert 0 <= evidence["timestamp_sec"] < duration
            payload = base64.b64decode(evidence["image_base64"], validate=True)
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image is not None and image.size > 0
            frames += 1
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "video-baseline-comparison")
    parser.add_argument("--key-file", type=Path, default=config.DATA_PROC / "video-service/api-key.txt")
    parser.add_argument("--protocol", type=Path, default=config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    args = parser.parse_args()
    key = args.key_file.read_text().strip()
    with urlopen(args.url + "/", timeout=10) as response:
        assert response.status == 200 and key.encode() not in response.read()
    for presented in ("", "incorrect-synthetic-key"):
        try:
            request_json(args.url + "/api/v1/jobs/unknown", presented)
        except HTTPError as error:
            assert error.code == 401
        else:
            raise AssertionError("Missing/incorrect API key was accepted")
    protocol = load_protocol(args.protocol)
    groups = protocol_groups(protocol, "development")
    model = joblib.load(args.run_dir / "models/baseline.joblib")
    assert model["protocol_fingerprint"] == protocol["fingerprint"]
    assert not set(model["training_groups"]) & set(protocol["holdout_group_ids"])
    cache = Path(json.loads((args.run_dir / "development.json").read_text())["cache_dir"])
    records = [joblib.load(cache / f"{g['group_id']}.joblib") for g in groups]
    if model.get("engine") == "mobilenet-temporal":
        probabilities = np.asarray([
            np.max(model["window_model"].predict_proba(
                np.stack([window["features"] for window in record["windows"]]))[:, 1])
            for record in records])
    else:
        features = np.stack([record["clip_features"] for record in records])
        probabilities = model["clip_model"].predict_proba(features)[:, 1]
    indices = [int(np.argmin(probabilities)), int(np.argmax(probabilities))]
    assert probabilities[indices[0]] < model["threshold"] <= probabilities[indices[1]]
    checks = []
    previous_positive = None
    for name, index in zip(("low_training_prediction", "high_training_prediction"), indices):
        group = groups[index]
        expected = "violation" if probabilities[index] >= model["threshold"] else "no_violation"
        payload = (config.ROOT / group["path"]).read_bytes()
        result, elapsed = upload_and_wait(args.url, key, payload, "unrelated-upload-name.mov")
        frames = validate_result(result, group["duration_sec"], expected)
        checks.append({"case": name, "status": result["status"], "verdict": result["verdict"],
                       "evidence_frames": frames, "elapsed_sec": elapsed, "upload_bytes": len(payload)})
        if expected == "violation":
            previous_positive = (payload, result, group["duration_sec"])
    payload, previous, duration = previous_positive
    renamed, elapsed = upload_and_wait(args.url, key, payload, "negative-label-filename.mov")
    validate_result(renamed, duration, "violation")
    assert {k: v for k, v in previous.items() if k != "job_id"} == {k: v for k, v in renamed.items() if k != "job_id"}
    checks.append({"case": "same_bytes_renamed", "identical_result": True, "elapsed_sec": elapsed})
    corrupt, elapsed = upload_and_wait(args.url, key, b"not-a-quicktime-video", "corrupt.mov")
    assert corrupt["status"] == "failed" and set(corrupt) == {"job_id", "status"}
    checks.append({"case": "corrupt_file", "status": "failed", "elapsed_sec": elapsed})
    result = {"ui_http_status": 200, "missing_and_wrong_key_status": 401,
              "checks": checks, "holdout_videos_used": 0,
              "notes": "Engineering flow checks on development examples; not an independent accuracy estimate. No frames or descriptions emitted to tools."}
    report = args.run_dir / "api_check.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    print("Local API verification report:", report, flush=True)


if __name__ == "__main__":
    main()
