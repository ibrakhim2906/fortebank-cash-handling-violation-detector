"""Draft action observations using an operator-provisioned loopback vision model.

This is an annotation assistant, not verified training truth or a detector. No
official target/timestamp, footage or description is sent to an external host.
Only aggregate status is printed; model responses remain in local sidecars.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import cv2
import numpy as np

from src import config
from src.video_actions import (ACTION_VOCABULARY, DEFAULT_ACTION_DIR, DEFAULT_PROTOCOL,
                               append_proposal, load_queue, load_reviews)


MODEL_ALIAS = "local-action-review"
MAX_RESPONSE_BYTES = 64 * 1024
FRAME_COUNT = 3
FRAME_MAX_SIDE = 384
PROMPT = (
    "Review these sampled frames in chronological order. Report only directly "
    "visible action primitives. Do not identify people, infer intent, financial "
    "correctness, rule violations, or hidden actions. Samples are incomplete: "
    "use unknown whenever an action cannot be established. A model proposal is "
    "not a verified label. Return only JSON with keys actions and visibility. "
    "actions maps every allowed action ID to present or unknown. "
    "visibility is clear, partial or unobservable. An action requires visible "
    "support across frames; a still object alone does not prove a handoff or "
    "repeated movement. Allowed action IDs and meanings: "
    + json.dumps(ACTION_VOCABULARY, sort_keys=True)
)
OBSERVATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "actions": {"type": "object", "additionalProperties": False,
                    "properties": {identifier: {"type": "string", "enum": ["present", "unknown"]}
                                   for identifier in ACTION_VOCABULARY},
                    "required": list(ACTION_VOCABULARY)},
        "visibility": {"type": "string", "enum": ["clear", "partial", "unobservable"]},
    }, "required": ["actions", "visibility"],
}
RECIPE = {"version": 2, "prompt": PROMPT, "response_schema": OBSERVATION_SCHEMA,
          "frame_count": FRAME_COUNT,
          "frame_max_side": FRAME_MAX_SIDE, "selection": "uniform_in_review_interval",
          "temperature": 0, "seed": config.SEED, "max_tokens": 256}
RECIPE_SHA256 = hashlib.sha256(json.dumps(RECIPE, sort_keys=True).encode()).hexdigest()


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Local vision requests cannot follow redirects")


def request_local_vision(images, *, port=8003, timeout=120):
    """Use only a literal loopback destination, without environment proxies."""
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Local vision port must be an unprivileged integer port")
    if not images or len(images) > FRAME_COUNT:
        raise ValueError("Local vision requires a bounded nonempty frame sample")
    contents = [{"type": "text", "text": PROMPT}]
    for payload in images:
        if not isinstance(payload, bytes) or not payload.startswith(b"\xff\xd8"):
            raise ValueError("Local model images must be in-memory JPEG bytes")
        contents.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")}})
    body = {"model": MODEL_ALIAS, "messages": [{"role": "user", "content": contents}],
            "temperature": 0, "seed": config.SEED, "max_tokens": 256,
            "stream": False, "response_format": {"type": "json_schema", "json_schema": {
                "name": "action_observations", "strict": True, "schema": OBSERVATION_SCHEMA}}}
    request = Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                      data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    opener = build_opener(ProxyHandler({}), NoRedirects())
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ValueError("Local model response is too large")
        envelope = json.loads(payload)
        content = envelope["choices"][0]["message"]["content"]
        observation = json.loads(content)
        states = observation.get("actions") if isinstance(observation, dict) else None
        if not isinstance(states, dict) or set(states) != set(ACTION_VOCABULARY):
            raise ValueError("Local model must return each declared action exactly once")
        observation["actions"] = [{"action_id": identifier, "presence": states[identifier]}
                                  for identifier in ACTION_VOCABULARY]
        return validate_observations(observation)
    except (OSError, HTTPError, ValueError, KeyError, IndexError, TypeError):
        # Never include model text, frame data or clip metadata in an exception.
        raise RuntimeError("Local vision did not return a valid action proposal") from None


def validate_observations(value):
    if not isinstance(value, dict) or set(value) != {"actions", "visibility"}:
        raise ValueError("Expected structured action observations")
    if value["visibility"] not in {"clear", "partial", "unobservable"}:
        raise ValueError("Unknown observation visibility")
    actions = value["actions"]
    if not isinstance(actions, list) or len(actions) > len(ACTION_VOCABULARY):
        raise ValueError("Expected bounded action observations")
    output, seen = [], set()
    for action in actions:
        if not isinstance(action, dict) or set(action) != {"action_id", "presence"}:
            raise ValueError("Action observation fields differ from the recipe")
        identifier, presence = action["action_id"], action["presence"]
        if identifier not in ACTION_VOCABULARY or identifier in seen:
            raise ValueError("Unknown or duplicated action observation")
        if presence not in {"present", "absent", "unknown"}:
            raise ValueError("Unknown action presence")
        # A few sampled frames cannot establish action absence over an interval.
        if presence == "absent" or value["visibility"] == "unobservable":
            presence = "unknown"
        if identifier == "unknown_observable_action":
            presence = "unknown"
        output.append({"action_id": identifier, "presence": presence})
        seen.add(identifier)
    return {"actions": output, "visibility": value["visibility"]}


def sample_task(task):
    """Decode only a source-verified development task, keeping pixels in memory."""
    import time
    from src.video_qwen_actions import _read_frame_at

    path = (config.ROOT / task["path"]).resolve()
    media_root = (config.DATA_PROC / "fortebank-20261017/open/videos").resolve()
    if (task.get("partition") != "development" or not path.is_relative_to(media_root)
            or path.suffix.lower() != ".mov"
            or hashlib.sha256(path.read_bytes()).hexdigest() != task["source_sha256"]):
        raise ValueError("Action source does not match a permitted development task")
    cap = cv2.VideoCapture(str(path))
    images, timestamps = [], []
    try:
        if not cap.isOpened():
            raise ValueError("Local action source could not be opened")
        start, end = task["interval"]["start_sec"], task["interval"]["end_sec"]
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Local action source has invalid frame timing")
        end = min(float(end), float(task["duration_sec"]) - 1 / fps)
        decoder_deadline = time.monotonic() + 30
        for timestamp in np.linspace(float(start), max(float(start), end), FRAME_COUNT):
            frame = _read_frame_at(cap, float(timestamp), fps, decoder_deadline)
            height, width = frame.shape[:2]
            scale = min(1.0, FRAME_MAX_SIDE / max(height, width))
            if scale < 1:
                frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            encoded, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not encoded:
                raise ValueError("A local action sample could not be encoded")
            images.append(data.tobytes())
            timestamps.append(float(timestamp))
    finally:
        cap.release()
    return images, timestamps


def generate_proposals(*, out_dir=DEFAULT_ACTION_DIR, protocol_path=DEFAULT_PROTOCOL,
                       port=8003, limit=3):
    """Populate local proposals after runtime provisioning, without training."""
    from scripts.provision_action_model import verify_ready

    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Choose between one and 100 local review tasks")
    # Validate local model provenance before any media is opened.
    model = verify_ready()
    queue = load_queue(out_dir, protocol_path)
    history = load_reviews(out_dir, protocol_path)
    existing = {record["task_id"] for record in history
                if record["provenance"]["kind"] == "human_review"
                or (record["provenance"].get("recipe_sha256") == RECIPE_SHA256
                    and record["provenance"].get("weights_sha256") == model["weights_sha256"])}
    tasks = [task for task in queue["tasks"] if task["task_id"] not in existing][:limit]
    saved, failed = 0, 0
    for task in tasks:
        try:
            images, timestamps = sample_task(task)
            observation = request_local_vision(images, port=port)
            provenance = {"kind": "local_model_proposal", "model_id": model["model_id"],
                          "weights_sha256": model["weights_sha256"],
                          "projector_sha256": model["projector_sha256"],
                          "recipe_sha256": RECIPE_SHA256,
                          "frame_sha256s": [hashlib.sha256(image).hexdigest() for image in images],
                          "sample_times_sec": timestamps}
            append_proposal(out_dir, task["task_id"], observation["actions"],
                            observation["visibility"], provenance, protocol_path)
            saved += 1
        except (OSError, ValueError, RuntimeError, cv2.error):
            failed += 1
    return {"tasks_attempted": len(tasks), "proposals_saved": saved,
            "tasks_failed": failed, "verified_labels_created": 0, "holdout_groups_used": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ACTION_DIR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = generate_proposals(out_dir=args.out_dir, protocol_path=args.protocol,
                                    port=args.port, limit=args.limit)
    except (OSError, ValueError, RuntimeError):
        raise SystemExit("Local action model is not ready; provision the approved runtime and start its loopback server") from None
    print(json.dumps(result), flush=True)
    if result["tasks_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
