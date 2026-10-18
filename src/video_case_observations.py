"""Fixed local static-state diagnostic, never a verdict/production feature engine."""
import base64
import hashlib
import json
import math

import cv2
import numpy as np

from src import config


MODEL_ALIAS = "local-action-review"
FIELDS = {
    "cabinet_state": ("open", "closed", "unknown"),
    "banknotes_visible": ("yes", "no", "unknown"),
    "banknotes_on_working_table": ("yes", "no", "unknown"),
    "person_state": ("at_desk", "elsewhere_in_room", "absent", "unknown"),
    "coat_or_handbag_on_surface": ("yes", "no", "unknown"),
    "visibility": ("clear", "limited", "insufficient"),
}
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {key: {"type": "string", "enum": list(values)}
                         for key, values in FIELDS.items()}, "required": list(FIELDS)}
PROMPT = (
    "The two images show the SAME INSTANT: a complete scene and a sharper workspace detail. "
    "Describe only literal visible states, using unknown when hidden, ambiguous or too small. "
    "cabinet_state: whether the metal storage cabinet door is visibly open or closed; "
    "unknown if no identifiable cabinet or its door cannot be assessed. "
    "banknotes_visible: yes only for identifiable banknotes or banknote stacks, not arbitrary paper. "
    "banknotes_on_working_table: whether banknotes rest on the desk/work surface; "
    "unknown if that surface cannot be assessed. "
    "person_state: at_desk if a person is seated/working at the desk, elsewhere_in_room if "
    "a person is visible away from it, absent if nobody is visible, otherwise unknown. "
    "Ignore people visible only through a glass wall or reflections when determining room presence. "
    "coat_or_handbag_on_surface: whether an unworn coat or handbag is placed on the working surface; "
    "do not count worn clothing, chairs or an operational fabric collection pouch as a coat/handbag. "
    "visibility: clear, limited or insufficient for assessing these states. "
    "Do not infer identity, authorization, ownership, intent, amounts, past/future actions, "
    "compliance, laptop lock state or legal conclusions. Return only the requested JSON."
)
RECIPE = {"version": 1, "purpose": "sampled_static_observer_qualification_only",
          "model_alias": MODEL_ALIAS, "fields": FIELDS, "prompt": PROMPT, "schema": SCHEMA,
          "fractions": [0.2, 0.5, 0.9], "views": ["overview", "workspace_detail"],
          "detail_normalized_xyxy": [0, 0.18, 1, 0.72], "max_side": 848,
          "jpeg_quality": 92, "seed": config.SEED, "temperature": 0,
          "cache_prompt": False, "max_tokens": 256, "context_tokens": 4096,
          "request_timeout_sec": 45, "total_timeout_sec": 900,
          "target_requests": 18, "synthetic_requests": 1, "repeat_requests": 1,
          "endpoint": "http://127.0.0.1:8003/v1/chat/completions"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def image_views(frame):
    """Preserve aspect ratio and native detail; never generate face-specific crops."""
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 8:
        raise ValueError("Static observation needs a valid color frame")
    height, width = frame.shape[:2]
    detail = frame[round(height * .18):round(height * .72), :width]
    payloads = []
    for view in (frame, detail):
        scale = min(1., RECIPE["max_side"] / max(view.shape[:2]))
        if scale < 1:
            view = cv2.resize(view, (round(view.shape[1] * scale), round(view.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ok, data = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, RECIPE["jpeg_quality"]])
        if not ok:
            raise ValueError("Static observation JPEG encoding failed")
        payloads.append(data.tobytes())
    return payloads


def request_body(images):
    if not isinstance(images, (list, tuple)) or len(images) != 2:
        raise ValueError("Static observation requires exactly two same-instant JPEG views")
    content = [{"type": "text", "text": PROMPT}]
    for payload in images:
        if not isinstance(payload, bytes) or not 0 < len(payload) <= 1024 * 1024 or not payload.startswith(b"\xff\xd8"):
            raise ValueError("Static observation image is not a bounded JPEG")
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or max(decoded.shape[:2]) > RECIPE["max_side"]:
            raise ValueError("Static observation JPEG dimensions are invalid")
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")}})
    return {"model": MODEL_ALIAS, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "seed": config.SEED, "cache_prompt": False,
            "max_tokens": RECIPE["max_tokens"], "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "case_visible_states", "strict": True, "schema": SCHEMA}}}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Static observation has duplicate JSON keys")
        result[key] = value
    return result


def validate_response(response):
    """Reject truncation, wrong models, context overflow and schema deviations."""
    try:
        if response.get("model") != MODEL_ALIAS or response.get("truncated") is True:
            raise ValueError
        choices = response["choices"]
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError
        message = choices[0]["message"]
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            raise ValueError
        value = json.loads(message["content"], object_pairs_hook=_unique)
        if not isinstance(value, dict) or set(value) != set(FIELDS) or any(value[k] not in states for k, states in FIELDS.items()):
            raise ValueError
        usage = response["usage"]
        if any(type(usage[k]) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            raise ValueError
        if (usage["prompt_tokens"] + RECIPE["max_tokens"] > RECIPE["context_tokens"]
                or usage["completion_tokens"] > RECIPE["max_tokens"]
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
            raise ValueError
        return {key: value[key] for key in FIELDS}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise ValueError("Local static response failed model, completion, context or state validation") from None


def observe(images, timeout=45, *, requester=None):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 45:
        raise ValueError("Static observation timeout must be positive and at most45 seconds")
    if requester is None:
        # This frozen transport disables proxies/redirects and kills its spawned
        # worker on a whole-request deadline; only fixed loopback routes exist.
        from scripts.check_temporal_protocol import request_json
        requester = request_json
    body = request_body(images)
    response = requester("/v1/chat/completions", body, timeout=timeout)
    return {"states": validate_response(response), "body_sha256": fingerprint(body),
            "response": response}
