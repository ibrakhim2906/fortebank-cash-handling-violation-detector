"""Strict loopback-only static observations and unverified local action review.

Sampled-image observations are model outputs, not verified action truth. A
second pass by the same model checks consistency only. No media, proposals or
model text is printed, logged or included in exceptions by this module.
"""
import base64
import hashlib
import json
import math
from urllib.request import ProxyHandler, Request, build_opener

from src import config
from src.video_action_proposals import NoRedirects, validate_observations
from src.video_actions import ACTION_VOCABULARY


FIELDS = {
    "people_count": ("none", "one", "two", "three_plus", "unclear"),
    "visible_hands": ("none", "one", "two", "three_plus", "unclear"),
    "hand_object_contact": ("visible", "not_visible", "unclear"),
    "rectangular_sheets_or_stacks": ("visible", "not_visible", "unclear"),
    "handheld_device": ("visible", "not_visible", "unclear"),
    "open_container": ("visible", "not_visible", "unclear"),
    "work_surface": ("visible", "not_visible", "unclear"),
    "visibility": ("clear", "partial", "unobservable"),
}
OBSERVATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {field: {"type": "string", "enum": list(states)}
                   for field, states in FIELDS.items()},
    "required": list(FIELDS),
}
ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "actions": {"type": "object", "additionalProperties": False,
                    "properties": {identifier: {"type": "string", "enum": ["present", "unknown"]}
                                   for identifier in ACTION_VOCABULARY},
                    "required": list(ACTION_VOCABULARY)},
        "visibility": {"type": "string", "enum": ["clear", "partial", "unobservable"]},
    }, "required": ["actions", "visibility"],
}
OBSERVATION_PROMPT = (
    "These images show the SAME instant: an overview and two detail crops. "
    "Report only visible contents at this instant. Use unclear if hidden or unreadable. "
    "not_visible means not visible in these images, never absent throughout the video. "
    "Count visible people and hands; report visible hand-object contact, rectangular "
    "sheets/stacks, handheld devices, open containers and work surface. Do not infer "
    "motion, identity, intent, financial correctness, rule compliance or unseen facts. "
    "Return only the requested JSON categories."
)
ACTION_DRAFT_PROMPT = (
    "Inspect these chronological video frames. Draft only directly visible action "
    "primitives. Use present when supported by the images, otherwise unknown. "
    "Do not identify people or infer intent, legal conclusions, rule compliance, "
    "financial correctness or unseen facts. This is an unverified model proposal. "
    "Return JSON with actions mapping every allowed ID to present or unknown, and "
    "visibility clear, partial or unobservable. Allowed IDs and meanings: "
    + json.dumps(ACTION_VOCABULARY, sort_keys=True)
)
ACTION_CHECK_PROMPT = (
    "Check the previous unverified model proposal against these chronological "
    "video frames. Revise every action to present only when visibly supported, "
    "otherwise unknown. Do not identify people or infer intent, legal conclusions, "
    "rule compliance, financial correctness or unseen facts. Agreement with a "
    "previous model output is a consistency check, never independent verification. "
    "Return JSON with actions mapping every allowed ID to present or unknown, and "
    "visibility clear, partial or unobservable. Allowed IDs and meanings: "
    + json.dumps(ACTION_VOCABULARY, sort_keys=True)
    + " Previous unverified action states: "
)
MODEL_ALIAS = "local-action-review"
MAX_RESPONSE_BYTES = 64 * 1024
MAX_IMAGE_BYTES = 1024 * 1024
RECIPE = {
    "version": 3, "model_alias": MODEL_ALIAS,
    "static_prompt": OBSERVATION_PROMPT, "static_schema": OBSERVATION_SCHEMA,
    "action_draft_prompt": ACTION_DRAFT_PROMPT, "action_check_prompt": ACTION_CHECK_PROMPT,
    "action_schema": ACTION_SCHEMA,
    "prior_action_serialization": "vocabulary-order present/unknown map; missing states unknown",
    "static_max_side": 640, "action_max_side": 640, "max_images": 3,
    "temperature": 0, "seed": config.SEED, "max_tokens": 256,
    "max_timeout_seconds": 45, "endpoint": "http://127.0.0.1:8003/v1/chat/completions",
}
RECIPE_SHA256 = hashlib.sha256(json.dumps(RECIPE, sort_keys=True).encode()).hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate structured JSON field")
        value[key] = item
    return value


def _request(images, prompt, schema, timeout):
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= 45):
        raise ValueError("Local observation timeout must be finite and at most 45 seconds")
    if not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 3:
        raise ValueError("Local observations require one to three bounded JPEGs")
    contents = [{"type": "text", "text": prompt}]
    for image in images:
        if not isinstance(image, bytes) or not image.startswith(b"\xff\xd8") or len(image) > MAX_IMAGE_BYTES:
            raise ValueError("Local observations require bounded in-memory JPEG bytes")
        contents.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")}})
    body = {"model": MODEL_ALIAS, "messages": [{"role": "user", "content": contents}],
            "temperature": 0, "seed": config.SEED, "max_tokens": 256, "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "local_observations", "strict": True, "schema": schema}}}
    request = Request(RECIPE["endpoint"], data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    opener = build_opener(ProxyHandler({}), NoRedirects())
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ValueError("Local observation response exceeds its bound")
        envelope = json.loads(payload, object_pairs_hook=_unique_object)
        content = envelope["choices"][0]["message"]["content"]
        return json.loads(content, object_pairs_hook=_unique_object)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError("Local model did not return valid structured observations") from None


def request_observations(images, timeout=45):
    """Return eight strict static fields from local same-instant image views."""
    value = _request(images, OBSERVATION_PROMPT, OBSERVATION_SCHEMA, timeout)
    if (not isinstance(value, dict) or set(value) != set(FIELDS)
            or any(not isinstance(value[field], str) or value[field] not in states
                   for field, states in FIELDS.items())):
        raise RuntimeError("Local model static observations differ from the fixed schema")
    return {field: value[field] for field in FIELDS}


def request_action_review(images, prior_actions=None, timeout=45):
    """Draft or check actions locally; neither path creates verified truth."""
    prompt = ACTION_DRAFT_PROMPT
    if prior_actions is not None:
        try:
            previous = validate_observations({"actions": prior_actions, "visibility": "partial"})
        except (ValueError, KeyError, TypeError):
            raise ValueError("Previous action proposal differs from the fixed vocabulary") from None
        states = {item["action_id"]: item["presence"] for item in previous["actions"]}
        prior = {identifier: states.get(identifier, "unknown") for identifier in ACTION_VOCABULARY}
        prompt = ACTION_CHECK_PROMPT + json.dumps(prior, separators=(",", ":"))
    value = _request(images, prompt, ACTION_SCHEMA, timeout)
    try:
        if not isinstance(value, dict) or set(value) != {"actions", "visibility"}:
            raise ValueError("Action proposal fields differ from the schema")
        states = value["actions"]
        if (not isinstance(states, dict) or set(states) != set(ACTION_VOCABULARY)
                or any(state not in {"present", "unknown"} for state in states.values())):
            raise ValueError("Action proposal states differ from the vocabulary")
        value["actions"] = [{"action_id": identifier, "presence": states[identifier]}
                            for identifier in ACTION_VOCABULARY]
        return validate_observations(value)
    except (ValueError, KeyError, TypeError):
        raise RuntimeError("Local model action review differs from the fixed schema") from None
