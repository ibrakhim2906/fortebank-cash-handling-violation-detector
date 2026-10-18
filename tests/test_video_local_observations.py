"""Mocked local HTTP only: no model requests, source footage or frames."""
import hashlib
import json
from urllib.error import URLError

import pytest

from src import video_local_observations as V


JPEG = b"\xff\xd8synthetic-jpeg-placeholder\xff\xd9"


def _static():
    return {field: states[0] for field, states in V.FIELDS.items()}


def _actions():
    return {"actions": {identifier: "unknown" for identifier in V.ACTION_VOCABULARY},
            "visibility": "clear"}


@pytest.fixture
def local_http(monkeypatch):
    state = {"value": _static(), "calls": [], "handlers": [], "raw_response": None}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, bound):
            assert bound == V.MAX_RESPONSE_BYTES + 1
            if state["raw_response"] is not None:
                return state["raw_response"][:bound]
            content = json.dumps(state["value"])
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    class Opener:
        def open(self, request, timeout):
            state["calls"].append((request, timeout))
            return Response()

    def build(*handlers):
        state["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(V, "build_opener", build)
    return state


def test_fixed_fields_and_recipe_hash_are_explicit_and_deterministic():
    assert list(V.FIELDS) == ["people_count", "visible_hands", "hand_object_contact",
                              "rectangular_sheets_or_stacks", "handheld_device",
                              "open_container", "work_surface", "visibility"]
    assert sum(map(len, V.FIELDS.values())) == 28
    assert V.FIELDS["people_count"] == ("none", "one", "two", "three_plus", "unclear")
    assert V.OBSERVATION_SCHEMA["required"] == list(V.FIELDS)
    assert V.OBSERVATION_SCHEMA["additionalProperties"] is False
    assert V.RECIPE_SHA256 == hashlib.sha256(json.dumps(V.RECIPE, sort_keys=True).encode()).hexdigest()
    assert "SAME instant" in V.OBSERVATION_PROMPT
    assert "never absent throughout the video" in V.OBSERVATION_PROMPT
    assert "unverified" in V.ACTION_DRAFT_PROMPT
    assert "never independent verification" in V.ACTION_CHECK_PROMPT


def test_observations_use_only_loopback_without_proxies_or_redirects(local_http):
    assert V.request_observations([JPEG] * 3, timeout=12) == _static()
    request, timeout = local_http["calls"][0]
    assert request.full_url == "http://127.0.0.1:8003/v1/chat/completions"
    assert timeout == 12
    assert local_http["handlers"][0].proxies == {}
    assert isinstance(local_http["handlers"][1], V.NoRedirects)
    body = json.loads(request.data)
    assert body["temperature"] == 0 and body["seed"] == 42 and body["max_tokens"] == 256
    assert body["response_format"]["json_schema"]["schema"] == V.OBSERVATION_SCHEMA
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["messages"][0]["content"][0]["text"] == V.OBSERVATION_PROMPT
    assert len(body["messages"][0]["content"]) == 4


def test_static_not_visible_stays_sample_specific_and_unknowns_are_preserved(local_http):
    value = {field: states[-1] for field, states in V.FIELDS.items()}
    value["hand_object_contact"] = "not_visible"
    value["visibility"] = "partial"
    local_http["value"] = value
    assert V.request_observations([JPEG]) == value


@pytest.mark.parametrize("value", [
    {}, {**_static(), "invented_rule": "unsupported"},
    {**_static(), "people_count": 0}, {**_static(), "people_count": "four"},
    {**_static(), "visibility": "adequate"},
    {**_static(), "handheld_device": ["visible"]},
])
def test_bad_static_schema_never_emits_model_content(local_http, value):
    local_http["value"] = value
    with pytest.raises(RuntimeError, match="fixed schema") as failure:
        V.request_observations([JPEG])
    assert "unsupported" not in str(failure.value)


@pytest.mark.parametrize("images", [[], [JPEG] * 4, ["string"], [b"not-jpeg"],
                                      [b"\xff\xd8" + b"x" * V.MAX_IMAGE_BYTES]])
def test_invalid_or_oversized_images_never_reach_http(local_http, images):
    with pytest.raises(ValueError):
        V.request_observations(images)
    assert local_http["calls"] == []


@pytest.mark.parametrize("timeout", [0, -1, 46, float("inf"), float("nan"), True, "45"])
def test_invalid_timeout_never_reaches_http(local_http, timeout):
    with pytest.raises(ValueError):
        V.request_observations([JPEG], timeout=timeout)
    assert local_http["calls"] == []


@pytest.mark.parametrize("raw", [b"private model text", b"[]", b"{}", b"x" * (64 * 1024 + 1),
                                  b'{"choices":[],"choices":[]}',
                                  b'{"choices":[{"message":{"content":"{\\"people_count\\":\\"one\\",\\"people_count\\":\\"two\\"}"}}]}'])
def test_malformed_duplicate_or_oversized_response_is_private(local_http, raw):
    local_http["raw_response"] = raw
    with pytest.raises(RuntimeError) as failure:
        V.request_observations([JPEG])
    assert "private model text" not in str(failure.value)


def test_network_errors_are_sanitized(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            raise URLError("sensitive local model details")

    monkeypatch.setattr(V, "build_opener", lambda *args: Opener())
    with pytest.raises(RuntimeError) as failure:
        V.request_observations([JPEG])
    assert "sensitive" not in str(failure.value)


def test_draft_actions_return_old_normalized_observation_schema(local_http):
    local_http["value"] = _actions()
    local_http["value"]["actions"]["hand_object_contact"] = "present"
    value = V.request_action_review([JPEG] * 3)
    assert value["visibility"] == "clear"
    assert {item["action_id"] for item in value["actions"]} == set(V.ACTION_VOCABULARY)
    assert next(item for item in value["actions"] if item["action_id"] == "hand_object_contact")["presence"] == "present"
    body = json.loads(local_http["calls"][0][0].data)
    assert body["messages"][0]["content"][0]["text"] == V.ACTION_DRAFT_PROMPT
    assert body["response_format"]["json_schema"]["schema"] == V.ACTION_SCHEMA


def test_same_model_action_check_is_unverified_and_prior_is_strict(local_http):
    local_http["value"] = _actions()
    previous = [{"action_id": "hand_object_contact", "presence": "present"}]
    V.request_action_review([JPEG] * 3, prior_actions=previous)
    prompt = json.loads(local_http["calls"][0][0].data)["messages"][0]["content"][0]["text"]
    assert prompt.startswith(V.ACTION_CHECK_PROMPT)
    assert "consistency check" in prompt and "never independent verification" in prompt
    prior = json.loads(prompt[len(V.ACTION_CHECK_PROMPT):])
    assert prior["hand_object_contact"] == "present"
    assert prior["object_handoff"] == "unknown"
    assert set(prior) == set(V.ACTION_VOCABULARY)


def test_arbitrary_prior_text_cannot_enter_the_model_prompt(local_http):
    with pytest.raises(ValueError, match="Previous action proposal"):
        V.request_action_review([JPEG], prior_actions="private free text")
    assert local_http["calls"] == []


@pytest.mark.parametrize("value", [
    {"actions": {}, "visibility": "clear"},
    {**_actions(), "visibility": "invented"},
    {**_actions(), "comment": "private unstructured opinion"},
    {"actions": {**_actions()["actions"], "hand_object_contact": "absent"}, "visibility": "clear"},
])
def test_action_review_requires_exact_map_and_present_unknown_only(local_http, value):
    local_http["value"] = value
    with pytest.raises(RuntimeError, match="fixed schema") as failure:
        V.request_action_review([JPEG])
    assert "private unstructured" not in str(failure.value)


def test_unobservable_action_review_cannot_assert_present(local_http):
    local_http["value"] = _actions()
    local_http["value"]["actions"]["hand_object_contact"] = "present"
    local_http["value"]["visibility"] = "unobservable"
    value = V.request_action_review([JPEG])
    assert all(item["presence"] == "unknown" for item in value["actions"])
