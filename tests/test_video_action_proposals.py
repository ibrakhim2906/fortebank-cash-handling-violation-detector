"""Local proposal safety using synthetic bytes, never supplied footage/models."""
import io
import json
import sys
from types import SimpleNamespace

import pytest

from src import video_action_proposals as proposals


@pytest.mark.parametrize("value", [
    {"actions": [], "visibility": "invisible"},
    {"actions": [{"action_id": "invented", "presence": "present"}], "visibility": "clear"},
    {"actions": [{"action_id": "object_handoff", "presence": "definite"}], "visibility": "clear"},
    {"actions": [], "visibility": "clear", "description": "unrequested model text"},
])
def test_rejects_unbounded_or_unrecognized_model_fields(value):
    with pytest.raises(ValueError):
        proposals.validate_observations(value)


@pytest.mark.parametrize("presence,visibility,expected", [
    ("present", "clear", "present"),
    ("present", "unobservable", "unknown"),
    ("absent", "clear", "unknown"),
])
def test_sampled_frames_never_establish_action_absence(presence, visibility, expected):
    result = proposals.validate_observations({
        "actions": [{"action_id": "object_handoff", "presence": presence}],
        "visibility": visibility})
    assert result["actions"][0]["presence"] == expected


def test_request_has_fixed_loopback_target_no_proxy_and_no_official_target(monkeypatch):
    response = {"choices": [{"message": {"content": json.dumps({
        "actions": {identifier: "unknown" for identifier in proposals.ACTION_VOCABULARY},
        "visibility": "partial"})}}]}
    handlers = []

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://127.0.0.1:8003/v1/chat/completions"
            body = json.loads(request.data)
            assert body["seed"] == proposals.config.SEED
            assert body["temperature"] == 0 and body["stream"] is False
            assert body["model"] == "local-action-review"
            assert body["response_format"]["json_schema"]["schema"] == proposals.OBSERVATION_SCHEMA
            assert set(body["messages"][0]) == {"role", "content"}
            assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            assert "official_target" not in request.data.decode()
            assert "anchor_sec" not in request.data.decode()
            return io.BytesIO(json.dumps(response).encode())

    def opener(*items):
        handlers.extend(items)
        return Opener()

    monkeypatch.setenv("HTTP_PROXY", "http://external-proxy.invalid")
    monkeypatch.setattr(proposals, "build_opener", opener)
    assert proposals.request_local_vision([b"\xff\xd8synthetic\xff\xd9"])["visibility"] == "partial"
    assert handlers[0].proxies == {}
    assert isinstance(handlers[1], proposals.NoRedirects)


def test_redirect_cannot_forward_frames_outside_loopback():
    with pytest.raises(ValueError, match="cannot follow redirects"):
        proposals.NoRedirects().redirect_request(None, None, 307, "redirect", {}, "https://example.com")


@pytest.mark.parametrize("port", ["8003", True, 80, 65536])
def test_no_arbitrary_endpoint_or_privileged_port(port):
    with pytest.raises(ValueError, match="port"):
        proposals.request_local_vision([b"\xff\xd8synthetic"], port=port)


def test_bad_response_never_exposes_model_text(monkeypatch):
    secret_text = "sensitive-local-observation"
    class Opener:
        def open(self, *args, **kwargs):
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": secret_text}}]}).encode())
    monkeypatch.setattr(proposals, "build_opener", lambda *args: Opener())
    with pytest.raises(RuntimeError) as error:
        proposals.request_local_vision([b"\xff\xd8synthetic"])
    assert secret_text not in str(error.value)


def test_runtime_must_be_verified_before_any_media_is_opened(monkeypatch):
    def unavailable():
        raise ValueError("Not provisioned")
    monkeypatch.setitem(sys.modules, "scripts.provision_action_model", SimpleNamespace(verify_ready=unavailable))
    monkeypatch.setattr(proposals, "sample_task", lambda *args: pytest.fail("Media must stay unopened"))
    with pytest.raises(ValueError, match="Not provisioned"):
        proposals.generate_proposals()


def test_existing_human_reviews_are_skipped_and_new_outputs_are_only_proposals(monkeypatch):
    model = {"model_id": "synthetic-local", "weights_sha256": "a" * 64, "projector_sha256": "b" * 64}
    monkeypatch.setitem(sys.modules, "scripts.provision_action_model", SimpleNamespace(verify_ready=lambda: model))
    monkeypatch.setattr(proposals, "load_queue", lambda *args: {"tasks": [{"task_id": "human"}, {"task_id": "new"}]})
    monkeypatch.setattr(proposals, "load_reviews", lambda *args: [{"task_id": "human", "provenance": {"kind": "human_review"}}])
    monkeypatch.setattr(proposals, "sample_task", lambda task: ([b"synthetic-frame"], [2.0]))
    monkeypatch.setattr(proposals, "request_local_vision", lambda *args, **kwargs: {"actions": [], "visibility": "partial"})
    saved = []
    monkeypatch.setattr(proposals, "append_proposal", lambda *args: saved.append(args))
    result = proposals.generate_proposals()
    assert result == {"tasks_attempted": 1, "proposals_saved": 1, "tasks_failed": 0,
                      "verified_labels_created": 0, "holdout_groups_used": 0}
    assert saved[0][1] == "new"
    assert saved[0][4]["kind"] == "local_model_proposal"
    assert saved[0][4]["recipe_sha256"] == proposals.RECIPE_SHA256
