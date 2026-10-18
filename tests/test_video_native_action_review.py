"""Synthetic local video and mocked requests only; no model or bank footage."""
from contextlib import contextmanager
import hashlib
import json
import time
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src import video_native_action_review as R
from src.video_actions import ACTION_IDS


def proposal(present=()):
    return {"visibility": "clear", "actions": [{"action_id": identifier,
             "presence": "present" if identifier in present else "unknown"} for identifier in ACTION_IDS]}


def envelope(present=()):
    value = {"visibility": "clear", "actions": {identifier: "present" if identifier in present else "unknown"
                                                for identifier in ACTION_IDS}}
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}


def task(identifier="one", group="dev"):
    return {"task_id": identifier, "group_id": group, "source_sha256": "a" * 64,
            "path": "private-label-file.mov", "partition": "development",
            "official_target": 1, "anchor_sec": 7., "interval": {"start_sec": 0., "end_sec": 2.}}


def samples(count=4):
    return [{"actual_sec": index * .5, "source_frame_index": index * 15,
             "source_frame_sha256": "b" * 64, "jpeg_sha256": "c" * 64} for index in range(count)]


@pytest.mark.parametrize("start,end,expected", [
    (0., 2., [0., .5, 1., 1.5]), (1., 3.5, [1., 1.5, 2., 2.5]),
    (0., 20., [index * .5 for index in range(16)]),
])
def test_sampling_is_contiguous_even_and_never_padded(start, end, expected):
    np.testing.assert_array_equal(R.interval_sample_times(start, end), expected)


@pytest.mark.parametrize("start,end", [(0., 1.5), (0., 1.), (2., 2.), (-1., 2.),
                                      (0., 301.), (True, 2.), (0., float("inf"))])
def test_short_or_invalid_intervals_fail(start, end):
    with pytest.raises(ValueError):
        R.interval_sample_times(start, end)


def test_native_request_is_fixed_loopback_video_with_times_and_sanitized_prior(monkeypatch):
    transport = Mock(return_value=envelope((ACTION_IDS[0],)))
    monkeypatch.setattr(R._protocol, "request_json", transport)
    first = R.request_native_review(b"RIFFsynthetic-AVI", samples(), {"alias": "local-action-review"}, time.monotonic() + 10)
    R.request_native_review(b"RIFFsynthetic-AVI", samples(), {"alias": "local-action-review"},
                           time.monotonic() + 10, first["actions"])
    for call in transport.call_args_list:
        assert call.args[0] == "/v1/chat/completions"
        body = call.args[1]
        assert body["seed"] == R.config.SEED and body["temperature"] == 0 and body["max_tokens"] == 256
        assert body["response_format"]["json_schema"]["schema"] == R.ACTION_SCHEMA
        text, video = body["messages"][0]["content"]
        assert text["text"].startswith("Chronological video-frame index to actual original source seconds: 0=0.000000s")
        assert "3=1.500000s" in text["text"]
        assert "private-label-file" not in text["text"] and "official_target" not in text["text"] and "anchor_sec" not in text["text"]
        assert video["type"] == "input_video" and set(video["input_video"]) == {"data"}
        assert 0 < call.kwargs["timeout"] <= 10
    assert "Previous unverified action states" in transport.call_args_list[1].args[1]["messages"][0]["content"][0]["text"]


def test_transport_error_does_not_expose_private_response(monkeypatch):
    monkeypatch.setattr(R._protocol, "request_json", Mock(side_effect=RuntimeError("private model output")))
    with pytest.raises(RuntimeError, match="request failed") as error:
        R.request_native_review(b"RIFFsynthetic", samples(), {"alias": "local-action-review"}, time.monotonic() + 5)
    assert "private model output" not in str(error.value)


@pytest.mark.parametrize("defect", ["length", "missing", "extra", "bad_state", "duplicate"])
def test_response_requires_complete_strict_schema_and_stopped_generation(defect):
    response = envelope()
    if defect == "length":
        response["choices"][0]["finish_reason"] = "length"
    else:
        value = json.loads(response["choices"][0]["message"]["content"])
        if defect == "missing":
            value["actions"].pop(ACTION_IDS[-1])
        elif defect == "extra":
            value["rule"] = "unsupported"
        elif defect == "bad_state":
            value["actions"][ACTION_IDS[0]] = "absent"
        if defect == "duplicate":
            response["choices"][0]["message"]["content"] = json.dumps(value)[:-1] + ',"visibility":"clear"}'
        else:
            response["choices"][0]["message"]["content"] = json.dumps(value)
    with pytest.raises(ValueError, match="fixed action schema"):
        R.parse_proposal(response)


@pytest.fixture
def local_run(tmp_path, monkeypatch):
    events = []
    model = {"model_id": "synthetic-model", "alias": "local-action-review", "weights_sha256": "b" * 64,
             "projector_sha256": "c" * 64, "runtime_release": "synthetic-runtime"}
    decoder = {"archive_sha256": "d" * 64}
    verify_model = Mock(side_effect=lambda: events.append("model") or model)
    verify_decoder = Mock(side_effect=lambda: events.append("decoder") or decoder)
    gate = Mock(side_effect=lambda *args: events.append("gate"))

    @contextmanager
    def server(*args):
        events.append("start")
        try:
            yield
        finally:
            events.append("stop")

    queue = {"tasks": [task()]}
    history, append = Mock(return_value=[]), Mock()
    sample = Mock(side_effect=lambda *args: events.append("sample") or (b"RIFFsynthetic", samples()))
    request = Mock(return_value=proposal((ACTION_IDS[0],)))
    monkeypatch.setattr(R, "verify_model", verify_model)
    monkeypatch.setattr(R, "verify_decoder", verify_decoder)
    monkeypatch.setattr(R._protocol, "verify_temporal_gate", gate)
    monkeypatch.setattr(R._protocol, "native_server", server)
    monkeypatch.setattr(R, "review_recipe", lambda: ({"synthetic": True}, "f" * 64))
    monkeypatch.setattr(R, "load_queue", lambda *args: queue)
    monkeypatch.setattr(R, "load_protocol", lambda *args: {"fingerprint": "synthetic-frozen-protocol",
                        "development_group_ids": ["dev"], "holdout_group_ids": ["reserved"]})
    monkeypatch.setattr(R, "load_reviews", history)
    monkeypatch.setattr(R, "sample_task_video", sample)
    monkeypatch.setattr(R, "request_native_review", request)
    monkeypatch.setattr(R, "append_proposal", append)
    return {"directory": tmp_path / "actions", "events": events, "queue": queue, "history": history,
            "model": verify_model, "decoder": verify_decoder, "gate": gate, "sample": sample,
            "request": request, "append": append}


def run(fixture, **kwargs):
    return R.review_tasks(out_dir=fixture["directory"], protocol_path="synthetic-protocol", gate_path="synthetic-gate", **kwargs)


def test_assets_and_gate_precede_media_once_and_server_is_cleaned_up(local_run):
    result = run(local_run)
    assert local_run["events"] == ["model", "decoder", "gate", "start", "sample", "stop"]
    local_run["model"].assert_called_once()
    local_run["decoder"].assert_called_once()
    assert local_run["request"].call_count == 2 and local_run["append"].call_count == 1
    assert result["machine_rechecked_proposals"] == 1 and result["verified_labels_created"] == 0
    assert result["semantic_accuracy_verified"] is False and result["bulk_expansion_allowed"] is False
    call = local_run["append"].call_args
    assert call.args[4]["kind"] == "local_model_proposal"
    journal = local_run["directory"] / "native-review-v1/audits.jsonl"
    record = json.loads(journal.read_text())
    assert record["status"] == "completed" and record["verified"] is False
    assert len(record["samples"]) == 4 and record["evidence_sample_indices"] == [0, 1, 3]
    assert "reviewer_id" not in record and "official_target" not in record


def test_failed_synthetic_gate_stops_before_server_or_sampling(local_run):
    local_run["gate"].side_effect = ValueError("synthetic gate failed")
    with pytest.raises(ValueError, match="gate failed"):
        run(local_run)
    assert local_run["events"] == ["model", "decoder"]
    local_run["sample"].assert_not_called()


def test_holdout_and_unknown_group_cannot_enter_native_server(local_run):
    for group in ("reserved", "unknown"):
        local_run["queue"]["tasks"] = [task(group=group)]
        with pytest.raises(ValueError, match="Reserved or unknown"):
            run(local_run)
    assert "start" not in local_run["events"]
    local_run["sample"].assert_not_called()


def test_human_review_and_existing_native_recipe_are_skipped(local_run):
    local_run["history"].return_value = [{"task_id": "one", "provenance": {"kind": "human_review"}}]
    assert run(local_run)["tasks_attempted"] == 0
    local_run["history"].return_value = [{"task_id": "one", "provenance": {
        "kind": "local_model_proposal", "recipe_sha256": "f" * 64}}]
    assert run(local_run)["tasks_attempted"] == 0
    local_run["sample"].assert_not_called()


def test_fixed_first_three_pool_cannot_expand_on_repeat_invocation(local_run):
    local_run["queue"]["tasks"] = [task(str(index)) for index in range(5)]
    assert run(local_run)["tasks_attempted"] == 3
    assert local_run["sample"].call_count == 3
    assert run(local_run)["tasks_attempted"] == 0
    assert local_run["sample"].call_count == 3
    with pytest.raises(ValueError, match="three-task"):
        run(local_run, limit=4)


def test_human_review_after_first_run_never_expands_frozen_three_task_pool(local_run):
    local_run["queue"]["tasks"] = [task(str(index)) for index in range(5)]
    assert run(local_run, limit=1)["tasks_attempted"] == 1
    local_run["history"].return_value = [{"task_id": "0", "provenance": {"kind": "human_review"}}]
    assert run(local_run)["tasks_attempted"] == 2
    assert [call.args[0]["task_id"] for call in local_run["sample"].call_args_list] == ["0", "1", "2"]
    paths = list((local_run["directory"] / "native-review-v1").glob("selection-*.json"))
    assert len(paths) == 1
    selection = json.loads(paths[0].read_text())
    assert selection["task_ids"] == ["0", "1", "2"]
    assert selection["protocol_fingerprint"] == "synthetic-frozen-protocol"
    assert len(selection["queue_sha256"]) == 64


def test_frozen_selection_refuses_changed_queue_instead_of_silently_replacing_it(local_run):
    run(local_run)
    local_run["queue"]["tasks"].append(task("unexpected-new-task"))
    with pytest.raises(ValueError, match="frozen recipe, queue or protocol"):
        run(local_run)
    assert local_run["sample"].call_count == 1


def test_all_unknown_stops_bulk_and_human_arriving_during_inference_is_not_replaced(local_run):
    local_run["request"].return_value = proposal()
    result = run(local_run)
    assert result["bulk_stopped_for_unknowns"] is True
    assert result["consensus_present_observations"] == 0


def test_human_arriving_during_inference_is_never_replaced(local_run):
    local_run["history"].side_effect = [[], [{"task_id": "one", "provenance": {"kind": "human_review"}}]]
    result = run(local_run)
    assert result["machine_rechecked_proposals"] == 0
    local_run["append"].assert_not_called()


def test_request_failure_preserves_private_failure_and_cleans_server(local_run):
    local_run["request"].side_effect = RuntimeError("private response detail")
    result = run(local_run)
    assert result["tasks_failed"] == 1 and local_run["events"][-1] == "stop"
    local_run["append"].assert_not_called()
    content = (local_run["directory"] / "native-review-v1/audits.jsonl").read_text()
    assert "private response detail" not in content
    assert json.loads(content)["status"] == "failed"


def test_actual_synthetic_video_sampling_roundtrips_even_contiguous_frames(tmp_path, monkeypatch):
    source = tmp_path / "source.mov"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    if not writer.isOpened():
        pytest.skip("No local synthetic video codec")
    try:
        for index in range(40):
            writer.write(np.full((64, 96, 3), (20 + index, 70, 100), np.uint8))
    finally:
        writer.release()
    monkeypatch.setattr(R, "MEDIA_ROOT", tmp_path)
    value = task()
    value.update(path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    value["interval"] = {"start_sec": .5, "end_sec": 3.}
    payload, records = R.sample_task_video(value, tmp_path, time.monotonic() + 10)
    assert payload.startswith(b"RIFF") and len(records) == 4
    assert [record["requested_sec"] for record in records] == [.5, 1., 1.5, 2.]
    assert [record["actual_sec"] for record in records] == pytest.approx([.5, 1., 1.5, 2.], abs=.1)
    assert [record["source_frame_index"] for record in records] == [5, 10, 15, 20]
    for record in records:
        assert record["source_shape"] == record["view_shape"] == [64, 96, 3]
        assert len(record["source_frame_sha256"]) == len(record["jpeg_sha256"]) == len(record["avi_decoded_frame_sha256"]) == 64


def test_sampler_rejects_reserved_partition_and_bad_source_hash_before_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_open_video", lambda *args: pytest.fail("Invalid source cannot decode"))
    value = task()
    value["partition"] = "holdout"
    with pytest.raises(ValueError, match="development task"):
        R.sample_task_video(value, tmp_path, time.monotonic() + 5)


@pytest.fixture
def short_source(tmp_path, monkeypatch):
    source = tmp_path / "roundoff.mov"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    if not writer.isOpened():
        pytest.skip("No local synthetic video codec")
    try:
        for index in range(40):
            writer.write(np.full((64, 96, 3), (20 + index, 70, 100), np.uint8))
    finally:
        writer.release()
    monkeypatch.setattr(R, "MEDIA_ROOT", tmp_path)
    value = task()
    value.update(path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return value


def test_small_protocol_end_roundoff_is_clamped_and_preserved_in_audit(short_source, tmp_path):
    short_source["interval"] = {"start_sec": 0., "end_sec": 4.001}
    payload, records = R.sample_task_video(short_source, tmp_path, time.monotonic() + 10)
    assert payload.startswith(b"RIFF") and len(records) == 8
    for record in records:
        assert record["declared_interval"] == {"start_sec": 0., "end_sec": 4.001}
        assert record["sampling_interval"] == {"start_sec": 0., "end_sec": 4.}
        assert record["source_duration_sec"] == 4. and record["interval_end_clamped"] is True
        assert record["requested_sec"] < 4. and record["actual_sec"] < 4.


def test_protocol_end_beyond_source_clock_tolerance_fails_before_frame_reads(short_source, tmp_path, monkeypatch):
    short_source["interval"] = {"start_sec": 0., "end_sec": 4.251}
    monkeypatch.setattr(R, "_read_frame_at", lambda *args: pytest.fail("Outside tolerance must fail before decoding frames"))
    with pytest.raises(ValueError, match="exceeds the source video"):
        R.sample_task_video(short_source, tmp_path, time.monotonic() + 5)
