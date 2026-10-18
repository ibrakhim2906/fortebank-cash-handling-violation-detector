"""Pure synthetic object evidence; no media/model/file/label access."""
from copy import deepcopy

import pytest

from src.video_object_evidence import build_object_evidence


def detection(class_id=67, track_id=1, score=.8):
    return {"class_id": class_id, "track_id": track_id, "score": score, "box_xyxy": [.2, .3, .4, .5]}


def audit(times=(0., .5, 1., 1.5), class_id=67, track_id=1):
    return [{"timestamp_sec": timestamp, "source_frame_index": int(timestamp * 30),
             "detections": [detection(class_id, track_id)]} for timestamp in times]


@pytest.mark.parametrize("class_id,expected", [
    (24, "bag_like_object_visible"), (26, "bag_like_object_visible"),
    (28, "bag_like_object_visible"), (63, "laptop_like_object_visible"),
    (67, "phone_like_object_visible"),
])
def test_stable_appearance_uses_original_first_middle_last_times(class_id, expected):
    rows = audit(class_id=class_id)
    result = build_object_evidence(rows, .75, [.8])
    assert result["type"] == expected
    assert result["evidence_timestamps_sec"] == [0., .5, 1.5]
    assert result["selection"] == {"window_start_sec": 0., "window_end_sec": 2., "candidate_score": .8,
                                   "kind": "stable_object", "class_id": class_id, "track_id": 1,
                                   "samples": 4, "span_sec": 1.5, "mean_score": .8}
    assert "похожий" in result["description"] and "неподтверждённая" in result["description"]
    assert "нормативный источник автоматически не определены" in result["description"]


@pytest.mark.parametrize("rows", [
    audit(times=(0., .5)), audit(times=(0., .4, .8)), audit(class_id=0),
    [{"timestamp_sec": float(index / 2), "source_frame_index": index * 15,
      "detections": [detection(track_id=index)]} for index in range(4)],
])
def test_short_unstable_person_or_separate_tracks_keep_generic_candidate(rows):
    result = build_object_evidence(rows, .375)
    assert result["type"] == "generic"
    assert result["evidence_timestamps_sec"] == [.375]
    assert result["selection"]["kind"] == "generic"


def test_exact_minimum_three_samples_and_one_second_span_is_eligible():
    result = build_object_evidence(audit(times=(.1, .6, 1.1)), .6)
    assert result["type"] == "phone_like_object_visible"
    assert result["evidence_timestamps_sec"] == [.1, .6, 1.1]


def test_window_is_preselected_and_cannot_be_changed_by_higher_score_or_object():
    rows = audit(times=(0., .5)) + audit(times=(2., 2.5, 3., 3.5), class_id=63, track_id=2)
    result = build_object_evidence(rows, .25, [.1, .99])
    assert result["type"] == "generic" and result["evidence_timestamps_sec"] == [.25]
    assert result["selection"]["candidate_score"] == .1


def test_end_boundary_belongs_to_next_window_and_cannot_complete_previous_track():
    rows = audit(times=(1., 1.5, 2.))
    result = build_object_evidence(rows, 1.25)
    assert result["type"] == "generic"
    result_next = build_object_evidence(rows, 2.)
    assert result_next["selection"]["window_start_sec"] == 2.


def test_ranking_prefers_sample_count_then_span_then_score_then_fixed_ids():
    rows = audit()
    for index, row in enumerate(rows):
        row["detections"].append(detection(63, 2, .9))
        if index < 3:
            row["detections"].append(detection(24, 3, .99))
    assert build_object_evidence(rows, .75)["type"] == "laptop_like_object_visible"
    for row in rows:
        row["detections"] = [detection(67, 1, .8), detection(63, 2, .8)]
    assert build_object_evidence(rows, .75)["selection"]["class_id"] == 63
    for row in rows:
        row["detections"] = [detection(67, 3, .8), detection(67, 2, .8)]
    assert build_object_evidence(rows, .75)["selection"]["track_id"] == 2


def test_input_order_is_irrelevant_for_detection_ranking_and_inputs_are_not_mutated():
    rows = audit()
    for row in rows:
        row["detections"].append(detection(63, 2, .9))
    original = deepcopy(rows)
    expected = build_object_evidence(rows, .75)
    assert rows == original
    for row in rows:
        row["detections"].reverse()
    assert build_object_evidence(rows, .75) == expected


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -.1, 0., .349, 1.1, True, "0.8"])
def test_bad_detector_scores_are_rejected(score):
    rows = audit()
    rows[0]["detections"][0]["score"] = score
    with pytest.raises(ValueError):
        build_object_evidence(rows, .75)


def test_frozen_confidence_boundary_allows_only_floating_point_tolerance():
    rows = audit()
    for row in rows:
        row["detections"][0]["score"] = .35 - 1e-8
    assert build_object_evidence(rows, .75)["type"] == "phone_like_object_visible"
    rows[1]["detections"][0]["score"] = .35 - 1e-5
    with pytest.raises(ValueError, match="confidence threshold"):
        build_object_evidence(rows, .75)


def test_same_track_cannot_jump_to_unmatched_geometry():
    rows = audit()
    rows[1]["detections"][0]["box_xyxy"] = [.6, .3, .8, .5]
    with pytest.raises(ValueError, match="IoU policy"):
        build_object_evidence(rows, .75)


def test_expired_track_id_cannot_be_reused_even_in_a_later_candidate_window():
    rows = audit(times=(0., .5, 2., 2.5, 3.))
    with pytest.raises(ValueError, match="expired"):
        build_object_evidence(rows, 2.5)


def test_exact_track_gap_and_iou_boundaries_remain_valid():
    rows = audit(times=(0., 1., 1.5))
    shift = .4 * (1 - .3) / (1 + .3)
    rows[0]["detections"][0]["box_xyxy"] = [0., .2, .4, .6]
    rows[1]["detections"][0]["box_xyxy"] = [shift, .2, .4 + shift, .6]
    rows[2]["detections"][0]["box_xyxy"] = [shift, .2, .4 + shift, .6]
    assert build_object_evidence(rows, .75)["type"] == "phone_like_object_visible"


@pytest.mark.parametrize("box", [[0, 0, 0, .5], [0, 0, 1.1, 1], [.5, 0, .4, 1],
                                 [0, 0, float("nan"), 1], [0, .5], "box"])
def test_bad_boxes_are_rejected(box):
    rows = audit()
    rows[0]["detections"][0]["box_xyxy"] = box
    with pytest.raises(ValueError):
        build_object_evidence(rows, .75)


def test_invalid_track_metadata_and_class_switches_are_rejected():
    for invalid_id in (-1, 1.5, True, "1"):
        rows = audit()
        rows[0]["detections"][0]["track_id"] = invalid_id
        with pytest.raises(ValueError):
            build_object_evidence(rows, .75)
    rows = audit()
    rows[1]["detections"][0]["class_id"] = 63
    with pytest.raises(ValueError, match="change its class"):
        build_object_evidence(rows, .75)
    rows = audit()
    rows[0]["detections"].append(detection())
    with pytest.raises(ValueError, match="duplicated track"):
        build_object_evidence(rows, .75)


def test_invalid_source_times_indices_and_schema_are_rejected():
    for field, value in (("timestamp_sec", float("nan")), ("timestamp_sec", -.1),
                         ("timestamp_sec", True), ("source_frame_index", 1.5), ("source_frame_index", -1)):
        rows = audit()
        rows[0][field] = value
        with pytest.raises(ValueError):
            build_object_evidence(rows, .75)
    for field in ("timestamp_sec", "source_frame_index"):
        rows = audit()
        rows[1][field] = rows[0][field]
        with pytest.raises(ValueError, match="strictly increase"):
            build_object_evidence(rows, .75)
    for rows in ([], None, [{"timestamp_sec": 0., "detections": []}]):
        with pytest.raises(ValueError):
            build_object_evidence(rows, .75)


@pytest.mark.parametrize("scores", [[.8, .7], [], [float("nan")], [1.1], [True], ".8", {"score": .8}])
def test_window_scores_are_validated_but_never_tuned(scores):
    with pytest.raises(ValueError):
        build_object_evidence(audit(), .75, scores)


def test_candidate_must_be_finite_and_belong_to_an_existing_window():
    for candidate in (float("nan"), float("inf"), -.1, True, 2.5):
        with pytest.raises(ValueError):
            build_object_evidence(audit(), candidate)
