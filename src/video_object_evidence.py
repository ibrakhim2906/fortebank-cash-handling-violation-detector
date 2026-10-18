"""Optional presentation of existing anonymous detections; never a detector.

This module reads no files, media, models or labels and changes no verdict or
candidate window. Outputs describe unverified object appearance only. Enable
it separately after the detector's quality and operational checks pass and an
exact source-index/hash evidence decoder is available; timestamps alone do not
verify the displayed source pixels.
"""
from collections import defaultdict
import math
from numbers import Integral, Real


CLASS_IDS = (0, 24, 26, 28, 63, 67)
WINDOW_SECONDS = 2.0
MIN_SAMPLES = 3
MIN_SPAN_SECONDS = 1.0
MIN_CONFIDENCE = .35
TRACK_IOU = .3
TRACK_MAX_GAP_SECONDS = 1.0
FLOAT_TOLERANCE = 1e-7
_CATEGORIES = {
    24: ("bag_like_object_visible", "сумку, рюкзак или чемодан"),
    26: ("bag_like_object_visible", "сумку, рюкзак или чемодан"),
    28: ("bag_like_object_visible", "сумку, рюкзак или чемодан"),
    63: ("laptop_like_object_visible", "ноутбук"),
    67: ("phone_like_object_visible", "телефон"),
}


def _finite_number(value, *, minimum=0., maximum=300.):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError("Object evidence requires valid finite numeric metadata")
    return float(value)


def _nonnegative_integer(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Object evidence requires nonnegative integer source and track indices")
    return int(value)


def _box_iou(first, second):
    width = max(0., min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0., min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_first + area_second - intersection)


def _validate_audit(object_audit):
    if not isinstance(object_audit, (list, tuple)) or not object_audit:
        raise ValueError("Object evidence requires a nonempty existing detection audit")
    output, track_history = [], {}
    previous_time, previous_index = -1., -1
    for row in object_audit:
        if not isinstance(row, dict) or not {"timestamp_sec", "source_frame_index", "detections"} <= set(row):
            raise ValueError("Object evidence audit is missing required source fields")
        timestamp = _finite_number(row["timestamp_sec"])
        frame_index = _nonnegative_integer(row["source_frame_index"])
        if timestamp <= previous_time or frame_index <= previous_index:
            raise ValueError("Object evidence source times and frame indices must strictly increase")
        if "requested_sec" in row:
            _finite_number(row["requested_sec"])
        detections = row["detections"]
        if not isinstance(detections, list) or len(detections) > 100:
            raise ValueError("Object evidence requires a bounded detection list")
        checked, seen_tracks = [], set()
        for item in detections:
            if not isinstance(item, dict) or set(item) != {"class_id", "score", "box_xyxy", "track_id"}:
                raise ValueError("Object evidence detection fields differ from the expected audit schema")
            class_id = _nonnegative_integer(item["class_id"])
            track_id = _nonnegative_integer(item["track_id"])
            if class_id not in CLASS_IDS or track_id in seen_tracks:
                raise ValueError("Object evidence has an unsupported class or duplicated track in one source sample")
            previous = track_history.get(track_id)
            if previous is not None and previous["class_id"] != class_id:
                raise ValueError("An anonymous object track cannot change its class")
            score = _finite_number(item["score"], maximum=1.)
            if score < MIN_CONFIDENCE - FLOAT_TOLERANCE:
                raise ValueError("Object evidence score is below the frozen detector confidence threshold")
            box = item["box_xyxy"]
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError("Object evidence requires four normalized box coordinates")
            box = [_finite_number(value, maximum=1.) for value in box]
            if not (box[0] < box[2] and box[1] < box[3]):
                raise ValueError("Object evidence requires a nonempty normalized box")
            if previous is not None:
                if timestamp - previous["timestamp_sec"] > TRACK_MAX_GAP_SECONDS + 1e-8:
                    raise ValueError("An expired anonymous object track cannot be reused")
                if _box_iou(previous["box_xyxy"], box) < TRACK_IOU:
                    raise ValueError("Anonymous object track geometry violates the frozen IoU policy")
            track_history[track_id] = {"class_id": class_id, "timestamp_sec": timestamp, "box_xyxy": box}
            seen_tracks.add(track_id)
            checked.append({"class_id": class_id, "track_id": track_id, "score": score, "box_xyxy": box})
        output.append({"timestamp_sec": timestamp, "source_frame_index": frame_index, "detections": checked})
        previous_time, previous_index = timestamp, frame_index
    return output


def build_object_evidence(object_audit, candidate_timestamp_sec, window_scores=None):
    """Explain one already selected positive window without selecting a verdict.

    Optional ``window_scores`` is a numeric sequence aligned to increasing
    two-second bins present in the audit. It is validated and recorded only;
    it never changes the supplied candidate, threshold or ranking of tracks.
    All returned stable-object times are original sampled source timestamps.
    """
    rows = _validate_audit(object_audit)
    candidate = _finite_number(candidate_timestamp_sec)
    bins = sorted({int(row["timestamp_sec"] // WINDOW_SECONDS) for row in rows})
    selected_bin = int(candidate // WINDOW_SECONDS)
    if selected_bin not in bins:
        raise ValueError("Object evidence candidate must belong to an existing sampled window")
    candidate_score = None
    if window_scores is not None:
        if isinstance(window_scores, (str, bytes, dict, set)):
            raise ValueError("Object evidence scores must be a sequence aligned to existing windows")
        try:
            scores = list(window_scores)
        except TypeError:
            raise ValueError("Object evidence scores must be a sequence aligned to existing windows") from None
        if len(scores) != len(bins):
            raise ValueError("Object evidence scores do not match the existing sampled windows")
        scores = [_finite_number(score, maximum=1.) for score in scores]
        candidate_score = scores[bins.index(selected_bin)]
    start, end = selected_bin * WINDOW_SECONDS, (selected_bin + 1) * WINDOW_SECONDS
    tracks = defaultdict(list)
    for row in rows:
        if not start <= row["timestamp_sec"] < end:
            continue
        for item in row["detections"]:
            if item["class_id"] in _CATEGORIES:
                tracks[(item["class_id"], item["track_id"])].append((row["timestamp_sec"], item["score"]))
    stable = []
    for (class_id, track_id), observations in tracks.items():
        span = observations[-1][0] - observations[0][0]
        if len(observations) >= MIN_SAMPLES and span >= MIN_SPAN_SECONDS:
            confidence = sum(value[1] for value in observations) / len(observations)
            stable.append((class_id, track_id, observations, span, confidence))
    selection = {"window_start_sec": start, "window_end_sec": end, "candidate_score": candidate_score}
    if not stable:
        return {"type": "generic", "evidence_timestamps_sec": [candidate],
                "description": (
                    "Локальная модель выделила фрагмент для проверки человеком. "
                    "В выбранном окне нет достаточно устойчивого повторного выделения "
                    "одного объекта для отдельного описания. Тип нарушения и нормативный "
                    "источник автоматически не установлены."
                ), "selection": {**selection, "kind": "generic"}}
    class_id, track_id, observations, span, confidence = min(
        stable, key=lambda item: (-len(item[2]), -item[3], -item[4], item[0], item[1]))
    indices = (0, (len(observations) - 1) // 2, len(observations) - 1)
    timestamps = [observations[index][0] for index in indices]
    category, object_name = _CATEGORIES[class_id]
    return {"type": category, "evidence_timestamps_sec": timestamps,
            "description": (
                f"На выбранных кадрах локальный детектор повторно выделяет объект, похожий на {object_name}. "
                "Это неподтверждённая классификация модели. Видимый объект сам по себе "
                "не устанавливает нарушение. Правило и нормативный источник автоматически не определены."
            ),
            "selection": {**selection, "kind": "stable_object", "class_id": class_id,
                          "track_id": track_id, "samples": len(observations),
                          "span_sec": span, "mean_score": confidence}}
