"""Verified offline YOLOX-S object observations, without action/identity claims.

Only six public COCO object classes are retained. An object detection is not
evidence of ownership, cash handling, an action, or a regulatory violation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading

import cv2
import numpy as np

from scripts import fetch_video_objects as _assets
from src import config as C
from src.video_analysis import VideoAnalysisError, _check_deadline

MODEL_PATH = _assets.MODEL_PATH
WEIGHTS_SHA256 = _assets.WEIGHTS_SHA256
CLASS_IDS = (0, 24, 26, 28, 63, 67)
CLASS_NAMES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase",
               63: "laptop", 67: "cell phone"}
INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 100
FEATURE_CONFIG = {
    "model_id": "OpenCVZoo/YOLOX-S-2022nov",
    "source_commit": "47534e27c9851bb1128ccc0102f1145e27f23f98",
    "weights_sha256": WEIGHTS_SHA256,
    "license": "Apache-2.0",
    "input_size": [INPUT_SIZE, INPUT_SIZE],
    "color": "RGB",
    "normalization": "none; float32 pixel values 0..255",
    "letterbox": "top-left; floor resized dimensions; INTER_LINEAR; padding 114",
    "strides": [8, 16, 32],
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "nms_iou_threshold": NMS_IOU_THRESHOLD,
    "nms": "per-class xywh",
    "class_ids": list(CLASS_IDS),
    "max_detections": MAX_DETECTIONS,
    "opencv_threads": 2,
    "seed": C.SEED,
}


def _anchors() -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for stride in FEATURE_CONFIG["strides"]:
        width = height = INPUT_SIZE // stride
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        grids.append(np.stack((x, y), axis=2).reshape(-1, 2))
        strides.append(np.full((height * width, 1), stride))
    return np.vstack(grids).astype(np.float32), np.vstack(strides).astype(np.float32)


_GRIDS, _STRIDES = _anchors()


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Match the pinned Zoo demo: RGB, top-left letterbox and unscaled float32."""
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size):
        raise VideoAnalysisError("Object recognition requires a nonempty uint8 three-channel BGR frame")
    height, width = frame.shape[:2]
    ratio = min(INPUT_SIZE / height, INPUT_SIZE / width)
    resized_width, resized_height = int(width * ratio), int(height * ratio)
    if min(resized_width, resized_height) < 1:
        raise VideoAnalysisError("Object recognition does not support this frame aspect ratio")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.float32)
    canvas[:resized_height, :resized_width] = resized.astype(np.float32)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]), ratio


def postprocess(raw: np.ndarray, source_shape: tuple[int, ...], ratio: float) -> list[dict]:
    """Decode the exact 8400-anchor output and suppress boxes within each class."""
    try:
        values = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise VideoAnalysisError("YOLOX-S returned an invalid numeric output") from exc
    if values.shape != (1, 8400, 85) or not np.isfinite(values).all():
        raise VideoAnalysisError("YOLOX-S returned invalid output dimensions or nonfinite values")
    values = values[0]
    if ((values[:, 4:] < 0) | (values[:, 4:] > 1)).any():
        raise VideoAnalysisError("YOLOX-S returned invalid object/class probabilities")
    if (len(source_shape) < 2 or min(source_shape[:2]) <= 0
            or not np.isfinite(ratio) or ratio <= 0):
        raise VideoAnalysisError("YOLOX-S source geometry is invalid")
    height, width = source_shape[:2]
    try:
        with np.errstate(over="raise", invalid="raise"):
            centers = (values[:, :2].astype(np.float64) + _GRIDS) * _STRIDES
            sizes = np.exp(values[:, 2:4].astype(np.float64)) * _STRIDES
            boxes = np.concatenate((centers - sizes / 2, sizes), axis=1)
    except FloatingPointError as exc:
        raise VideoAnalysisError("YOLOX-S returned invalid box geometry") from exc
    if (not np.isfinite(boxes).all() or (sizes <= 0).any()
            or (np.abs(boxes) > np.finfo(np.float32).max).any()):
        raise VideoAnalysisError("YOLOX-S returned invalid box geometry")
    class_ids = values[:, 5:].argmax(axis=1)
    scores = values[:, 4] * values[np.arange(len(values)), 5 + class_ids]
    selected = np.flatnonzero((scores >= CONFIDENCE_THRESHOLD) & np.isin(class_ids, CLASS_IDS))
    candidates = []
    for class_id in CLASS_IDS:
        indices = selected[class_ids[selected] == class_id]
        if not len(indices):
            continue
        keep = cv2.dnn.NMSBoxes(boxes[indices].tolist(), scores[indices].tolist(),
                                CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD)
        for local_index in np.asarray(keep, dtype=int).reshape(-1):
            index = int(indices[local_index])
            x, y, box_width, box_height = boxes[index] / ratio
            xyxy = np.clip(np.asarray([x / width, y / height,
                                      (x + box_width) / width, (y + box_height) / height]), 0, 1)
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue  # An otherwise valid object wholly in padding has no source area.
            candidates.append({"class_id": int(class_id), "score": float(scores[index]),
                               "box_xyxy": xyxy.tolist()})
    candidates.sort(key=lambda item: (-item["score"], item["class_id"], *item["box_xyxy"]))
    return candidates[:MAX_DETECTIONS]


class ObjectDetector:
    """Load only pinned local bytes after checking their license and provenance."""

    def __init__(self, model_path: str | Path | None = None):
        try:
            ready = _assets.verify_ready()
        except (OSError, ValueError) as exc:
            raise VideoAnalysisError("Pinned local YOLOX-S assets or provenance are missing/invalid") from exc
        if (ready.get("weights_sha256") != WEIGHTS_SHA256
                or ready.get("source_commit") != FEATURE_CONFIG["source_commit"]
                or ready.get("license") != "Apache-2.0"):
            raise VideoAnalysisError("Local YOLOX-S readiness provenance differs from the fixed configuration")
        path = Path(model_path or ready["model_path"])
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise VideoAnalysisError("Local YOLOX-S model could not be read") from exc
        if hashlib.sha256(payload).hexdigest() != WEIGHTS_SHA256:
            raise VideoAnalysisError("Local YOLOX-S model SHA-256 differs from the approved weights")
        cv2.setNumThreads(2)
        cv2.setRNGSeed(C.SEED)
        try:
            # Loading the verified snapshot avoids a replace-between-hash-and-load race.
            self._net = cv2.dnn.readNetFromONNX(np.frombuffer(payload, dtype=np.uint8))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            outputs = self._net.getUnconnectedOutLayersNames()
            if len(outputs) != 1:
                raise VideoAnalysisError("YOLOX-S must expose exactly one raw detector output")
            self._output_name = outputs[0]
        except cv2.error as exc:
            raise VideoAnalysisError("OpenCV could not load the verified local YOLOX-S model") from exc
        self._lock = threading.Lock()
        self.provenance = dict(ready)

    def detect(self, frame: np.ndarray, deadline: float | None = None) -> list[dict]:
        """Observe supported objects locally; propagate model failures explicitly."""
        _check_deadline(deadline)
        blob, ratio = preprocess(frame)
        with self._lock:
            _check_deadline(deadline)
            try:
                self._net.setInput(blob)
                raw = self._net.forward(self._output_name)
            except cv2.error as exc:
                raise VideoAnalysisError("Local YOLOX-S inference failed") from exc
        _check_deadline(deadline)
        result = postprocess(raw, frame.shape, ratio)
        _check_deadline(deadline)
        return result
