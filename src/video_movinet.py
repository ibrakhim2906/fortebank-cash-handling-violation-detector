"""Local frozen MoViNet-A0-Stream classifier logits, not action/violation truth.

The official TFLite stream exposes logits and recurrent states. Every window
starts with zero states; state is never shared across windows or recordings.
Runtime/model installation and commercial-license approval happen separately.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from src import config
from src.video_analysis import VideoAnalysisError, _check_deadline, _open_video


ENGINE = "movinet-a0-stream"
DEFAULT_MODEL_PATH = config.MODELS / "movinet" / "movinet_a0_stream.tflite"
WEIGHTS_SHA256 = "2afdb24bc127104171df2d133245beb3b8828705cc8d93f1aeb1e24392838924"
MODEL_VARIANT = "a0-stream-kinetics-600-classification-tflite-float16"
MODEL_VERSION = 2
RUNTIME_DISTRIBUTION = "ai-edge-litert"
RUNTIME_VERSION = "2.3.0"
IMAGE_SIZE, LOGIT_DIM, CPU_THREADS = 172, 600, 2
SAMPLE_FPS, WINDOW_SECONDS, STRIDE_SECONDS = 5.0, 8.0, 2.0
MAX_DURATION_SECONDS, PROCESSING_TIMEOUT_SECONDS = 300.0, 240.0
MAX_SOURCE_FRAMES = 18_000
WINDOW_FEATURE_NAMES = tuple(f"kinetics600_logit_{index}" for index in range(LOGIT_DIM))
INHERITED_SOURCES = {"src/video_analysis.py": hashlib.sha256(
    (config.ROOT / "src/video_analysis.py").read_bytes()).hexdigest(),
    "scripts/provision_movinet.py": hashlib.sha256((config.ROOT / "scripts/provision_movinet.py").read_bytes()).hexdigest()}
FEATURE_CONFIG = {
    "engine": ENGINE, "version": 1, "weights_sha256": WEIGHTS_SHA256,
    "model_variant": MODEL_VARIANT, "model_version": MODEL_VERSION,
    "weight_storage": "float16; signature inputs/states/logits checked independently",
    "representation": "final streaming classifier logits; not penultimate embeddings",
    "logit_dimension": LOGIT_DIM, "input_shape": [1, 1, IMAGE_SIZE, IMAGE_SIZE, 3],
    "preprocessing": "BGR to RGB; aspect-preserving centered black letterbox; float32 /255",
    "resize": "rounded dimensions; INTER_AREA shrink, INTER_LINEAR enlarge",
    "sample_fps": SAMPLE_FPS, "window_seconds": WINDOW_SECONDS, "stride_seconds": STRIDE_SECONDS,
    "tail_policy": "stop at first window reaching clip end; shorten that window without padding",
    "sampling": "first sequential frame at/after each requested time; original decoder timestamps",
    "timestamp_fallback": "source frame index / fps when timestamps invalid or nonincreasing",
    "state_policy": "fresh zero state per window; final output after all actual window frames",
    "clip_pooling": None, "runtime": RUNTIME_DISTRIBUTION, "runtime_version": RUNTIME_VERSION,
    "cpu_threads": CPU_THREADS, "seed": config.SEED, "metadata_features": False,
    "inference_network_required": False, "inherited_sources": INHERITED_SOURCES,
}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _asset(model_path=None, expected_sha256=None):
    if model_path is None and expected_sha256 is None:
        from scripts.provision_movinet import verify_ready
        ready = verify_ready()
        model_path, expected_sha256 = ready["model_path"], ready["weights_sha256"]
    path = Path(DEFAULT_MODEL_PATH if model_path is None else model_path)
    expected = WEIGHTS_SHA256 if expected_sha256 is None else expected_sha256
    if (not isinstance(expected, str) or len(expected) != 64
            or any(value not in "0123456789abcdef" for value in expected)):
        raise VideoAnalysisError("Approved MoViNet weights SHA-256 is not pinned")
    if expected != WEIGHTS_SHA256:
        raise VideoAnalysisError("MoViNet checksum is outside the approved model variant")
    if not path.is_file() or path.is_symlink() or _digest(path) != expected:
        raise VideoAnalysisError("Local MoViNet weights differ from the approved SHA-256")
    return path, expected


def get_encoder_provenance(model_path=None, expected_sha256=None):
    _, expected = _asset(model_path, expected_sha256)
    try:
        version = importlib.metadata.version(RUNTIME_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        raise VideoAnalysisError("Approved LiteRT runtime is not installed") from None
    if version != RUNTIME_VERSION:
        raise VideoAnalysisError("Installed LiteRT runtime differs from the pinned version")
    for relative, expected_source in INHERITED_SOURCES.items():
        if _digest(config.ROOT / relative) != expected_source:
            raise VideoAnalysisError("MoViNet inherited decoder source changed")
    return {"engine": ENGINE, "engine_module": "src.video_movinet", "source_sha256": _digest(__file__),
            "weights_sha256": expected, "runtime_distribution": RUNTIME_DISTRIBUTION,
            "model_variant": MODEL_VARIANT, "model_version": MODEL_VERSION,
            "runtime_version": version, "inherited_sources": dict(INHERITED_SOURCES),
            "feature_config_sha256": hashlib.sha256(json.dumps(FEATURE_CONFIG, sort_keys=True).encode()).hexdigest()}


def letterbox_bgr(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
            or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
        raise VideoAnalysisError("MoViNet requires a nonempty uint8 BGR image")
    height, width = frame.shape[:2]
    ratio = IMAGE_SIZE / max(height, width)
    size = (max(1, min(IMAGE_SIZE, round(width * ratio))), max(1, min(IMAGE_SIZE, round(height * ratio))))
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    left, top = (IMAGE_SIZE - size[0]) // 2, (IMAGE_SIZE - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = resized
    return canvas


def preprocess_frame(frame):
    image = cv2.cvtColor(letterbox_bgr(frame), cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    return np.ascontiguousarray(image[None, None])


class MoViNetEncoder:
    def __init__(self, model_path=None, expected_sha256=None):
        path, expected = _asset(model_path, expected_sha256)
        self.provenance = get_encoder_provenance(path, expected)
        # Hash before loading; pass verified bytes to prevent a file-swap race.
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise VideoAnalysisError("MoViNet weights changed before loading")
        try:
            from ai_edge_litert import interpreter as runtime
        except ImportError:
            raise VideoAnalysisError("Approved LiteRT runtime is not installed") from None
        self.interpreter = runtime.Interpreter(model_content=payload, num_threads=CPU_THREADS)
        self.interpreter.allocate_tensors()
        signatures = self.interpreter.get_signature_list()
        key = "serving_default" if "serving_default" in signatures else next(iter(signatures), None)
        if len(signatures) != 1 or key is None:
            raise VideoAnalysisError("MoViNet requires one unambiguous streaming signature")
        self.runner = self.interpreter.get_signature_runner(key)
        details, outputs = self.runner.get_input_details(), self.runner.get_output_details()
        if "image" not in details or "logits" not in outputs:
            raise VideoAnalysisError("MoViNet signature lacks image input or logits output")
        image = details["image"]
        if tuple(image["shape"]) != (1, 1, IMAGE_SIZE, IMAGE_SIZE, 3) or image["dtype"] != np.float32:
            raise VideoAnalysisError("MoViNet image signature differs from the fixed recipe")
        self.initial_states = {}
        for name, detail in details.items():
            if name == "image":
                continue
            shape, dtype = tuple(int(size) for size in detail["shape"]), np.dtype(detail["dtype"])
            if (not shape or any(size <= 0 for size in shape) or dtype.kind not in "fiub"
                    or math.prod(shape) * dtype.itemsize > 32 * 1024 * 1024):
                raise VideoAnalysisError("MoViNet stream state has unsupported shape or dtype")
            self.initial_states[name] = np.zeros(shape, dtype)
        if (not self.initial_states or set(outputs) != set(self.initial_states) | {"logits"}
                or tuple(outputs["logits"]["shape"]) != (1, LOGIT_DIM)
                or outputs["logits"]["dtype"] != np.float32
                or sum(value.nbytes for value in self.initial_states.values()) > 64 * 1024 * 1024):
            raise VideoAnalysisError("MoViNet streaming outputs differ from the pinned contract")
        self.provenance.update(interpreter_source_sha256=_digest(runtime.__file__),
                               opencv_version=cv2.__version__, numpy_version=np.__version__)
        self.signature = {"inputs": {name: {"shape": [int(size) for size in detail["shape"]], "dtype": str(np.dtype(detail["dtype"]))}
                                     for name, detail in details.items()},
                          "outputs": {name: {"shape": [int(size) for size in detail["shape"]], "dtype": str(np.dtype(detail["dtype"]))}
                                      for name, detail in outputs.items()}}
        self._lock = threading.Lock()

    def encode_window(self, frames_bgr, deadline=None):
        if not 1 <= len(frames_bgr) <= int(SAMPLE_FPS * WINDOW_SECONDS):
            raise VideoAnalysisError("MoViNet window requires one to forty actual frames")
        with self._lock:
            states = {name: value.copy() for name, value in self.initial_states.items()}
            logits = None
            for frame in frames_bgr:
                _check_deadline(deadline)
                outputs = self.runner(image=preprocess_frame(frame), **states)
                _check_deadline(deadline)
                if not isinstance(outputs, dict) or set(outputs) != set(states) | {"logits"}:
                    raise VideoAnalysisError("MoViNet returned an invalid streaming result")
                logits = np.asarray(outputs["logits"])
                if logits.shape != (1, LOGIT_DIM) or logits.dtype != np.float32 or not np.isfinite(logits).all():
                    raise VideoAnalysisError("MoViNet returned invalid classifier logits")
                for name, previous in states.items():
                    value = np.asarray(outputs[name])
                    if value.shape != previous.shape or value.dtype != previous.dtype or not np.isfinite(value).all():
                        raise VideoAnalysisError("MoViNet returned invalid stream state")
                    states[name] = value.copy()
            return logits[0].copy()


def window_intervals(duration_sec):
    if not math.isfinite(duration_sec) or not 0 < duration_sec <= MAX_DURATION_SECONDS:
        raise VideoAnalysisError("MoViNet duration is outside the supported bound")
    windows, start = [], 0.0
    while True:
        windows.append({"start_sec": start, "end_sec": min(duration_sec, start + WINDOW_SECONDS)})
        if start + WINDOW_SECONDS >= duration_sec - 1e-9:
            return windows
        start += STRIDE_SECONDS


def _sample(path, start_sec, end_sec, deadline):
    cap, fps, count = _open_video(path, deadline)
    try:
        duration = count / fps
        if (count <= 0 or fps < SAMPLE_FPS or duration > MAX_DURATION_SECONDS
                or not 0 <= start_sec < end_sec <= duration + 1e-6):
            raise VideoAnalysisError("MoViNet interval or source metadata is unsupported")
        requested = [start_sec + index / SAMPLE_FPS for index in range(math.ceil((end_sec - start_sec) * SAMPLE_FPS))
                     if start_sec + index / SAMPLE_FPS < end_sec - 1e-8]
        frames, audit, previous, cursor = [], [], -1.0, 0
        for index in range(min(count + 1, MAX_SOURCE_FRAMES + 1)):
            _check_deadline(deadline)
            if cursor == len(requested):
                break
            if index >= MAX_SOURCE_FRAMES or not cap.grab():
                raise VideoAnalysisError("MoViNet source ended before the requested sampling completed")
            actual = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            timestamp_source = "decoder"
            if not math.isfinite(actual) or actual < 0 or (index > 0 and actual <= previous):
                actual, timestamp_source = index / fps, "source_frame_index_over_fps"
            previous = actual
            if actual + 1e-7 < requested[cursor]:
                continue
            if actual >= end_sec or actual - requested[cursor] > max(.05, 2 / fps):
                raise VideoAnalysisError("MoViNet sampled frame is outside its requested interval/tolerance")
            ok, frame = cap.retrieve()
            if not ok or frame is None or frame.size == 0:
                raise VideoAnalysisError("MoViNet could not retrieve a requested source frame")
            frames.append(letterbox_bgr(frame))
            audit.append({"requested_sec": requested[cursor], "actual_sec": actual,
                          "source_frame_index": index, "timestamp_source": timestamp_source,
                          "source_frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                          "source_shape": list(frame.shape)})
            cursor += 1
        if cursor != len(requested) or not frames:
            raise VideoAnalysisError("MoViNet sampling did not complete")
        return frames, audit, duration
    finally:
        cap.release()


def _sampling(audit):
    return {"requested_times_sec": [row["requested_sec"] for row in audit],
            "actual_times_sec": [row["actual_sec"] for row in audit],
            "source_frame_indices": [row["source_frame_index"] for row in audit],
            "source_frame_sha256": [row["source_frame_sha256"] for row in audit],
            "source_shapes": [row["source_shape"] for row in audit],
            "timestamp_sources": [row["timestamp_source"] for row in audit],
            "feature_config": dict(FEATURE_CONFIG)}


def extract_interval(path, start_sec, end_sec, deadline=None, encoder=None):
    started = time.monotonic()
    deadline = min(deadline, started + PROCESSING_TIMEOUT_SECONDS) if deadline is not None else started + PROCESSING_TIMEOUT_SECONDS
    if (not all(type(value) in (int, float) and math.isfinite(value) for value in (start_sec, end_sec))
            or end_sec - start_sec > WINDOW_SECONDS + 1e-6):
        raise VideoAnalysisError("MoViNet interval must be finite and at most eight seconds")
    _check_deadline(deadline)
    cv2.setNumThreads(CPU_THREADS)
    cv2.setRNGSeed(config.SEED)
    encoder = encoder or MoViNetEncoder()
    frames, audit, duration = _sample(path, start_sec, end_sec, deadline)
    features = encoder.encode_window(frames, deadline)
    return {"features": features, "start_sec": start_sec, "end_sec": end_sec, "duration_sec": duration,
            "sampling": _sampling(audit), "extraction_sec": time.monotonic() - started,
            "encoder_provenance": encoder.provenance}


def extract_video(path, deadline=None, encoder=None):
    """Extract dense windows only; no additional clip-level feature family."""
    started = time.monotonic()
    deadline = min(deadline, started + PROCESSING_TIMEOUT_SECONDS) if deadline is not None else started + PROCESSING_TIMEOUT_SECONDS
    _check_deadline(deadline)
    cv2.setNumThreads(CPU_THREADS)
    cv2.setRNGSeed(config.SEED)
    encoder = encoder or MoViNetEncoder()
    cap, fps, count = _open_video(path, deadline)
    cap.release()
    duration = count / fps
    intervals = window_intervals(duration)
    frames, audit, duration = _sample(path, 0.0, duration, deadline)
    windows = []
    for interval in intervals:
        indices = [index for index, row in enumerate(audit)
                   if interval["start_sec"] <= row["requested_sec"] < interval["end_sec"] - 1e-8]
        values = encoder.encode_window([frames[index] for index in indices], deadline)
        sample = [audit[index] for index in indices]
        windows.append({**interval, "timestamp_sec": (sample[0]["actual_sec"] + sample[-1]["actual_sec"]) / 2,
                        "features": values, "sampling": _sampling(sample)})
    return {"windows": windows, "duration_sec": duration, "sampling": _sampling(audit),
            "extraction_sec": time.monotonic() - started, "encoder_provenance": encoder.provenance}
