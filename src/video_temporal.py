"""Offline max-window detection over the frozen local MobileNetV2 descriptors.

The supervised window head nominates a review candidate. The maximum score is
not a calibrated video probability and does not establish an action or rule.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import joblib
import numpy as np

from src import config as C
from src import video_pretrained as _pretrained
from src.video_analysis import VideoAnalysisError, _check_deadline, _positive_probability, evidence_frames

SCHEMA_VERSION = 1
ENGINE = "mobilenet-temporal"
WEIGHTS_SHA256 = _pretrained.WEIGHTS_SHA256
DEFAULT_WEIGHTS_PATH = _pretrained.DEFAULT_WEIGHTS_PATH
WINDOW_FEATURE_NAMES = _pretrained.WINDOW_FEATURE_NAMES
CLIP_FEATURE_NAMES = _pretrained.CLIP_FEATURE_NAMES
FEATURE_CONFIG = {
    **_pretrained.FEATURE_CONFIG,
    "engine": ENGINE,
    "detection_mode": "max-window",
    "pretrained_source_sha256": hashlib.sha256(Path(_pretrained.__file__).read_bytes()).hexdigest(),
}


def extract_video(path: str | Path, deadline: float | None = None) -> dict:
    """Reuse the frozen local extractor; no second feature family or decoding."""
    result = _pretrained.extract_video(path, deadline)
    result["sampling"] = {**result["sampling"], "feature_config": dict(FEATURE_CONFIG)}
    return result


def _load_model(model_path: str | Path | None) -> dict:
    path = Path(model_path or os.environ.get(
        "DATATHON_VIDEO_MODEL", str(C.DATA_PROC / "video-mobilenet-temporal/models/baseline.joblib")))
    if not path.is_file():
        raise VideoAnalysisError("Local temporal window model is missing; train it before analysis")
    # Read only trusted local training artifacts, never an uploaded pickle file.
    artifact = joblib.load(path)
    if (not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION
            or artifact.get("engine") != ENGINE):
        raise VideoAnalysisError("Unsupported local temporal model schema or engine")
    names = artifact.get("feature_names", {})
    if not isinstance(names, dict):
        raise VideoAnalysisError("Local temporal model feature names are invalid")
    for key, expected in (("clip", CLIP_FEATURE_NAMES), ("window", WINDOW_FEATURE_NAMES)):
        if tuple(names.get(key, ())) != expected:
            raise VideoAnalysisError(f"Local temporal model {key} feature names do not match")
    if artifact.get("feature_config") != FEATURE_CONFIG:
        raise VideoAnalysisError("Local temporal model feature configuration does not match")
    if artifact.get("weights_sha256") != WEIGHTS_SHA256:
        raise VideoAnalysisError("Local temporal model encoder weights do not match")
    if artifact.get("source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise VideoAnalysisError("Local temporal model was built with different analysis source code")
    if not hasattr(artifact.get("window_model"), "predict_proba"):
        raise VideoAnalysisError("Local temporal window model does not support probabilities")
    threshold = artifact.get("threshold", 0.5)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold != 0.5:
        raise VideoAnalysisError("Local temporal model requires the fixed 0.5 threshold")
    return artifact


def analyze_video(path: str | Path, model_path: str | Path | None = None,
                  deadline: float | None = None) -> dict:
    """Use the maximum local window score for the verdict and its evidence."""
    _check_deadline(deadline)
    artifact = _load_model(model_path)
    extracted = extract_video(path, deadline)
    windows = extracted["windows"]
    if not windows:
        raise VideoAnalysisError("Local temporal extractor returned no windows")
    values = np.vstack([row["features"] for row in windows])
    scores = _positive_probability(artifact["window_model"], values)
    _check_deadline(deadline)
    if float(np.max(scores)) < artifact.get("threshold", 0.5):
        return {"verdict": "no_violation", "violations": []}
    timestamp = float(windows[int(np.argmax(scores))]["timestamp_sec"])
    frames = evidence_frames(path, [timestamp], deadline)
    return {
        "verdict": "violation",
        "violations": [{
            "type": "Визуальный признак для проверки",
            "timecode_sec": frames[0]["timestamp_sec"],
            "description": (
                "Локальная модель на визуальных признаках MobileNetV2 выделила "
                "короткий фрагмент, похожий на размеченные положительные примеры. "
                "Кадр выбран как кандидат для проверки человеком. Конкретное "
                "действие, тип нарушения и нормативный источник этой моделью "
                "не установлены. Это не юридическое или дисциплинарное заключение."
            ),
            "evidence_frames": frames,
        }],
    }
