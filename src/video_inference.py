"""Dispatch trusted local video artifacts to a fixed allowlist of engines."""
import hashlib
import importlib
import json
import os
from pathlib import Path

import joblib

from src import config


ENGINES = {"numeric": "src.video_analysis", "mobilenet": "src.video_pretrained",
           "mobilenet-temporal": "src.video_temporal", "qwen-actions": "src.video_qwen_actions",
           "qwen-spatial-v3": "src.video_qwen_spatial_v3", "object-tracks": "src.video_object_tracks"}


def analyze_video(path, model_path=None, deadline=None):
    selected = os.environ.get("DATATHON_VIDEO_ENGINE", "artifact")
    if selected == "semantic-v1":
        from src.video_semantic import analyze_video as semantic_analyze
        return semantic_analyze(path, deadline=deadline)
    if selected != "artifact":
        raise ValueError("Unsupported configured video engine")
    model_path = Path(model_path or os.environ.get(
        "DATATHON_VIDEO_MODEL", config.DATA_PROC / "video-baseline-comparison/models/baseline.joblib"))
    if not model_path.is_file():
        raise ValueError("The configured local video model is unavailable")
    manifest_path = model_path.parent.parent / "model_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if manifest.get("model_sha256") != actual:
            raise ValueError("Local video artifact differs from its manifest")
    # Operator-controlled artifacts only. Uploaded files are never unpickled.
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("Unsupported local video artifact")
    engine = bundle.get("engine", "numeric")
    if engine not in ENGINES:
        raise ValueError("Unsupported local video engine")
    for relative, expected in bundle.get("inherited_sources", {}).items():
        if relative not in {"src/video_analysis.py", "src/video_pretrained.py",
                            "src/video_action_proposals.py", "src/video_actions.py",
                            "scripts/provision_action_model.py", "src/video_qwen_actions.py",
                            "src/video_local_observations.py", "src/video_objects.py",
                            "scripts/fetch_video_objects.py"}:
            raise ValueError("Unsupported inherited feature source")
        if hashlib.sha256((config.ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Inherited feature code differs from the trained artifact")
    module = importlib.import_module(ENGINES[engine])
    return module.analyze_video(path, model_path=model_path, deadline=deadline)
