"""Dispatcher checks using trusted synthetic dictionaries, never footage."""
import hashlib
import json
from types import SimpleNamespace

import joblib
import pytest

from src import video_inference as inference


def _artifact(tmp_path, bundle):
    path = tmp_path / "run" / "models" / "baseline.joblib"
    path.parent.mkdir(parents=True)
    joblib.dump(bundle, path)
    return path


@pytest.mark.parametrize("engine,module", [
    (None, "src.video_analysis"),
    ("mobilenet", "src.video_pretrained"),
    ("mobilenet-temporal", "src.video_temporal"),
    ("qwen-actions", "src.video_qwen_actions"),
    ("qwen-spatial-v3", "src.video_qwen_spatial_v3"),
    ("object-tracks", "src.video_object_tracks"),
])
def test_routes_only_fixed_local_engines(tmp_path, monkeypatch, engine, module):
    bundle = {} if engine is None else {"engine": engine}
    model = _artifact(tmp_path, bundle)
    calls = []

    def analyze(path, model_path, deadline):
        calls.append((path, model_path, deadline))
        return {"verdict": "no_violation", "violations": []}

    imports = []

    def load(name):
        imports.append(name)
        return SimpleNamespace(analyze_video=analyze)

    monkeypatch.setattr(inference.importlib, "import_module", load)
    result = inference.analyze_video("synthetic.mov", model, deadline=123.0)
    assert result == {"verdict": "no_violation", "violations": []}
    assert imports == [module]
    assert calls == [("synthetic.mov", model, 123.0)]


def test_unknown_engine_rejected_before_import(tmp_path, monkeypatch):
    model = _artifact(tmp_path, {"engine": "arbitrary.module"})

    def unexpected_import(_):
        pytest.fail("An unknown artifact engine must never be imported")

    monkeypatch.setattr(inference.importlib, "import_module", unexpected_import)
    with pytest.raises(ValueError, match="Unsupported local video engine"):
        inference.analyze_video("synthetic.mov", model)


def test_manifest_corruption_rejected_before_unpickle(tmp_path, monkeypatch):
    model = _artifact(tmp_path, {"engine": "mobilenet"})
    manifest = model.parent.parent / "model_manifest.json"
    manifest.write_text(json.dumps({
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }))
    model.write_bytes(model.read_bytes() + b"changed artifact")

    def unexpected_load(_):
        pytest.fail("Corrupt manifested artifacts must be rejected before unpickling")

    monkeypatch.setattr(inference.joblib, "load", unexpected_load)
    with pytest.raises(ValueError, match="differs from its manifest"):
        inference.analyze_video("synthetic.mov", model)


@pytest.mark.parametrize("source", ["src/video_analysis.py", "src/video_pretrained.py"])
def test_changed_inherited_source_rejected_before_import(tmp_path, monkeypatch, source):
    model = _artifact(tmp_path, {
        "engine": "mobilenet-temporal",
        "inherited_sources": {source: "0" * 64},
    })

    def unexpected_import(_):
        pytest.fail("Stale inherited code must be rejected before selecting an engine")

    monkeypatch.setattr(inference.importlib, "import_module", unexpected_import)
    with pytest.raises(ValueError, match="Inherited feature code differs"):
        inference.analyze_video("synthetic.mov", model)


def test_unknown_inherited_source_rejected(tmp_path, monkeypatch):
    model = _artifact(tmp_path, {
        "inherited_sources": {"../unrelated.py": "0" * 64},
    })

    def unexpected_import(_):
        pytest.fail("Unsupported inherited paths must never select an engine")

    monkeypatch.setattr(inference.importlib, "import_module", unexpected_import)
    with pytest.raises(ValueError, match="Unsupported inherited feature source"):
        inference.analyze_video("synthetic.mov", model)
