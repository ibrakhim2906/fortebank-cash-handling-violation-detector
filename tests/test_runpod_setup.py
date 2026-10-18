import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts import runpod_setup as setup


def archive(path, *, name="bin/llama-server", link=None):
    with tarfile.open(path, "w:gz") as stream:
        info = tarfile.TarInfo(name)
        if link:
            info.type = tarfile.SYMTYPE; info.linkname = link
            stream.addfile(info)
        else:
            content = b"verified-runtime\n"; info.size = len(content); info.mode = 0o755
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("name,link", [("../escape", None), ("/absolute", None), ("lib/bad.so", "/tmp/outside")])
def test_runtime_rejects_archive_escape(tmp_path, name, link):
    packed = tmp_path / "runtime.tar.gz"; archive(packed, name=name, link=link)
    with pytest.raises((ValueError, tarfile.FilterError)):
        setup.unpack(packed, tmp_path / "runtime")
    assert not (tmp_path / "runtime").exists()


def test_runtime_reuse_detects_changed_executable(tmp_path):
    packed = tmp_path / "runtime.tar.gz"; archive(packed)
    dest = tmp_path / "runtime"
    first = setup.unpack(packed, dest)
    assert setup.unpack(packed, dest) == first
    (dest / "bin/llama-server").write_text("changed")
    with pytest.raises(ValueError, match="refusing overwrite"):
        setup.unpack(packed, dest)


def test_runtime_rejects_added_library(tmp_path):
    packed = tmp_path / "runtime.tar.gz"; archive(packed)
    dest = tmp_path / "runtime"; setup.unpack(packed, dest)
    (dest / "bin/untracked.so").write_text("injected")
    with pytest.raises(ValueError):
        setup.unpack(packed, dest)


def test_gpu_preflight_requires_one_large_gpu():
    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "NVIDIA A40, 46068, 570.00\n", "")
    assert setup.gpu_preflight("30b", runner)["name"] == "NVIDIA A40"
    def too_small(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "NVIDIA T4, 15360, 570.00\n", "")
    with pytest.raises(ValueError):
        setup.gpu_preflight("30b", too_small)


def test_cpu_host_fails_before_any_download(tmp_path, monkeypatch):
    def unavailable(*args):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(setup, "gpu_preflight", unavailable)
    monkeypatch.setattr(setup, "provision_asset", lambda *a: pytest.fail("Download must not start on a CPU host"))
    with pytest.raises(FileNotFoundError):
        setup.install(tmp_path, "30b")


def test_gpu_launch_keeps_model_private_and_enables_offload():
    command = setup.serve_command({"server": "server", "model_path": "weights", "projector_path": "projector"})
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--n-gpu-layers") + 1] == "999"
    assert "--no-mmproj-offload" not in command
    assert "--log-disable" in command


def test_bundle_excludes_private_material_and_unapproved_configs(tmp_path):
    names = ["pyproject.toml", "uv.lock", ".python-version", "src/task.py", "scripts/task.py", "app/video_ui.html",
             "deploy/runpod-assets.json", "docs/runpod_setup.md", "docs/action_rule_candidates.md",
             "data/processed/private-key", "data/raw/video.mov", "models/weights.gguf", ".env", "deploy/secret.json"]
    for name in names:
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("placeholder")
    output = tmp_path / "data/processed/code.tar.gz"
    result = setup.bundle(tmp_path, output)
    assert result["contains_credentials"] is False
    with tarfile.open(output) as stream:
        selected = set(stream.getnames())
    assert "src/task.py" in selected
    assert not any(n.startswith(("data/", "models/", "tools/")) or n == ".env" for n in selected)
    assert "deploy/secret.json" not in selected
    manifest = json.loads(output.with_suffix(".gz.manifest.json").read_text())
    assert set(manifest["files"]) == selected
    with pytest.raises(ValueError):
        setup.bundle(tmp_path, output)


def test_bundle_rejects_linked_source(tmp_path):
    for name in ["pyproject.toml", "uv.lock", ".python-version", "deploy/runpod-assets.json", "docs/runpod_setup.md", "docs/action_rule_candidates.md"]:
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("placeholder")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/task.py").symlink_to(tmp_path / "pyproject.toml")
    with pytest.raises(ValueError, match="linked"):
        setup.bundle(tmp_path, tmp_path / "data/processed/code.tar.gz")
