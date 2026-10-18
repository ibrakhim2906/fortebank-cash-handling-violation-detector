import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts import provision_action_model as provision
from scripts.serve_action_model import command


def asset(content=b"synthetic weights"):
    return provision.Asset("models/local-action-model/test.gguf", "https://invalid.example/test", len(content),
                           hashlib.sha256(content).hexdigest())


def opener_for(content):
    return lambda *args, **kwargs: io.BytesIO(content)


def archive(tmp_path, members):
    path = tmp_path / "runtime.tar.gz"
    with tarfile.open(path, "w:gz") as output:
        for name, content, link in members:
            member = tarfile.TarInfo(name)
            member.mode = 0o755
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = link
                output.addfile(member)
            else:
                member.size = len(content)
                output.addfile(member, io.BytesIO(content))
    return path


def test_asset_integrity_and_no_overwrite(tmp_path):
    value = asset()
    provision.provision_asset(tmp_path, value, opener_for(b"synthetic weights"))
    assert provision.verify_asset(tmp_path, value)
    provision.provision_asset(tmp_path, value, lambda *a, **k: pytest.fail("Unexpected network"))
    destination = provision.safe_path(tmp_path, value.path)
    destination.write_bytes(b"existing different data")
    with pytest.raises(ValueError, match="refusing overwrite"):
        provision.provision_asset(tmp_path, value, opener_for(b"synthetic weights"))
    assert destination.read_bytes() == b"existing different data"


@pytest.mark.parametrize("payload", [b"synthetic weightS", b"too long" * 20, b"short"])
def test_corrupt_download_never_installed(tmp_path, payload):
    with pytest.raises(ValueError):
        provision.provision_asset(tmp_path, asset(), opener_for(payload))
    assert not provision.safe_path(tmp_path, asset().path).exists()
    assert not list((tmp_path / "models/local-action-model").glob("tmp*"))


@pytest.mark.parametrize("path", ["data/raw/file", "../models/local-action-model/a", "/tmp/a",
                                  "models/local-action-model/../../data/raw/a"])
def test_paths_cannot_escape_approved_directories(tmp_path, path):
    with pytest.raises(ValueError):
        provision.safe_path(tmp_path, path)


def test_destination_symlink_rejected(tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    (tmp_path / "models").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        provision.safe_path(tmp_path, "models/local-action-model/a")


@pytest.mark.parametrize("name,link", [("../escaped", ""), ("/absolute", ""),
                                     ("lib.so", "../../escaped"), ("lib.so", "/absolute")])
def test_archive_traversal_or_external_links_rejected(tmp_path, name, link):
    path = archive(tmp_path, [(name, b"bad", link)])
    with pytest.raises(ValueError, match="Unsafe"):
        provision.extract_runtime(tmp_path, path)


def test_internal_archive_link_is_materialized_and_existing_file_preserved(tmp_path):
    path = archive(tmp_path, [("bin/llama-server", b"executable", ""),
                              ("bin/libmodel.so.0", b"library", ""),
                              ("bin/libmodel.so", b"", "libmodel.so.0")])
    files = provision.extract_runtime(tmp_path, path)
    link = tmp_path / "tools/local-action-model/runtime/bin/libmodel.so"
    assert not link.is_symlink()
    assert link.read_bytes() == b"library"
    assert len(files) == 3
    link.write_bytes(b"user change")
    with pytest.raises(ValueError, match="refusing overwrite"):
        provision.extract_runtime(tmp_path, path)
    assert link.read_bytes() == b"user change"


def test_dot_prefixed_archive_links_are_internal_regular_files(tmp_path):
    path = archive(tmp_path, [("./bin/lib.so.0", b"library", ""),
                              ("./bin/lib.so", b"", "lib.so.0")])
    result = provision.extract_runtime(tmp_path, path)
    assert len(result) == 2
    assert (tmp_path / "tools/local-action-model/runtime/bin/lib.so").read_bytes() == b"library"


def test_duplicate_archive_paths_rejected_before_writing(tmp_path):
    path = archive(tmp_path, [("bin/tool", b"one", ""), ("./bin/tool", b"two", "")])
    with pytest.raises(ValueError, match="Duplicate"):
        provision.extract_runtime(tmp_path, path)
    assert not (tmp_path / "tools").exists()


def test_launcher_is_cpu_loopback_only_with_local_files(tmp_path):
    manifest = {"runtime_files": {"tools/local-action-model/runtime/bin/llama-server": "hash"}}
    args = command(tmp_path, manifest)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "8003"
    assert args[args.index("--n-gpu-layers") + 1] == "0"
    assert args[args.index("--ctx-size") + 1] == "4096"
    assert args[args.index("--parallel") + 1] == "1"
    assert "--log-disable" in args
    assert not any(arg.startswith("http") or arg in ("-hf", "--hf-repo", "--tools") for arg in args)


def test_manifest_rejects_modified_install_plan(tmp_path):
    path = provision.safe_path(tmp_path, provision.MANIFEST)
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version": 1, "assets": []}')
    with pytest.raises(ValueError, match="pinned plan"):
        provision.verify_manifest(tmp_path)


def test_default_inspection_does_not_download_or_create_directories(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(provision.config, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["provision_action_model"])
    monkeypatch.setattr(provision, "provision_asset", lambda *a, **k: pytest.fail("Unexpected installation"))
    provision.main()
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "inspect_only"
    assert result["dependency_approval_required"] is True
    assert not list(tmp_path.iterdir())


def test_runtime_and_manifest_cannot_be_changed_together(tmp_path, monkeypatch):
    source = archive(tmp_path, [("bin/llama-server", b"original executable", "")])
    fake_assets = (asset(b"model"),
                   provision.Asset("models/local-action-model/projector.gguf", "https://invalid.example", 9,
                                   hashlib.sha256(b"projector").hexdigest()),
                   provision.Asset("tools/local-action-model/archive.tar.gz", "https://invalid.example",
                                   source.stat().st_size, provision.digest(source)))
    monkeypatch.setattr(provision, "ASSETS", fake_assets)
    for value, content in zip(fake_assets, (b"model", b"projector", source.read_bytes())):
        provision.provision_asset(tmp_path, value, opener_for(content))
    files = provision.extract_runtime(tmp_path, provision.safe_path(tmp_path, fake_assets[2].path))
    manifest = {"schema_version": 1, "assets": [provision.asdict(a) for a in fake_assets], "runtime_files": files}
    manifest_path = provision.safe_path(tmp_path, provision.MANIFEST)
    manifest_path.write_text(json.dumps(manifest))
    assert provision.verify_manifest(tmp_path)["runtime_files"] == files
    binary = provision.safe_path(tmp_path, next(iter(files)))
    binary.write_bytes(b"modified executable")
    manifest["runtime_files"][next(iter(files))] = provision.digest(binary)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="refusing overwrite"):
        provision.verify_manifest(tmp_path)
