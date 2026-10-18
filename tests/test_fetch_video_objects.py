import hashlib
import io
import json
from unittest.mock import Mock

import pytest

from scripts import fetch_video_objects as fetch


def asset(payload=b"synthetic-onnx"):
    return fetch.Asset(fetch.MODEL_RELATIVE, "https://invalid.example/model", len(payload),
                       hashlib.sha256(payload).hexdigest())


def test_explicit_download_checks_bytes_and_hash_and_never_repeats_network(tmp_path):
    value = asset()
    fetch.download_asset(tmp_path, value, lambda *a, **k: io.BytesIO(b"synthetic-onnx"))
    assert fetch.verify_asset(tmp_path, value)
    no_network = Mock(side_effect=AssertionError("unexpected network"))
    fetch.download_asset(tmp_path, value, no_network)
    no_network.assert_not_called()


@pytest.mark.parametrize("payload", [b"synthetic-onnX", b"truncated", b"oversized-payload-for-the-pin"])
def test_corrupt_or_wrong_size_download_never_installed(tmp_path, payload):
    with pytest.raises(ValueError, match="pinned"):
        fetch.download_asset(tmp_path, asset(), lambda *a, **k: io.BytesIO(payload))
    destination = fetch.checked_path(tmp_path, fetch.MODEL_RELATIVE)
    assert not destination.exists()
    assert not list(destination.parent.glob("tmp*"))


def test_existing_mismatch_is_preserved_without_network(tmp_path):
    destination = fetch.checked_path(tmp_path, fetch.MODEL_RELATIVE)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"operator-owned-existing-content")
    no_network = Mock()
    with pytest.raises(ValueError, match="refusing overwrite"):
        fetch.download_asset(tmp_path, asset(), no_network)
    assert destination.read_bytes() == b"operator-owned-existing-content"
    no_network.assert_not_called()


def test_default_inspection_creates_no_files_or_downloads(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch.config, "ROOT", tmp_path)
    download = Mock(side_effect=AssertionError("unexpected download"))
    monkeypatch.setattr(fetch, "download_asset", download)
    fetch.main([])
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "inspect_only"
    assert not any(result["verified_present"].values())
    assert not list(tmp_path.iterdir())
    download.assert_not_called()


@pytest.mark.parametrize("relative", ["data/raw/model.onnx", "../model.onnx", "/tmp/model.onnx",
                                      "models/opencv-zoo/../../data/raw/model.onnx"])
def test_destinations_are_fixed_and_cannot_reach_raw(tmp_path, relative):
    with pytest.raises(ValueError, match="fixed installation plan"):
        fetch.checked_path(tmp_path, relative)


def test_installation_symlink_rejected(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "models").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        fetch.checked_path(tmp_path, fetch.MODEL_RELATIVE)


def test_verify_ready_checks_license_manifest_and_weights_without_network(tmp_path, monkeypatch):
    weight, license_value = b"synthetic-onnx", b"synthetic license"
    values = (asset(weight), fetch.Asset(fetch.LICENSE_RELATIVE, "https://invalid.example/license",
                                        len(license_value), hashlib.sha256(license_value).hexdigest()))
    monkeypatch.setattr(fetch, "ASSETS", values)
    for value, payload in zip(values, (weight, license_value)):
        fetch.download_asset(tmp_path, value, lambda *a, content=payload, **k: io.BytesIO(content))
    manifest = {**fetch.expected_manifest(), "retrieved_at_utc": "2026-10-17T00:00:00+00:00"}
    path = fetch.checked_path(tmp_path, fetch.MANIFEST_RELATIVE)
    path.write_text(json.dumps(manifest))
    ready = fetch.verify_ready(tmp_path)
    assert ready["source_commit"] == fetch.SOURCE_COMMIT
    assert ready["license"] == "Apache-2.0"
    manifest["assets"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest differs"):
        fetch.verify_ready(tmp_path)


def test_destination_race_does_not_replace_existing_content(tmp_path):
    value = asset()
    destination = fetch.checked_path(tmp_path, fetch.MODEL_RELATIVE)

    def opener(*args, **kwargs):
        destination.write_bytes(b"concurrent operator content")
        return io.BytesIO(b"synthetic-onnx")

    with pytest.raises(FileExistsError):
        fetch.download_asset(tmp_path, value, opener)
    assert destination.read_bytes() == b"concurrent operator content"
    assert not list(destination.parent.glob("tmp*"))
