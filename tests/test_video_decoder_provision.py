"""Small synthetic archives only; no real decoder is downloaded or executed."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts import provision_video_decoder as P


def make_archive(extra=()):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:xz") as archive:
        for name, payload in ((f"{P.ARCHIVE_ROOT}/bin/ffmpeg", b"synthetic ffmpeg; never execute"),
                              (f"{P.ARCHIVE_ROOT}/bin/ffprobe", b"synthetic ffprobe; never execute"),
                              (f"{P.ARCHIVE_ROOT}/LICENSE.txt", b"synthetic LGPL text"),
                              (f"{P.ARCHIVE_ROOT}/doc/ignored.txt", b"must not extract")):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        for member in extra:
            archive.addfile(member, io.BytesIO(b"x" * member.size) if member.isfile() else None)
    return stream.getvalue()


@pytest.fixture
def synthetic_assets(monkeypatch):
    def setup(extra=()):
        archive = make_archive(extra)
        gpl = b"synthetic GPL text"
        assets = (replace(P.ARCHIVE, size=len(archive), sha256=hashlib.sha256(archive).hexdigest()),
                  replace(P.GPL_LICENSE, size=len(gpl), sha256=hashlib.sha256(gpl).hexdigest()))
        monkeypatch.setattr(P, "ARCHIVE", assets[0])
        monkeypatch.setattr(P, "GPL_LICENSE", assets[1])
        monkeypatch.setattr(P, "ASSETS", assets)
        monkeypatch.setattr(P, "ARCHIVE_SHA256", assets[0].sha256)
        monkeypatch.setattr(P, "LGPL_BYTES", len(b"synthetic LGPL text"))
        monkeypatch.setattr(P, "LGPL_SHA256", hashlib.sha256(b"synthetic LGPL text").hexdigest())
        monkeypatch.setattr(P, "INSTALLED_FILES", {
            f"{P.TOOLS_DIR}/bin/ffmpeg": {
                "size": len(b"synthetic ffmpeg; never execute"),
                "sha256": hashlib.sha256(b"synthetic ffmpeg; never execute").hexdigest()},
            f"{P.TOOLS_DIR}/bin/ffprobe": {
                "size": len(b"synthetic ffprobe; never execute"),
                "sha256": hashlib.sha256(b"synthetic ffprobe; never execute").hexdigest()},
            f"{P.TOOLS_DIR}/licenses/COPYING.LGPLv3": {
                "size": P.LGPL_BYTES, "sha256": P.LGPL_SHA256},
        })
        responses = {assets[0].url: archive, assets[1].url: gpl}
        calls = []

        def opener(url, **kwargs):
            calls.append(url)
            return io.BytesIO(responses[url])

        return opener, calls

    return setup


def test_default_inspection_has_no_network_writes_or_execution(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Inspection must not install or download")

    monkeypatch.setattr(P, "install", forbidden)
    monkeypatch.setattr(P, "urlopen", forbidden)
    monkeypatch.setattr(P, "inspect", lambda: {"mode": "inspect_only"})
    P.main([])
    assert json.loads(capsys.readouterr().out) == {"mode": "inspect_only"}
    assert not list(tmp_path.iterdir())


def test_synthetic_install_is_local_bounded_verified_and_idempotent(tmp_path, synthetic_assets):
    opener, calls = synthetic_assets()
    result = P.install(tmp_path, opener=opener)
    assert len(calls) == 2
    assert set(Path(result["bin_dir"]).iterdir()) == {Path(result["ffmpeg"]), Path(result["ffprobe"])}
    assert result["archive_sha256"] == P.ARCHIVE_SHA256
    assert result["installed_bytes"] > 0
    assert result["total_disk_bytes"] == sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
    assert not list(tmp_path.rglob("ignored.txt"))
    assert P.install(tmp_path, opener=opener) == result
    assert len(calls) == 2


def test_mismatched_existing_asset_is_preserved_without_network(tmp_path, synthetic_assets):
    opener, calls = synthetic_assets()
    destination = P.safe_path(tmp_path, P.ARCHIVE.path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"preserve me")
    with pytest.raises(ValueError, match="refusing overwrite"):
        P.install(tmp_path, opener=opener)
    assert destination.read_bytes() == b"preserve me"
    assert calls == []


def test_hash_or_oversize_download_is_rejected_before_publication(tmp_path, synthetic_assets):
    synthetic_assets()
    for payload in (b"bad", b"bad" * P.ARCHIVE.size):
        with pytest.raises(ValueError, match="pinned|SHA-256"):
            P._download(tmp_path, P.ARCHIVE, lambda *args, **kwargs: io.BytesIO(payload))
        assert not P.safe_path(tmp_path, P.ARCHIVE.path).exists()


@pytest.mark.parametrize("kind", ["traversal", "absolute", "escaping_link", "selected_link", "device", "duplicate"])
def test_unsafe_archive_entries_never_extract_executables(tmp_path, synthetic_assets, kind):
    name = {"traversal": f"{P.ARCHIVE_ROOT}/../escape", "absolute": "/escape",
            "escaping_link": f"{P.ARCHIVE_ROOT}/doc/link", "selected_link": f"{P.ARCHIVE_ROOT}/bin/ffmpeg",
            "device": f"{P.ARCHIVE_ROOT}/device", "duplicate": f"{P.ARCHIVE_ROOT}/LICENSE.txt"}[kind]
    member = tarfile.TarInfo(name)
    if kind in {"escaping_link", "selected_link"}:
        member.type, member.linkname = tarfile.SYMTYPE, "../../../escape"
    elif kind == "device":
        member.type = tarfile.CHRTYPE
    opener, _ = synthetic_assets([member])
    with pytest.raises(ValueError):
        P.install(tmp_path, opener=opener)
    assert not (tmp_path / P.TOOLS_DIR / "bin").exists()
    assert not (tmp_path / "escape").exists()


def test_tampered_executable_or_manifest_fails_and_is_preserved(tmp_path, synthetic_assets):
    opener, _ = synthetic_assets()
    result = P.install(tmp_path, opener=opener)
    binary = Path(result["ffmpeg"])
    binary.write_bytes(b"preserved tampered executable")
    with pytest.raises(ValueError, match="refusing overwrite"):
        P.install(tmp_path, opener=opener)
    assert binary.read_bytes() == b"preserved tampered executable"


def test_symlink_destination_is_rejected_without_writing_outside(tmp_path, synthetic_assets):
    opener, _ = synthetic_assets()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / P.TOOLS_DIR).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        P.install(tmp_path, opener=opener)
    assert not list(outside.iterdir())


def test_archive_member_bound_is_enforced(tmp_path, synthetic_assets, monkeypatch):
    opener, _ = synthetic_assets()
    monkeypatch.setattr(P, "MAX_ARCHIVE_MEMBERS", 2)
    with pytest.raises(ValueError, match="member bound"):
        P.install(tmp_path, opener=opener)
    assert not (tmp_path / P.TOOLS_DIR / "bin").exists()


def test_readiness_and_idempotent_install_do_not_unpack_archive_again(tmp_path, synthetic_assets, monkeypatch):
    opener, _ = synthetic_assets()
    expected = P.install(tmp_path, opener=opener)

    def forbidden(*args, **kwargs):
        pytest.fail("Readiness must hash pinned bytes directly without tar/xz unpacking")

    monkeypatch.setattr(P, "_selected_files", forbidden)
    monkeypatch.setattr(P.tarfile, "open", forbidden)
    assert P.verify_ready(tmp_path) == expected
    assert P.install(tmp_path, opener=opener) == expected


def test_corrupted_binary_with_matching_forged_manifest_is_rejected(tmp_path, synthetic_assets):
    opener, _ = synthetic_assets()
    result = P.install(tmp_path, opener=opener)
    binary = Path(result["ffmpeg"])
    payload = b"forged executable and forged manifest"
    binary.write_bytes(payload)
    manifest_path = P.safe_path(tmp_path, P.MANIFEST)
    manifest = json.loads(manifest_path.read_text())
    manifest["installed_files"][f"{P.TOOLS_DIR}/bin/ffmpeg"] = {
        "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest differs"):
        P.verify_ready(tmp_path)
    assert binary.read_bytes() == payload
