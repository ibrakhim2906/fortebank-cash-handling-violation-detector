"""Inspect pinned local FFmpeg tools; --install is for explicit human approval only.

No command executes a downloaded binary. Installation is confined to a separate
project tools directory and never changes PATH, Python packages, models or data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import tarfile
import tempfile
from urllib.request import urlopen

from src import config

TOOLS_DIR = "tools/local-video-decoder"
RELEASE = "autobuild-2026-10-16-13-04"
RELEASE_ID = 414971902
ASSET_ID = 647361913
BUILDER_COMMIT = "63ac3265b0789ea16c7b1f8351cd2ad195a28722"
FFMPEG_COMMIT = "46d899d7a02bc63237472cc4897258d5ddf274ac"
ARCHIVE_NAME = "ffmpeg-n8.1.3-18-g46d899d7a0-linux64-lgpl-8.1.tar.xz"
ARCHIVE_ROOT = ARCHIVE_NAME.removesuffix(".tar.xz")
ARCHIVE_SHA256 = "70df7305a38b8dc518e35e44d1f165edc74033082ae53b5d977acfa2849fe8fb"
LGPL_SHA256 = "da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768"
LGPL_BYTES = 7651
MAX_ARCHIVE_MEMBERS = 2000
MAX_UNPACKED_BYTES = 1024 ** 3
MAX_MANIFEST_BYTES = 65536
MANIFEST = f"{TOOLS_DIR}/manifest.json"


@dataclass(frozen=True)
class Asset:
    path: str
    url: str
    size: int
    sha256: str


ARCHIVE = Asset(
    f"{TOOLS_DIR}/archive/{ARCHIVE_NAME}",
    f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{RELEASE}/{ARCHIVE_NAME}",
    137124016, ARCHIVE_SHA256)
GPL_LICENSE = Asset(
    f"{TOOLS_DIR}/licenses/COPYING.GPLv3",
    f"https://raw.githubusercontent.com/FFmpeg/FFmpeg/{FFMPEG_COMMIT}/COPYING.GPLv3",
    35147, "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903")
ASSETS = (ARCHIVE, GPL_LICENSE)
SELECTED = {
    f"{ARCHIVE_ROOT}/bin/ffmpeg": f"{TOOLS_DIR}/bin/ffmpeg",
    f"{ARCHIVE_ROOT}/bin/ffprobe": f"{TOOLS_DIR}/bin/ffprobe",
    f"{ARCHIVE_ROOT}/LICENSE.txt": f"{TOOLS_DIR}/licenses/COPYING.LGPLv3",
}
# These member bytes were derived from the approved archive, whose complete
# SHA-256 and archive safety checks passed during the first real installation.
# Runtime checks bind to these pins instead of decompressing 137 MB repeatedly.
INSTALLED_FILES = {
    f"{TOOLS_DIR}/bin/ffmpeg": {
        "size": 142166568, "sha256": "9087520002e707613b9c20fd3e251474bf85eaa16156ba4df42652231f7b1046"},
    f"{TOOLS_DIR}/bin/ffprobe": {
        "size": 141933032, "sha256": "db87fc6e0f78cf83b7aacb2d21f7658216f0f861834f3612c9931728256e026e"},
    f"{TOOLS_DIR}/licenses/COPYING.LGPLv3": {"size": LGPL_BYTES, "sha256": LGPL_SHA256},
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def safe_path(root, relative):
    root, rel = Path(root).resolve(), PurePosixPath(relative)
    if (rel.is_absolute() or ".." in rel.parts or "\\" in relative or ":" in relative
            or not relative.startswith(TOOLS_DIR + "/")):
        raise ValueError("Decoder destination is outside the fixed tools directory")
    destination = root.joinpath(*rel.parts)
    for part in (destination, *destination.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Decoder destination contains a symlink")
    if not destination.resolve().is_relative_to(root):
        raise ValueError("Decoder destination escapes the project")
    return destination


def verify_asset(root, asset):
    path = safe_path(root, asset.path)
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != asset.size or digest(path) != asset.sha256:
        raise ValueError("Existing decoder asset differs; refusing overwrite")
    return True


def _download(root, asset, opener):
    if verify_asset(root, asset):
        return
    destination = safe_path(root, asset.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            count = 0
            with opener(asset.url, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > asset.size:
                        raise ValueError("Decoder download exceeds the pinned byte count")
                    output.write(block)
        if count != asset.size or digest(temporary) != asset.sha256:
            raise ValueError("Decoder download size or SHA-256 differs from the pinned asset")
        os.link(temporary, destination)  # Atomic publication without replacing an existing file.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _archive_name(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
            or not path.parts or path.parts[0] != ARCHIVE_ROOT):
        raise ValueError("Unsafe decoder archive path")
    return str(path)


def _members(archive):
    members, total = {}, 0
    for member in archive:
        if len(members) >= MAX_ARCHIVE_MEMBERS:
            raise ValueError("Decoder archive exceeds its member bound")
        name = _archive_name(member.name)
        if name in members or member.size < 0:
            raise ValueError("Duplicate or invalid decoder archive entry")
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError("Unsupported decoder archive special file")
        total += member.size
        if total > MAX_UNPACKED_BYTES:
            raise ValueError("Decoder archive exceeds its unpacked byte bound")
        members[name] = member
    for name, member in members.items():
        if member.issym() or member.islnk():
            visited = set()
            while member.issym() or member.islnk():
                if name in visited:
                    raise ValueError("Cyclic decoder archive link")
                visited.add(name)
                target = member.linkname
                if PurePosixPath(target).is_absolute() or "\\" in target or ":" in target:
                    raise ValueError("Unsafe decoder archive link")
                if member.issym():
                    target = str(PurePosixPath(name).parent / target)
                name = _archive_name(posixpath.normpath(target))
                if name not in members:
                    raise ValueError("Decoder archive link target is missing")
                member = members[name]
    for selected in SELECTED:
        if selected not in members or not members[selected].isfile():
            raise ValueError("Required decoder binaries/license must be regular archive files")
    return members


def _selected_files(root, *, install_files=False):
    if not verify_asset(root, ARCHIVE):
        raise ValueError("Pinned decoder archive is missing")
    expected = {}
    archive_path = safe_path(root, ARCHIVE.path)
    with tarfile.open(archive_path, "r:xz") as archive:
        members = _members(archive)
        # Hash and check every existing destination before writing any extracted file.
        for name, relative in SELECTED.items():
            member = members[name]
            value = hashlib.sha256()
            with archive.extractfile(member) as stream:
                while block := stream.read(1024 * 1024):
                    value.update(block)
            expected[relative] = {"size": member.size, "sha256": value.hexdigest()}
            if relative.endswith("COPYING.LGPLv3") and (
                    member.size != LGPL_BYTES or value.hexdigest() != LGPL_SHA256):
                raise ValueError("Archive LGPL license differs from the pinned FFmpeg source")
            destination = safe_path(root, relative)
            if destination.exists() and (not destination.is_file()
                    or destination.stat().st_size != member.size or digest(destination) != value.hexdigest()):
                raise ValueError("Existing decoder executable/license differs; refusing overwrite")
        if expected != INSTALLED_FILES:
            raise ValueError("Decoder archive member bytes differ from the approved installed-file pins")
        for name, relative in SELECTED.items():
            destination = safe_path(root, relative)
            if destination.exists():
                continue
            if not install_files:
                raise ValueError("Installed decoder executable/license is missing")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                    temporary = Path(output.name)
                    with archive.extractfile(members[name]) as stream:
                        while block := stream.read(1024 * 1024):
                            output.write(block)
                if digest(temporary) != expected[relative]["sha256"]:
                    raise ValueError("Decoder extraction hash differs")
                temporary.chmod(0o755 if "/bin/" in relative else 0o644)
                os.link(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return expected


def expected_manifest():
    return {"schema_version": 1, "component": "FFmpeg/ffprobe Linux x86_64 LGPL static",
            "release": RELEASE, "release_id": RELEASE_ID, "asset_id": ASSET_ID,
            "builder_commit": BUILDER_COMMIT, "ffmpeg_commit": FFMPEG_COMMIT,
            "archive_sha256": ARCHIVE_SHA256, "license": "LGPL-3.0-or-later",
            "assets": [asdict(asset) for asset in ASSETS],
            "installed_files": INSTALLED_FILES,
            "runtime_network_required": False, "executes_binaries_during_install": False}


def _manifest(root):
    path = safe_path(root, MANIFEST)
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Decoder installation manifest is missing/oversized")
    value = json.loads(path.read_text())
    if any(value.get(key) != wanted for key, wanted in expected_manifest().items()):
        raise ValueError("Existing decoder manifest differs; refusing replacement")
    if not isinstance(value.get("installed_at_utc"), str):
        raise ValueError("Decoder installation provenance is missing")
    return value


def verify_ready(root=config.ROOT):
    root = Path(root)
    manifest = _manifest(root)
    if not all(verify_asset(root, asset) for asset in ASSETS):
        raise ValueError("Decoder installation assets are incomplete")
    files = INSTALLED_FILES
    for relative, expected in files.items():
        path = safe_path(root, relative)
        if (not path.is_file() or path.stat().st_size != expected["size"]
                or digest(path) != expected["sha256"]):
            raise ValueError("Existing decoder executable/license differs from approved pins; refusing overwrite")
    if not all(os.access(safe_path(root, f"{TOOLS_DIR}/bin/{name}"), os.X_OK)
               for name in ("ffmpeg", "ffprobe")):
        raise ValueError("Installed decoder tools are not executable")
    return {"bin_dir": str(safe_path(root, f"{TOOLS_DIR}/bin/ffmpeg").parent),
            "ffmpeg": str(safe_path(root, f"{TOOLS_DIR}/bin/ffmpeg")),
            "ffprobe": str(safe_path(root, f"{TOOLS_DIR}/bin/ffprobe")),
            "archive_sha256": ARCHIVE_SHA256, "asset_sha256": ARCHIVE_SHA256,
            "installed_bytes": sum(item["size"] for item in files.values()) + GPL_LICENSE.size,
            "total_disk_bytes": sum(asset.size for asset in ASSETS)
                                + sum(item["size"] for item in files.values())
                                + safe_path(root, MANIFEST).stat().st_size,
            "provenance": manifest}


def inspect(root=config.ROOT):
    present = {asset.path: verify_asset(root, asset) for asset in ASSETS}
    ready = None
    if safe_path(root, MANIFEST).exists():
        ready = verify_ready(root)
    return {"mode": "inspect_only", "approval_required_before_install": True,
            **expected_manifest(), "network_download_bytes": sum(asset.size for asset in ASSETS),
            "maximum_install_disk_bytes": sum(asset.size for asset in ASSETS)
                                          + MAX_UNPACKED_BYTES + MAX_MANIFEST_BYTES,
            "exact_installed_bytes_known": ready is not None,
            "verified_assets_present": present, "ready": ready}


def install(root=config.ROOT, *, opener=urlopen):
    """Call only following explicit approval; this function never executes a tool."""
    root = Path(root)
    if safe_path(root, MANIFEST).exists():
        return verify_ready(root)
    for asset in ASSETS:
        verify_asset(root, asset)  # Fail on an existing mismatch before any download.
    for asset in ASSETS:
        _download(root, asset, opener)
    files = _selected_files(root, install_files=True)
    manifest = {**expected_manifest(), "installed_files": files,
                "installed_at_utc": datetime.now(timezone.utc).isoformat()}
    destination = safe_path(root, MANIFEST)
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("Decoder manifest exceeds its byte bound")
    with destination.open("xb") as stream:
        stream.write(payload)
    return verify_ready(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Install only after explicit human approval")
    args = parser.parse_args(argv)
    print(json.dumps(install() if args.install else inspect(), indent=2), flush=True)


if __name__ == "__main__":
    main()
