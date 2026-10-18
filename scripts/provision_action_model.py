"""Inspect a pinned local action-label model; install only after human approval."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from urllib.request import urlopen

from src import config


MODEL_REVISION = "52d6c8ffea26cc873ac5ad116f8631268d7eb503"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct-GGUF"
MODEL_SHA256 = "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"
PROJECTOR_SHA256 = "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82"
RELEASE = "b11707"
MODEL_DIR = "models/local-action-model"
TOOLS_DIR = "tools/local-action-model"
MANIFEST = f"{TOOLS_DIR}/manifest.json"


@dataclass(frozen=True)
class Asset:
    path: str
    url: str
    size: int
    sha256: str


MODEL_BASE = f"https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/{MODEL_REVISION}"
ASSETS = (
    Asset(f"{MODEL_DIR}/Qwen3VL-2B-Instruct-Q4_K_M.gguf", f"{MODEL_BASE}/Qwen3VL-2B-Instruct-Q4_K_M.gguf", 1107409952,
          "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae"),
    Asset(f"{MODEL_DIR}/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", f"{MODEL_BASE}/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", 445053216,
          "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82"),
    Asset(f"{TOOLS_DIR}/llama-{RELEASE}-bin-ubuntu-x64.tar.gz",
          f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/llama-{RELEASE}-bin-ubuntu-x64.tar.gz",
          17914097, "88879871f765c3dec22260046e10a76e2ed38de5bdf4a7f68049245a8f4056c8"),
    Asset(f"{MODEL_DIR}/LICENSE.apache-2.0",
          "https://raw.githubusercontent.com/QwenLM/Qwen3-VL/96588727e44c78b25ba03ea03b8e12f7e64fd0da/LICENSE",
          11357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
    Asset(f"{TOOLS_DIR}/LICENSE.mit", "https://raw.githubusercontent.com/ggml-org/llama.cpp/b11707/LICENSE",
          1078, "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d"),
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """All writes stay in the two approved project-local installation trees."""
    root = root.resolve()
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or not any(
        relative.startswith(prefix + "/") for prefix in (MODEL_DIR, TOOLS_DIR)
    ):
        raise ValueError("Local model destination is outside the approved directories")
    result = root.joinpath(*rel.parts)
    for path in (result, *result.parents):
        if path == root:
            break
        if path.is_symlink():
            raise ValueError("Symlink in local model destination")
    if not result.resolve().is_relative_to(root):
        raise ValueError("Local model destination escapes the project")
    return result


def verify_asset(root: Path, asset: Asset) -> bool:
    path = safe_path(root, asset.path)
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != asset.size or digest(path) != asset.sha256:
        raise ValueError(f"Existing asset differs; refusing overwrite: {asset.path}")
    return True


def provision_asset(root: Path, asset: Asset, opener=urlopen) -> None:
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
                        raise ValueError("Download exceeds pinned byte count")
                    output.write(block)
        if count != asset.size or digest(temporary) != asset.sha256:
            raise ValueError("Downloaded asset size or SHA-256 differs")
        os.link(temporary, destination)  # atomic, fails if destination appeared
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_runtime(root: Path, archive_path: Path, *, verify_only=False) -> dict[str, str]:
    """Materialize only bounded internal regular files; never create archive links."""
    runtime_prefix = f"{TOOLS_DIR}/runtime"
    extracted = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            name = str(PurePosixPath(member.name))
            if name in members:
                raise ValueError("Duplicate runtime archive path")
            members[name] = member
        if len(members) > 2000 or sum(max(0, m.size) for m in members.values()) > 512 * 1024 * 1024:
            raise ValueError("Runtime archive exceeds extraction bounds")
        for name, member in members.items():
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError("Unsafe runtime archive path")
            safe_path(root, f"{runtime_prefix}/{name}")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError("Unsupported runtime archive entry")
            if member.issym() or member.islnk():
                target = PurePosixPath(member.linkname)
                if target.is_absolute() or ".." in target.parts:
                    raise ValueError("Unsafe runtime archive link")
        for name, member in members.items():
            if member.isdir():
                continue
            source = member
            visited = set()
            while source.issym() or source.islnk():
                if source.name in visited:
                    raise ValueError("Cyclic runtime archive link")
                visited.add(source.name)
                target = (PurePosixPath(source.name).parent / source.linkname if source.issym()
                          else PurePosixPath(source.linkname))
                source = members.get(str(target))
                if source is None:
                    raise ValueError("Runtime link target is absent")
            if not source.isfile():
                raise ValueError("Runtime link target is not a regular file")
            relative = f"{runtime_prefix}/{name}"
            destination = safe_path(root, relative)
            stream = archive.extractfile(source)
            if stream is None:
                raise ValueError("Runtime archive file could not be read")
            with stream:
                content = stream.read(512 * 1024 * 1024 + 1)
            expected = hashlib.sha256(content).hexdigest()
            if destination.exists():
                if not destination.is_file() or digest(destination) != expected:
                    raise ValueError("Existing runtime differs; refusing overwrite")
            elif verify_only:
                raise ValueError("A pinned runtime file is missing")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as output:
                    output.write(content)
                destination.chmod(0o700 if source.mode & 0o111 else 0o600)
            extracted[relative] = expected
    return extracted


def verify_manifest(root: Path) -> dict:
    path = safe_path(root, MANIFEST)
    manifest = json.loads(path.read_text())
    expected = [asdict(asset) for asset in ASSETS]
    if manifest.get("assets") != expected or manifest.get("schema_version") != 1:
        raise ValueError("Local model manifest does not match the pinned plan")
    for asset in ASSETS:
        if not verify_asset(root, asset):
            raise ValueError("A pinned local model asset is missing")
    runtime_files = manifest.get("runtime_files", {})
    if not runtime_files:
        raise ValueError("No verified runtime files in manifest")
    pinned_runtime_files = extract_runtime(root, safe_path(root, ASSETS[2].path), verify_only=True)
    if runtime_files != pinned_runtime_files:
        raise ValueError("Runtime manifest differs from the pinned archive")
    for relative, expected_hash in runtime_files.items():
        if not relative.startswith(f"{TOOLS_DIR}/runtime/"):
            raise ValueError("Runtime manifest file is outside its directory")
        runtime = safe_path(root, relative)
        if not runtime.is_file() or digest(runtime) != expected_hash:
            raise ValueError("Local runtime hash mismatch")
    return manifest


def verify_ready(root: Path | None = None) -> dict:
    """Verify all local bytes before a caller reads media; performs no network I/O."""
    root = config.ROOT if root is None else root
    manifest = verify_manifest(root)
    return {**manifest, "model_id": MODEL_ID, "weights_sha256": MODEL_SHA256,
            "projector_sha256": PROJECTOR_SHA256, "model_revision": MODEL_REVISION,
            "model_path": str(safe_path(root, ASSETS[0].path)),
            "projector_path": str(safe_path(root, ASSETS[1].path)),
            "runtime_release": RELEASE, "alias": "local-action-review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Download/install only after explicit human dependency approval")
    args = parser.parse_args()
    root = config.ROOT
    statuses = [{**asdict(asset), "verified_present": verify_asset(root, asset)} for asset in ASSETS]
    if not args.install:
        print(json.dumps({"mode": "inspect_only", "dependency_approval_required": True,
                          "total_download_bytes": sum(asset.size for asset in ASSETS), "assets": statuses}, indent=2))
        return
    for asset in ASSETS:
        provision_asset(root, asset)
    runtime_files = extract_runtime(root, safe_path(root, ASSETS[2].path))
    manifest = {"schema_version": 1, "assets": [asdict(asset) for asset in ASSETS],
                "runtime_files": runtime_files, "inference_network_required": False}
    destination = safe_path(root, MANIFEST)
    payload = json.dumps(manifest, indent=2) + "\n"
    if destination.exists():
        if destination.read_text() != payload:
            raise ValueError("Existing installation manifest differs; refusing overwrite")
    else:
        with destination.open("x") as output:
            output.write(payload)
    verify_manifest(root)
    print(json.dumps({"installed": True, "manifest": str(destination), "runtime_file_count": len(runtime_files)}))


if __name__ == "__main__":
    main()
