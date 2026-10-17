"""Inspect or explicitly download the pinned Apache-2.0 OpenCV Zoo YOLOX-S."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

from src import config


MODEL_ID = "OpenCVZoo/YOLOX-S-2022nov"
SOURCE_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
WEIGHTS_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
WEIGHTS_BYTES = 35_858_002
LICENSE_SHA256 = "0ec3668d3274bcf29e8a29e9576d5a2cd96fc78d3c5bec4387355a796e5d9088"
MODEL_RELATIVE = "models/opencv-zoo/object_detection_yolox_2022nov.onnx"
LICENSE_RELATIVE = "licenses/opencv-yolox/LICENSE"
MANIFEST_RELATIVE = "models/opencv-zoo/yolox-s.manifest.json"
MODEL_PATH = config.ROOT / MODEL_RELATIVE
WEIGHTS_URL = (f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{SOURCE_COMMIT}/"
               "models/object_detection_yolox/object_detection_yolox_2022nov.onnx")
LICENSE_URL = (f"https://raw.githubusercontent.com/opencv/opencv_zoo/{SOURCE_COMMIT}/"
               "models/object_detection_yolox/LICENSE")


@dataclass(frozen=True)
class Asset:
    path: str
    url: str
    size: int
    sha256: str


ASSETS = (Asset(MODEL_RELATIVE, WEIGHTS_URL, WEIGHTS_BYTES, WEIGHTS_SHA256),
          Asset(LICENSE_RELATIVE, LICENSE_URL, 11_371, LICENSE_SHA256))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def checked_path(root, relative):
    if relative not in {MODEL_RELATIVE, LICENSE_RELATIVE, MANIFEST_RELATIVE}:
        raise ValueError("Object model destination is outside the fixed installation plan")
    root = Path(root).resolve()
    destination = root / relative
    for part in (destination, *destination.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Object model destination contains a symlink")
    if not destination.resolve().is_relative_to(root):
        raise ValueError("Object model destination escapes the project")
    return destination


def verify_asset(root, asset):
    destination = checked_path(root, asset.path)
    if not destination.exists():
        return False
    if (not destination.is_file() or destination.stat().st_size != asset.size
            or digest(destination) != asset.sha256):
        raise ValueError(f"Existing object model asset differs; refusing overwrite: {asset.path}")
    return True


def download_asset(root, asset, opener=urlopen):
    if verify_asset(root, asset):
        return
    destination = checked_path(root, asset.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            size = 0
            with opener(asset.url, timeout=60) as response:
                while block := response.read(256 * 1024):
                    size += len(block)
                    if size > asset.size:
                        raise ValueError("Object model download exceeds the pinned byte count")
                    output.write(block)
        if size != asset.size or digest(temporary) != asset.sha256:
            raise ValueError("Object model download differs from the pinned size or SHA-256")
        os.link(temporary, destination)  # Atomic and cannot replace another file.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def expected_manifest():
    return {"schema_version": 1, "model_id": MODEL_ID, "source_commit": SOURCE_COMMIT,
            "license": "Apache-2.0", "assets": [asdict(asset) for asset in ASSETS],
            "inference_network_required": False}


def verify_ready(root=None):
    """Verify existing local weights/license/provenance; never access a network."""
    root = config.ROOT if root is None else Path(root)
    for asset in ASSETS:
        if not verify_asset(root, asset):
            raise ValueError("Pinned YOLOX-S assets are missing; use the explicit downloader")
    manifest = json.loads(checked_path(root, MANIFEST_RELATIVE).read_text())
    plan = expected_manifest()
    if any(manifest.get(key) != value for key, value in plan.items()):
        raise ValueError("Local object model manifest differs from the pinned plan")
    if not isinstance(manifest.get("retrieved_at_utc"), str):
        raise ValueError("Local object model retrieval provenance is missing")
    return {**plan, "model_path": str(checked_path(root, MODEL_RELATIVE)),
            "weights_sha256": WEIGHTS_SHA256, "weights_bytes": WEIGHTS_BYTES,
            "model_revision": SOURCE_COMMIT,
            "license_path": str(checked_path(root, LICENSE_RELATIVE)),
            "license_url": LICENSE_URL, "license_sha256": LICENSE_SHA256,
            "weights_url": WEIGHTS_URL, "retrieved_at_utc": manifest["retrieved_at_utc"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the fixed model assets explicitly")
    args = parser.parse_args(argv)
    root = config.ROOT
    if not args.download:
        print(json.dumps({"mode": "inspect_only", **expected_manifest(),
                          "verified_present": {asset.path: verify_asset(root, asset) for asset in ASSETS}}, indent=2))
        return
    for asset in ASSETS:
        download_asset(root, asset)
    path = checked_path(root, MANIFEST_RELATIVE)
    if path.exists():
        verify_ready(root)
    else:
        manifest = {**expected_manifest(), "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
        with path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(manifest, indent=2) + "\n")
    ready = verify_ready(root)
    print(json.dumps({"ready": True, "model_id": ready["model_id"],
                      "weights_bytes": ready["weights_bytes"], "weights_sha256": ready["weights_sha256"],
                      "source_commit": ready["source_commit"], "license": ready["license"]}))


if __name__ == "__main__":
    main()
