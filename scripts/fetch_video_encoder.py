"""Explicit one-time download of a pinned local encoder; never called by inference."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

from src import config


WEIGHTS_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx"
WEIGHTS_SHA256 = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
LICENSE_URL = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/image_classification_mobilenet/LICENSE"
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def provision(url, destination, expected_hash, maximum):
    if destination.resolve().is_relative_to(config.DATA_RAW.resolve()):
        raise ValueError("Encoder files must not be written to data/raw")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Existing local encoder/license differs; refusing overwrite")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=60) as response:
                count = 0
                while chunk := response.read(256 * 1024):
                    count += len(chunk)
                    if count > maximum:
                        raise ValueError("Encoder download exceeds the pinned size bound")
                    output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Downloaded encoder/license SHA-256 differs from the pinned source")
        if destination.exists():
            raise ValueError("Destination appeared during download; refusing overwrite")
        temporary.rename(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=config.ROOT / "models/encoders/mobilenetv2.onnx")
    args = parser.parse_args()
    license_path = config.ROOT / "licenses/opencv-mobilenet/LICENSE"
    provision(WEIGHTS_URL, args.output, WEIGHTS_SHA256, 14_000_000)
    provision(LICENSE_URL, license_path, LICENSE_SHA256, 20_000)
    manifest = {"name": "OpenCV Zoo MobileNetV2 FP32", "weights_url": WEIGHTS_URL,
                "weights_sha256": WEIGHTS_SHA256, "weights_bytes": args.output.stat().st_size,
                "license": "Apache-2.0", "license_url": LICENSE_URL,
                "license_sha256": LICENSE_SHA256,
                "inference_network_required": False}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
