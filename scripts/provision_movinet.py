"""Pinned, explicit provisioning for the approved local temporal-model experiment."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import tarfile

from scripts.fetch_video_encoder import provision
from src import config


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/movinet/movinet_a0_stream.tflite"
LICENSE_DIR = ROOT / "licenses/movinet"
VARIANT = "a0-stream-kinetics-600-classification-tflite-float16"
WEIGHTS_SHA256 = "2afdb24bc127104171df2d133245beb3b8828705cc8d93f1aeb1e24392838924"
ARCHIVE_SHA256 = "513f859418f6a962fa4de09910c3ce6461440a99cd8d72d8c716a1198218940a"
ARCHIVE_URL = f"https://www.kaggle.com/api/v1/models/google/movinet/tfLite/{VARIANT}/2/download"
RUNTIME_VERSION = "2.3.0"


def verify_ready(require_runtime=True):
    """No network: reject changed weights or mismatched recorded weight rights."""
    if not MODEL_PATH.is_file():
        raise ValueError("Pinned local MoViNet weights are missing; run explicit provisioning")
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("MoViNet weights differ from the pinned experiment")
    evidence_path = LICENSE_DIR / "kaggle-model-metadata.json"
    manifest = json.loads((LICENSE_DIR / "provenance.json").read_text())
    evidence_bytes = evidence_path.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != manifest["license_evidence_sha256"]:
        raise ValueError("Recorded model-license evidence has changed")
    evidence = json.loads(evidence_bytes)
    model = next(x for x in evidence["models"] if x["ref"] == "google/movinet")
    instance = next(x for x in model["instances"] if x["slug"] == VARIANT)
    if (instance["licenseName"], instance["versionNumber"], instance["framework"]) != ("Apache 2.0", 2, "tfLite"):
        raise ValueError("Exact selected weight license/version is not verified")
    if manifest["weights_sha256"] != WEIGHTS_SHA256 or manifest["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Weight provenance does not match the pinned model")
    for filename, key in [("Apache-2.0.txt", "license_sha256"),
                          ("LiteRT-Apache-2.0.txt", "runtime_license_sha256")]:
        if hashlib.sha256((LICENSE_DIR / filename).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("Preserved license text differs from provenance")
    installed = None
    try:
        installed = importlib.metadata.version("ai-edge-litert")
    except importlib.metadata.PackageNotFoundError:
        if require_runtime:
            raise ValueError("Runtime installation awaits approval: ai-edge-litert==2.3.0") from None
    if require_runtime and installed != RUNTIME_VERSION:
        raise ValueError("MoViNet runtime must be exactly ai-edge-litert==2.3.0")
    return {"model_path": str(MODEL_PATH), "weights_sha256": WEIGHTS_SHA256,
            "runtime_version": installed, "variant": VARIANT, "version": 2}


def install_weights():
    """Download only the public model; never run automatically during inference."""
    archive = MODEL_PATH.parent / "source.tar.gz"
    provision(ARCHIVE_URL, archive, ARCHIVE_SHA256, 20_000_000)
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes()), mode="r:gz") as source:
        members = source.getmembers()
        if len(members) != 1 or members[0].name != "2.tflite" or not members[0].isfile():
            raise ValueError("Unexpected model archive member; refusing extraction")
        if members[0].size != 7_992_000:
            raise ValueError("Model member size differs from the pinned release")
        data = source.extractfile(members[0]).read()
    if hashlib.sha256(data).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Model member checksum mismatch")
    if MODEL_PATH.resolve().is_relative_to(config.DATA_RAW.resolve()):
        raise ValueError("Model destination must not be raw data")
    if MODEL_PATH.exists():
        if MODEL_PATH.read_bytes() != data:
            raise ValueError("Refusing to overwrite an existing different model")
    else:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MODEL_PATH.open("xb") as stream:
            stream.write(data)
    return verify_ready(require_runtime=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-weights", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    args = parser.parse_args()
    result = install_weights() if args.install_weights else verify_ready(not args.weights_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
