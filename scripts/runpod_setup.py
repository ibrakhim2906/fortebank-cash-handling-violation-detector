"""Prepare code bundles or install/serve approved pinned assets on one Runpod GPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import tarfile
import tempfile

from scripts.provision_action_model import ASSETS, Asset, digest, provision_asset, safe_path, verify_asset
from src import config


RUNTIME = "tools/local-action-model/runpod-b11707-cuda128"


def plan(root):
    value = json.loads((Path(root) / "deploy/runpod-assets.json").read_text())
    if value.get("schema_version") != 1 or value.get("runtime_release") != "b11707":
        raise ValueError("Unsupported GPU asset plan")
    return value


def assets(root, model):
    value = plan(root)
    return [Asset(**a) for a in value["runtime_assets"] + value["models"][model]["assets"]] + list(ASSETS[3:])


def gpu_preflight(model, runner=subprocess.run):
    if platform.machine() not in {"x86_64", "AMD64"}:
        raise ValueError("Pinned CUDA runtime requires x86_64")
    result = runner(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    rows = [line.split(",") for line in result.stdout.strip().splitlines() if line.strip()]
    minimum = 40000 if model == "30b" else 16000
    if len(rows) != 1 or len(rows[0]) != 3 or int(rows[0][1].strip()) < minimum:
        raise ValueError("Expected one GPU with sufficient VRAM for the selected model")
    return dict(zip(("name", "vram_mib", "driver"), (x.strip() for x in rows[0]), strict=True))


def tree_records(directory):
    directory = Path(directory).resolve()
    records = {}
    for path in sorted(directory.rglob("*")):
        if path == directory / "runtime-manifest.json":
            continue
        relative = path.relative_to(directory).as_posix()
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Runtime link escapes installation")
        if path.is_symlink():
            records[relative] = {"link": os.readlink(path)}
        elif path.is_file():
            records[relative] = {"sha256": digest(path), "executable": bool(path.stat().st_mode & 0o111)}
        elif not path.is_dir():
            raise ValueError("Unsupported runtime entry")
    return records


def unpack(archive_path, destination):
    """Extract only a hash-verified archive into a new, bounded private directory."""
    destination = Path(destination)
    if destination.exists():
        saved = json.loads((destination / "runtime-manifest.json").read_text())
        if saved["archive_sha256"] != digest(archive_path) or saved["files"] != tree_records(destination):
            raise ValueError("Existing GPU runtime differs; refusing overwrite")
        return saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".gpu-runtime-", dir=destination.parent))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            names = [str(PurePosixPath(m.name)) for m in members]
            if len(members) > 5000 or len(names) != len(set(names)) or sum(m.size for m in members) > 4 * 1024**3:
                raise ValueError("GPU archive exceeds extraction bounds or duplicates paths")
            for member in members:
                p = PurePosixPath(member.name)
                if p.is_absolute() or ".." in p.parts or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise ValueError("Unsafe GPU runtime archive member")
            archive.extractall(temporary, members=members, filter="data")
        for path in temporary.rglob("*"):
            if not path.resolve().is_relative_to(temporary.resolve()):
                raise ValueError("GPU runtime link escapes installation")
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() or path.stat().st_mode & 0o111 else 0o600)
        saved = {"archive_sha256": digest(archive_path), "files": tree_records(temporary)}
        manifest = temporary / "runtime-manifest.json"
        manifest.write_text(json.dumps(saved, sort_keys=True) + "\n"); manifest.chmod(0o600)
        if destination.exists():
            raise FileExistsError("GPU installation appeared concurrently")
        temporary.rename(destination)
        return saved
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def install(root, model):
    root = Path(root).resolve()
    hardware = gpu_preflight(model)  # fail before any download on a CPU/undersized machine
    selected = assets(root, model)
    required = sum(a.size for a in selected) + 6 * 1024**3
    if shutil.disk_usage(root).free < required:
        raise ValueError("Insufficient disk space for pinned models/runtime")
    for asset in selected:
        print("Preparing pinned asset:", Path(asset.path).name, flush=True)
        provision_asset(root, asset)
    for i, asset in enumerate(selected[:2]):
        unpack(safe_path(root, asset.path), safe_path(root, f"{RUNTIME}/runtime-{i}"))
    target = safe_path(root, f"{RUNTIME}/installation-{model}.json")
    value = {"schema_version": 1, "model": model, "hardware": hardware,
             "assets": [asdict(a) for a in selected],
             "asset_plan_sha256": digest(root / "deploy/runpod-assets.json")}
    if target.exists():
        old = json.loads(target.read_text())
        if old["assets"] != value["assets"] or old["asset_plan_sha256"] != value["asset_plan_sha256"]:
            raise ValueError("Existing GPU installation manifest differs")
    else:
        with target.open("x") as stream:
            json.dump(value, stream, indent=2); stream.write("\n")
        target.chmod(0o600)
    return verify(root, model)


def verify(root, model):
    root = Path(root).resolve()
    selected = assets(root, model)
    saved = json.loads(safe_path(root, f"{RUNTIME}/installation-{model}.json").read_text())
    if saved.get("model") != model or saved.get("assets") != [asdict(a) for a in selected] or saved.get("asset_plan_sha256") != digest(root / "deploy/runpod-assets.json"):
        raise ValueError("GPU installation does not match the approved asset plan")
    for asset in selected:
        if not verify_asset(root, asset):
            raise ValueError("Pinned GPU asset is missing")
    for i, asset in enumerate(selected[:2]):
        unpack(safe_path(root, asset.path), safe_path(root, f"{RUNTIME}/runtime-{i}"))
    runtime = safe_path(root, RUNTIME)
    servers = list(runtime.glob("runtime-*/**/llama-server"))
    if len(servers) != 1:
        raise ValueError("Expected exactly one pinned GPU model server")
    return {"model": model, "server": str(servers[0]),
            "model_path": str(safe_path(root, selected[2].path)),
            "projector_path": str(safe_path(root, selected[3].path)),
            "library_dirs": sorted({str(p.parent) for p in runtime.glob("runtime-*/**/*.so*")})}


def serve_command(info):
    return [info["server"], "-m", info["model_path"], "--mmproj", info["projector_path"],
            "--host", "127.0.0.1", "--port", "8003", "--n-gpu-layers", "999",
            "--ctx-size", "8192", "--parallel", "1", "--threads", "8", "--seed", str(config.SEED),
            "--image-max-tokens", "512", "--alias", "local-action-review", "--no-webui", "--log-disable"]


def serve(root, model):
    gpu_preflight(model)
    info = verify(root, model)
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "MTMD_", "HF_", "TRANSFORMERS_", "HUGGING_FACE_"))}
    environment.update(LD_LIBRARY_PATH=":".join(info["library_dirs"]), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LLAMA_LOG_DISABLE="1")
    check = subprocess.run([info["server"], "--list-devices"], env=environment, capture_output=True, text=True, check=True, timeout=30)
    if "CUDA" not in check.stdout + check.stderr:
        raise ValueError("Pinned runtime has no CUDA device; refusing a silent CPU fallback")
    print("Serving verified GPU observer on loopback8003; media/prompt logging disabled", flush=True)
    with open(os.devnull, "wb") as output:
        subprocess.run(serve_command(info), env=environment, cwd=root, stdout=output, stderr=output, check=True)


def bundle(root, output):
    """Explicit code-only allowlist; no data/models/tools/secrets/git state."""
    root = Path(root).resolve(); output = Path(output).resolve()
    if not output.is_relative_to((root / "data/processed").resolve()) or output.exists():
        raise ValueError("Bundle requires a new processed-data output path")
    chosen = [root / n for n in ("pyproject.toml", "uv.lock", ".python-version")]
    for directory, suffixes in [("src", {".py"}), ("scripts", {".py"}), ("app", {".py", ".html"})]:
        chosen.extend(p for p in (root / directory).rglob("*") if p.suffix in suffixes and "__pycache__" not in p.parts)
    chosen.extend(root / n for n in ("deploy/runpod-assets.json", "docs/runpod_setup.md", "docs/action_rule_candidates.md"))
    records = {}
    for path in chosen:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("Unexpected or linked code bundle source")
        records[path.relative_to(root).as_posix()] = digest(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for name in sorted(records):
            archive.add(root / name, arcname=name, recursive=False)
    output.chmod(0o600)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    with manifest.open("x") as stream:
        json.dump({"files": records, "archive_sha256": digest(output), "contains_bank_data": False,
                   "contains_credentials": False}, stream, indent=2); stream.write("\n")
    manifest.chmod(0o600)
    return {"bundle": str(output), "file_count": len(records), "bytes": output.stat().st_size,
            "sha256": digest(output), "contains_bank_data": False, "contains_credentials": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--install", action="store_true")
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--bundle", type=Path)
    parser.add_argument("--model", choices=("30b", "8b"), default="30b")
    args = parser.parse_args(); root = config.ROOT
    if args.bundle:
        result = bundle(root, args.bundle)
    elif args.install:
        result = install(root, args.model)
    elif args.serve:
        serve(root, args.model); return
    elif args.verify:
        result = verify(root, args.model)
    else:
        result = {"mode": "inspect_only", "downloads": False, "allocates_or_bills_provider": False,
                  "configuration": plan(root), "selected_assets": [asdict(a) for a in assets(root, args.model)]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
