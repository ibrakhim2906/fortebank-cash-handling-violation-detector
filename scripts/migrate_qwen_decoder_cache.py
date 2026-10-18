"""Preserve successful descriptors across the reviewed failed-seek-only repair.

This is a narrow one-time compatibility migration, not a general stale-cache
bypass. Old source snapshots, unchanged model/prompt/config/code, development
membership and source content hashes are checked before copying any record.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.video_eval import load_protocol, protocol_groups
from src.video_train import digest, engine_provenance, verify_feature_record


OLD_ENGINE_SHA256 = "c2a064c3a4cb5f2b1ab71f158c53b6a3d244c9789d00e82e43d86ec9e26bc507"
OLD_PROPOSAL_SHA256 = "434a15b14e3f816bbe0af7c7281103bce808834e7d03fd1886940883745b2f00"
POLICY = "frame_index_then_sequential_from_start"


def unchanged_ast(before, after, allowed_functions, *, config_assignment=False):
    def signature(source):
        module = ast.parse(source)
        kept = []
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in allowed_functions:
                continue
            if config_assignment and isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "FEATURE_CONFIG" for target in node.targets):
                continue
            kept.append(ast.dump(node, include_attributes=False))
        return kept
    if signature(before) != signature(after):
        raise ValueError("Code outside the reviewed decoder repair changed; migration refused")


def checked_configs(old, new):
    before, after = copy.deepcopy(old), copy.deepcopy(new)
    if after.pop("failed_seek_fallback", None) != POLICY or "failed_seek_fallback" in before:
        raise ValueError("Migration applies only to the declared failed-seek repair")
    old_helpers, new_helpers = before.pop("inherited_sources"), after.pop("inherited_sources")
    helper = "src/video_action_proposals.py"
    if (set(old_helpers) != set(new_helpers)
            or old_helpers.get(helper) != OLD_PROPOSAL_SHA256
            or {k: v for k, v in old_helpers.items() if k != helper}
            != {k: v for k, v in new_helpers.items() if k != helper}
            or before != after):
        raise ValueError("Model, prompt, sampling, features or unrelated helpers changed; migration refused")
    return old_helpers


def main(argv=None):
    from src import video_qwen_actions as engine
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=config.DATA_PROC / "video-qwen-actions")
    parser.add_argument("--destination-run", type=Path, default=config.DATA_PROC / "video-qwen-actions-v2")
    parser.add_argument("--extra-cache", type=Path, default=config.DATA_PROC / "video-qwen-benchmark/cache")
    args = parser.parse_args(argv)
    source, destination = args.source_run.resolve(), args.destination_run.resolve()
    if (source == destination or not destination.is_relative_to(config.DATA_PROC.resolve())
            or destination.exists()):
        raise ValueError("Use a fresh destination inside data/processed")
    old_source = source / "source_snapshot.py"
    old_proposals = source / "proposal_source_snapshot.py"
    if digest(old_source) != OLD_ENGINE_SHA256 or digest(old_proposals) != OLD_PROPOSAL_SHA256:
        raise ValueError("Old source snapshots differ from the interrupted experiment")
    unchanged_ast(old_source.read_text(), Path(engine.__file__).read_text(),
                  {"_read_frame_at", "_sample_interval"}, config_assignment=True)
    unchanged_ast(old_proposals.read_text(), (config.ROOT / "src/video_action_proposals.py").read_text(),
                  {"sample_task"})
    old_config = json.loads((source / "feature_config_snapshot.json").read_text())
    old_helpers = checked_configs(old_config, engine.FEATURE_CONFIG)
    provenance = engine_provenance("qwen-actions", engine)
    old_provenance = {**provenance, "source_sha256": OLD_ENGINE_SHA256,
                      "inherited_sources": old_helpers}
    protocol = load_protocol(config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    groups = {g["group_id"]: g for g in protocol_groups(protocol, "development")}
    records = {}
    for folder in (source / "cache", args.extra_cache):
        for path in sorted(folder.glob("*.joblib")):
            if path.stem not in groups:
                raise ValueError("Non-development cache member; migration refused before loading")
            item = joblib.load(path)
            verify_feature_record(item, old_provenance, old_config)
            if digest(config.ROOT / groups[path.stem]["path"]) != path.stem:
                raise ValueError("Development source content changed")
            if path.stem in records:
                raise ValueError("Duplicate migration input; inspect manually")
            records[path.stem] = (path, item)
    if not records:
        raise ValueError("No successful old records available")
    cache = destination / "cache"
    cache.mkdir(parents=True)
    for identifier, (path, item) in records.items():
        converted = {**item, **provenance, "feature_config": engine.FEATURE_CONFIG,
                     "derived_from_cache": str(path),
                     "decoder_compatibility": {"old_source_sha256": OLD_ENGINE_SHA256,
                                               "reason": "Successful original seeks use the unchanged path"}}
        output = cache / f"{identifier}.joblib"
        joblib.dump(converted, output)
        loaded = joblib.load(output)
        if (not np.array_equal(loaded["clip_features"], item["clip_features"])
                or len(loaded["windows"]) != len(item["windows"])
                or any(a["timestamp_sec"] != b["timestamp_sec"] or not np.array_equal(a["features"], b["features"])
                       for a, b in zip(loaded["windows"], item["windows"]))):
            raise ValueError("Descriptor bytes changed during metadata-only migration")
    summary = {"migrated_development_records": len(records), "holdout_records": 0,
               "source_snapshot_sha256": OLD_ENGINE_SHA256,
               "new_source_sha256": provenance["source_sha256"], "feature_values_changed": False}
    (destination / "decoder_migration.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
