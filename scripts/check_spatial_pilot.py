"""Label-blind operational pilot for the fixed local spatial observation recipe.

Only aggregate diagnostics leave this module. Local observations and sample
provenance stay in development-only derived caches, never verified labels.
"""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np

from src import config
from src.video_eval import load_protocol, protocol_groups
from src.video_train import digest, engine_provenance, load_engine, verify_feature_record


ENCODER = "qwen-spatial-v3"
PILOT_SIZE = 3
OBJECT_FIELDS = ("hand_object_contact", "rectangular_sheets_or_stacks",
                 "handheld_device", "open_container", "work_surface")


def assess(records, fields):
    """Evaluate only operational coverage/variation, never organizer targets."""
    windows = [observation for item in records for observation in item["observation_audit"]]
    varying_definite_fields = sum(
        len({sample["observation"][field] for sample in windows
             if sample["observation"][field] not in {"unclear", "unobservable"}}) >= 2
        for field in fields if field != "visibility"
    )
    definite = sum(sample["observation"][field] != "unclear"
                   for sample in windows for field in OBJECT_FIELDS)
    coverage = definite / (len(windows) * len(OBJECT_FIELDS)) if windows else 0.0
    vectors = np.stack([item["clip_features"] for item in records])
    unique = len(np.unique(vectors, axis=0))
    longest = max(item["extraction_sec"] for item in records)
    unobservable_clips = sum(all(sample["observation"]["visibility"] == "unobservable"
                                 for sample in item["observation_audit"]) for item in records)
    passed = (len(records) == PILOT_SIZE and len(windows) == PILOT_SIZE * 4
              and unique >= 2 and varying_definite_fields >= 2 and coverage >= 0.25
              and longest < 180 and not unobservable_clips)
    return {"passed": bool(passed), "pilot_groups": len(records), "valid_windows": len(windows),
            "unique_clip_vectors": unique, "varying_definite_fields": varying_definite_fields,
            "definite_object_fraction": float(coverage), "maximum_extraction_sec": float(longest),
            "all_unobservable_clips": int(unobservable_clips), "holdout_groups_used": 0,
            "binary_labels_used": False, "semantic_accuracy_verified": False}


def ensure_pilot(protocol, run_dir):
    run = Path(run_dir).resolve()
    if run.is_relative_to(config.DATA_RAW.resolve()):
        raise ValueError("Spatial pilot cannot write raw data")
    engine = load_engine(ENCODER)
    provenance = engine_provenance(ENCODER, engine)
    chosen = sorted(protocol_groups(protocol, "development"), key=lambda item: item["group_id"])[:PILOT_SIZE]
    if len(chosen) != PILOT_SIZE or any(item["group_id"] in protocol["holdout_group_ids"] for item in chosen):
        raise ValueError("Spatial pilot requires three permitted development recordings")
    cache = run / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for index, group in enumerate(chosen):
        source = config.ROOT / group["path"]
        if digest(source) != group["group_id"]:
            raise ValueError("Pilot source differs from frozen protocol")
        target = cache / (group["group_id"] + ".joblib")
        if target.exists():
            record = joblib.load(target)
            verify_feature_record(record, provenance, engine.FEATURE_CONFIG)
        else:
            started = time.monotonic()
            result = engine.extract_video(source, deadline=started + 240)
            record = {**provenance, "feature_config": engine.FEATURE_CONFIG,
                      "clip_features": np.asarray(result["clip_features"], dtype=float),
                      "windows": result["windows"], "duration_sec": result["duration_sec"],
                      "extraction_sec": time.monotonic() - started,
                      "observation_audit": result["observation_audit"], "sampling": result["sampling"]}
            joblib.dump(record, target)
        records.append(record)
        print(f"Spatial pilot completed development sample {index + 1}/{PILOT_SIZE}", flush=True)
    report = assess(records, engine.FIELDS)
    (run / "spatial_pilot.json").write_text(json.dumps({
        **report, "protocol_fingerprint": protocol["fingerprint"], **provenance,
        "feature_config": engine.FEATURE_CONFIG,
        "selection": "First three development group hashes sorted lexicographically; no targets or anchors",
    }, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "video-qwen-spatial-v3")
    parser.add_argument("--protocol", type=Path, default=config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    args = parser.parse_args(argv)
    report = ensure_pilot(load_protocol(args.protocol), args.run_dir)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
