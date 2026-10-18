"""Audit local numeric action caches; report operational counts, never semantics."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.video_actions import DEFAULT_ACTION_DIR, DEFAULT_PROTOCOL, aggregate_summary
from src.video_eval import load_protocol
from src.video_train import engine_provenance, load_engine, verify_feature_record


def audit(run_dir, *, allow_partial=False, protocol_path=DEFAULT_PROTOCOL,
          action_dir=DEFAULT_ACTION_DIR):
    run = Path(run_dir).resolve()
    if run.is_relative_to(config.DATA_RAW.resolve()) or not run.is_dir():
        raise ValueError("Audit requires an existing derived run outside raw data")
    protocol = load_protocol(protocol_path)
    expected, holdout = set(protocol["development_group_ids"]), set(protocol["holdout_group_ids"])
    paths = sorted((run / "cache").glob("*.joblib"))
    cached = {path.stem for path in paths}
    if cached & holdout or cached - expected or any(path.is_symlink() for path in paths):
        raise ValueError("Cache contains reserved, unknown or symlinked records")
    if not allow_partial and cached != expected:
        raise ValueError("Complete audit requires every development cache record")
    manifest_path = run / "model_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("protocol_fingerprint") != protocol["fingerprint"]
                or set(manifest.get("training_groups", [])) != expected):
            raise ValueError("Run manifest differs from frozen development protocol")
    engine = load_engine("qwen-actions")
    provenance = engine_provenance("qwen-actions", engine)
    if manifest_path.exists():
        verify_feature_record(manifest, provenance, engine.FEATURE_CONFIG)
    clips = []
    for path in paths:
        item = joblib.load(path)  # Trusted locally produced numeric cache only.
        verify_feature_record(item, provenance, engine.FEATURE_CONFIG)
        if item.get("protocol_fingerprint", protocol["fingerprint"]) != protocol["fingerprint"]:
            raise ValueError("Action cache protocol fingerprint differs from the frozen protocol")
        clip = np.asarray(item["clip_features"], dtype=float)
        windows = item["windows"]
        values = np.asarray([window["features"] for window in windows], dtype=float)
        duration = float(item["duration_sec"])
        timestamps = np.asarray([window["timestamp_sec"] for window in windows], dtype=float)
        if (clip.shape != (len(engine.CLIP_FEATURE_NAMES),)
                or values.shape != (engine.WINDOW_COUNT, len(engine.WINDOW_FEATURE_NAMES))
                or not np.isfinite(clip).all() or not np.isfinite(values).all()
                or not np.isfinite(duration) or duration <= 0
                or not np.isfinite(timestamps).all() or not ((0 <= timestamps) & (timestamps <= duration)).all()):
            raise ValueError("Action cache dimensions, timing or finite-value checks failed")
        count = len(engine.ACTION_IDS)
        if (not np.isin(values, [0, 1]).all()
                or not np.all(values[:, :count] + values[:, count:2 * count] == 1)
                or not np.all(values[:, 2 * count:].sum(axis=1) == 1)):
            raise ValueError("Action cache states are invalid; unknowns cannot become zero-filled records")
        if np.all(values[:, -1] == 1):
            raise ValueError("All-unobservable footage must fail extraction, never create a cache")
        statistics = {"max": values.max(axis=0), "mean": values.mean(axis=0), "std": values.std(axis=0)}
        if not np.allclose(clip, np.concatenate([statistics[key] for key in engine.CLIP_STATISTICS]), atol=1e-7, rtol=1e-6):
            raise ValueError("Action clip summaries differ from their fixed window observations")
        clips.append(clip)
    matrix = np.stack(clips) if clips else np.empty((0, len(engine.CLIP_FEATURE_NAMES)))
    labels = aggregate_summary(action_dir, protocol_path)
    result = {"operational_integrity_only": True, "partial": cached != expected,
              "cached_groups": len(cached), "expected_development_groups": len(expected),
              "missing_development_groups": len(expected - cached), "holdout_groups_used": 0,
              "unique_clip_vectors": len(np.unique(matrix, axis=0)),
              "nonconstant_features": int(np.count_nonzero(np.ptp(matrix, axis=0))) if len(matrix) else 0,
              "all_unobservable_records": 0,
              "proposal_records": labels["proposal_records"],
              "human_reviewed_tasks": labels["human_reviewed_tasks"],
              "verified_action_observations": labels["verified_action_observations"],
              "verified_action_classes": labels["verified_action_classes"]}
    (run / "action_observation_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "video-qwen-actions")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit(args.run_dir, allow_partial=args.allow_partial)
    except Exception:
        raise SystemExit("Local numeric observation audit failed; no media or model text was emitted") from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
