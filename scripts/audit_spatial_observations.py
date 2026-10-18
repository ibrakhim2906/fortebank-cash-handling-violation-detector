"""Audit private v3 observations and sampling; emit only aggregate counts."""
import argparse
import json
from pathlib import Path
import re

import joblib
import numpy as np

from src import config
from src.video_eval import load_protocol
from src.video_train import engine_provenance, load_engine, verify_feature_record


def audit(run_dir, protocol_path=None):
    run = Path(run_dir).resolve()
    if run.is_relative_to(config.DATA_RAW.resolve()):
        raise ValueError("Spatial audit requires derived data")
    protocol = load_protocol(protocol_path or config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    paths = list((run / "cache").glob("*.joblib"))
    expected = set(protocol["development_group_ids"])
    if {path.stem for path in paths} != expected or any(path.is_symlink() for path in paths):
        raise ValueError("Spatial audit requires every permitted development record and no others")
    engine = load_engine("qwen-spatial-v3")
    provenance = engine_provenance("qwen-spatial-v3", engine)
    clips, windows, elapsed = [], [], []
    for path in paths:
        record = joblib.load(path)  # Trusted operator-produced files only.
        verify_feature_record(record, provenance, engine.FEATURE_CONFIG)
        if (not np.isfinite(record["extraction_sec"]) or record["extraction_sec"] < 0
                or not np.isfinite(record["duration_sec"]) or not 0 < record["duration_sec"] <= 300):
            raise ValueError("Spatial extraction duration or runtime is invalid")
        source_fps = float(record["sampling"]["source_fps"])
        observations = record["observation_audit"]
        if len(observations) != 4 or len(record["windows"]) != 4:
            raise ValueError("Spatial audit found incomplete sampling")
        frame_times, frame_indices, values = [], [], []
        for observation, window in zip(observations, record["windows"]):
            expected_vector, _ = engine.observation_features(observation["observation"])
            if not np.array_equal(expected_vector, np.asarray(window["features"])):
                raise ValueError("Private observation and classifier feature values differ")
            actual, wanted = observation["actual_sec"], observation["requested_sec"]
            if (not np.isfinite([actual, wanted, source_fps]).all() or source_fps <= 0
                    or not 0 <= actual < record["duration_sec"]
                    or abs(actual - wanted) > max(.25, 2 / source_fps)
                    or actual != window["timestamp_sec"]):
                raise ValueError("Spatial sample timing differs from the frozen recipe")
            index = observation["source_frame_index"]
            if (isinstance(index, bool) or not isinstance(index, (int, np.integer))
                    or not 0 <= index < round(record["duration_sec"] * source_fps)):
                raise ValueError("Spatial source frame index is invalid")
            frame_times.append(actual); frame_indices.append(observation["source_frame_index"])
            views, shape = observation["views"], observation["source_shape"]
            if (not isinstance(shape, (list, tuple)) or len(shape) != 3 or shape[2] != 3
                    or any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
                           or value <= 0 for value in shape)):
                raise ValueError("Spatial source dimensions are invalid")
            if len(views) != 3:
                raise ValueError("Spatial sample is missing a source-aligned view")
            height, width = shape[:2]
            side = min(height, width)
            wanted_rects = [[0, 0, width, height], [0, 0, side, side],
                            [0, height - side, side, height] if height >= width
                            else [width - side, 0, width, side]]
            if [view["crop_rect_xyxy"] for view in views] != wanted_rects:
                raise ValueError("Spatial crop metadata differs from label-independent dimensions")
            if (not re.fullmatch(r"[0-9a-f]{64}", str(observation["source_frame_sha256"]))
                    or any(not re.fullmatch(r"[0-9a-f]{64}", str(view["sha256"])) for view in views)):
                raise ValueError("Spatial view hash provenance is incomplete")
            values.append(expected_vector)
        if not (np.diff(frame_times) > 0).all() or not (np.diff(frame_indices) > 0).all():
            raise ValueError("Spatial samples are not strictly chronological")
        matrix = np.stack(values)
        summaries = {"max": matrix.max(axis=0), "mean": matrix.mean(axis=0), "std": matrix.std(axis=0)}
        clip = np.asarray(record["clip_features"], dtype=float)
        if (clip.shape != (84,) or not np.isfinite(clip).all()
                or not np.allclose(clip, np.concatenate([summaries[key] for key in engine.CLIP_STATISTICS]))):
            raise ValueError("Spatial clip pooling differs from private sampled observations")
        clips.append(clip); windows.extend(values); elapsed.append(record["extraction_sec"])
    matrix = np.stack(clips)
    result = {"cached_development_groups": len(clips), "holdout_groups_used": 0,
              "sampled_instants": len(windows), "sampled_views": len(windows) * 3,
              "unique_clip_vectors": len(np.unique(matrix, axis=0)),
              "nonconstant_features": int(np.count_nonzero(np.ptp(matrix, axis=0))),
              "maximum_extraction_sec": float(max(elapsed)),
              "sampling_alignment_verified": True, "semantic_accuracy_verified": False,
              "operational_integrity_only": True}
    (run / "spatial_observation_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "video-qwen-spatial-v3")
    args = parser.parse_args(argv)
    try:
        result = audit(args.run_dir)
    except Exception:
        raise SystemExit("Private spatial audit failed; no media or observations emitted") from None
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
