"""Derive temporal-engine cache metadata from verified local MobileNet records.

The descriptors are identical. This explicit conversion avoids re-decoding any
footage and never changes the earlier cache or reads reserved-holdout records.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.video_eval import load_protocol, protocol_groups
from src.video_train import engine_provenance, load_engine, verify_feature_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=config.DATA_PROC / "video-mobilenet/cache")
    parser.add_argument("--destination", type=Path,
                        default=config.DATA_PROC / "video-mobilenet-temporal/cache")
    parser.add_argument("--protocol", type=Path,
                        default=config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    args = parser.parse_args(argv)
    source, destination = args.source.resolve(), args.destination.resolve()
    if not destination.is_relative_to(config.DATA_PROC.resolve()) or destination == source:
        raise ValueError("Use a separate derived cache inside data/processed")
    protocol = load_protocol(args.protocol)
    base, temporal = load_engine("mobilenet"), load_engine("mobilenet-temporal")
    before = engine_provenance("mobilenet", base)
    after = engine_provenance("mobilenet-temporal", temporal)
    destination.mkdir(parents=True, exist_ok=True)
    groups = protocol_groups(protocol, "development")
    if set(g["group_id"] for g in groups) & set(protocol["holdout_group_ids"]):
        raise ValueError("Reserved holdout cannot enter a derived cache")
    for group in groups:
        output = destination / f"{group['group_id']}.joblib"
        if output.exists():
            raise ValueError("Derived cache already exists; choose a fresh destination")
        item = joblib.load(source / output.name)
        verify_feature_record(item, before, base.FEATURE_CONFIG)
        if (np.shape(item["clip_features"]) != (len(base.CLIP_FEATURE_NAMES),)
                or not item["windows"]
                or any(np.shape(w["features"]) != (len(base.WINDOW_FEATURE_NAMES),)
                       for w in item["windows"])):
            raise ValueError("Cached descriptors differ from the frozen MobileNet schema")
        converted = {**item, **after, "feature_config": temporal.FEATURE_CONFIG,
                     "derived_from_cache": str(source / output.name)}
        joblib.dump(converted, output)
    print(f"Derived {len(groups)} development cache records; holdout records read: 0")


if __name__ == "__main__":
    main()
