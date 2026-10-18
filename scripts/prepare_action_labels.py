"""Prepare development-only action review metadata; print aggregate counts only."""
import argparse
import json
from pathlib import Path

from src.video_actions import (
    DEFAULT_ACTION_DIR, DEFAULT_PROTOCOL, aggregate_summary, prepare_queue,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ACTION_DIR)
    args = parser.parse_args(argv)
    prepare_queue(args.protocol, args.out_dir)
    print(json.dumps(aggregate_summary(args.out_dir, args.protocol), sort_keys=True))


if __name__ == "__main__":
    main()
