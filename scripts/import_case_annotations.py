"""Import approved human development notes; print only aggregate integrity counts."""
import argparse
import json
from pathlib import Path

from src.video_case_annotations import (
    DEFAULT_ARCHIVE, DEFAULT_OUTPUT, DEFAULT_PROTOCOL, import_annotations,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args(argv)
    result = import_annotations(args.archive, args.protocol, args.out_dir, args.version)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
