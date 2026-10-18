#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m scripts.fetch_video_objects --download
