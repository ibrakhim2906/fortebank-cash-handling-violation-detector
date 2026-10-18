#!/usr/bin/env bash
# Fetch one finished pod DEV30 run and score it locally with the frozen evaluator.
# Usage: tools/claude-recall/fetch_score.sh <run-name>   (e.g. recall-on-dev30)
set -euo pipefail
name=$1;repo=$(cd "$(dirname "$0")/../.." && pwd)
work=${CLAUDE_POD_WORK:?set CLAUDE_POD_WORK to a scratch directory}
mkdir -p "$work"
cat > "$work/export-$name.sh" <<EOF
set -euo pipefail
cd /workspace/claude-recall/data/processed
test "\$(ls claude-$name/*.json | wc -l)" -eq 30
python3 -c "import json;t=json.load(open('claude-runtime/$name-terminal.json'));assert t['returncode']==0 and t['records']==30,t"
echo MISSION_EXPORT_BEGIN
tar czf - -C claude-$name . | base64
echo MISSION_EXPORT_END
echo FREEZE_METADATA_BEGIN
cat claude-runtime/$name-freeze.json
echo
echo FREEZE_METADATA_END
cat claude-runtime/$name-terminal.json
EOF
python3 "$work/proxy_exec.py" "$work/export-$name.sh" --timeout 600 > "$work/export-$name.log"
cd "$repo"
PYTHONPATH="$repo" uv run python data/processed/runpod-setup-v1/score-dev-trial.py --log "$work/export-$name.log" --stem "mission-claude-$name" \
  --compare data/processed/mission-storage-veto-dev30-honest.json
