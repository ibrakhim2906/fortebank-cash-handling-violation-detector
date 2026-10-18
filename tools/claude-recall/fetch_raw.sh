#!/usr/bin/env bash
# Copy one finished pod run's raw JSON records into data/processed/mission-claude-<run>/.
# Usage: CLAUDE_POD_WORK=<scratch> tools/claude-recall/fetch_raw.sh <run-name>
set -euo pipefail
name=$1;repo=$(cd "$(dirname "$0")/../.." && pwd);work=${CLAUDE_POD_WORK:?}
cat > "$work/raw-$name.sh" <<EOF
set -euo pipefail
cd /workspace/claude-recall/data/processed
python3 -c "import json;t=json.load(open('claude-runtime/$name-terminal.json'));assert t['returncode']==0,t;print('TERMINAL',json.dumps(t))"
echo RAW_BEGIN; tar czf - -C claude-$name . | base64; echo RAW_END
EOF
python3 "$work/proxy_exec.py" "$work/raw-$name.sh" --timeout 600 > "$work/raw-$name.log"
grep '^TERMINAL' "$work/raw-$name.log"
dest="$repo/data/processed/mission-claude-$name";mkdir -p "$dest"
sed -n '/^RAW_BEGIN/,/^RAW_END/p' "$work/raw-$name.log" | sed '1d;$d' | tr -d '\r' | base64 -d | tar xzf - -C "$dest"
ls "$dest"/*.json | wc -l
