#!/usr/bin/env bash
# Existing approved A40 only. Does not provision compute or expose the model.
set -euo pipefail
cd /workspace/datathon-kit
export UV_NO_PROGRESS=1 UV_CACHE_DIR=/tmp/mission-uv-cache UV_LINK_MODE=hardlink
mission_profile="${AUDITOR_PROFILE:-qwen3-8b}"
if [ -z "${AUDITOR_PROFILE:-}" ] && [ -f data/processed/mission-runtime/active-profile.txt ]; then
  IFS= read -r mission_profile < data/processed/mission-runtime/active-profile.txt
fi
if [ "${1:-prepare}" = prepare ] && [ -n "${2:-}" ]; then mission_profile="$2";fi
mission_quantization=();mission_dtype=bfloat16
case "$mission_profile" in
qwen3-8b) mission_model_dir=/workspace/models/qwen3-vl-8b;mission_manifest=deploy/qwen3-vl-8b.json;;
qwen25-7b) mission_model_dir=/workspace/models/qwen25-vl-7b;mission_manifest=deploy/qwen25-vl-7b.json;;
qwen3-32b-awq) mission_model_dir=/workspace/models/qwen3-vl-32b-awq;mission_manifest=deploy/qwen3-vl-32b-awq.json;mission_quantization=(--quantization awq_marlin);mission_dtype=float16;;
*) printf 'Unknown pinned auditor profile\n' >&2;exit 2;;
esac
case "${1:-prepare}" in
prepare)
  if [ ! -f /opt/vllm/.mission-ready ]; then
    if [ ! -x /opt/vllm/bin/python ]; then uv venv --python /workspace/datathon-kit/.venv/bin/python /opt/vllm; fi
    uv pip install --python /opt/vllm/bin/python 'vllm==0.11.0' 'transformers==4.57.1'
    uv pip freeze --python /opt/vllm/bin/python > /opt/vllm/installed.txt
    /opt/vllm/bin/python -c 'import vllm,transformers; assert vllm.__version__=="0.11.0"; assert transformers.__version__=="4.57.1"'
    touch /opt/vllm/.mission-ready
    uv cache clean
  fi
  uv run --frozen --no-dev python -m scripts.download_auditor --profile "$mission_profile"
  ;;
model)
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  mission_seed="$(uv run --frozen --no-dev python -c 'from src.config import SEED; print(SEED)')"
  exec /opt/vllm/bin/vllm serve "$mission_model_dir" \
    --served-model-name qwen-vl --host 127.0.0.1 --port 8002 --dtype "$mission_dtype" "${mission_quantization[@]}" \
    --seed "$mission_seed" \
    --max-model-len 16384 --limit-mm-per-prompt '{"image":16,"video":0}' \
    --gpu-memory-utilization 0.86 --max-num-seqs 1 --enforce-eager --disable-log-requests \
    --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":262144}'
  ;;
service)
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AUDITOR=on VLLM_URL=http://127.0.0.1:8002 AUDITOR_CALL_TIMEOUT=30
  export AUDITOR_PROFILE="$mission_profile" AUDITOR_MANIFEST="$mission_manifest"
  export API_KEY="$(cat data/processed/runpod-service-v1/api-key.txt)"
  export JOBS_DIR=/workspace/datathon-kit/data/processed/mission-service/jobs
  exec uv run --frozen --no-dev uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers 1 --no-access-log
  ;;
supervise)
  mission_target="${2:?model or service}";mission_child="${3:-}"
  case "$mission_target" in model|service) ;; *) exit 2;; esac
  trap 'if [ -n "$mission_child" ]; then kill -TERM "$mission_child" 2>/dev/null || true; fi; exit 0' TERM INT
  # Optionally adopt the already healthy owned model process, avoiding a load.
  while [ -n "$mission_child" ] && kill -0 "$mission_child" 2>/dev/null; do sleep 2; done
  while true; do
    if [ -f "data/processed/mission-runtime/pause-$mission_target" ]; then sleep 2;continue;fi
    bash "$0" "$mission_target" & mission_child=$!
    printf '%s\n' "$mission_child" > "data/processed/mission-runtime/$mission_target-child.pid"
    wait "$mission_child" || true
    mission_child='';sleep 2
  done
  ;;
*) exit 2;;
esac
