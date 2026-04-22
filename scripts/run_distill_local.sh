#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${LOCAL_DFLASH_DISTILL_MODEL_NAME:-qwen3.6-35b-a3b-dflash-distill-local}"
LOCAL_HOST="${LOCAL_DFLASH_DISTILL_HOST:-127.0.0.1}"
LOCAL_PORT="${LOCAL_DFLASH_DISTILL_PORT:-8011}"
HEALTH_URL="http://${LOCAL_HOST}:${LOCAL_PORT}/health"
BASE_URL="http://${LOCAL_HOST}:${LOCAL_PORT}/v1"
LOCAL_DFLASH_DISTILL_AUTOSTART="${LOCAL_DFLASH_DISTILL_AUTOSTART:-1}"

ensure_local_server() {
  if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${LOCAL_DFLASH_DISTILL_AUTOSTART}" != "1" ]]; then
    echo "dflash-distill: server not responding at ${HEALTH_URL}" >&2
    echo "dflash-distill: run '${SCRIPT_DIR}/dflash_distill.sh restart' first" >&2
    return 1
  fi

  echo "dflash-distill: server unavailable, restarting wrapper in background..." >&2
  "${SCRIPT_DIR}/dflash_distill.sh" restart >&2
}

ensure_local_server

exec distill --provider dflash --model "${MODEL_NAME}" --host "${BASE_URL}" "$@"
