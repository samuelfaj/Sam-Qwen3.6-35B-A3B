#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${LOCAL_DFLASH_MODEL_NAME:-qwen3.6-35b-a3b-dflash-local}"
OPEN_CODE_MODEL="localdflash/${MODEL_NAME}"
LOCAL_HOST="${LOCAL_DFLASH_HOST:-127.0.0.1}"
LOCAL_PORT="${LOCAL_DFLASH_PORT:-8010}"
HEALTH_URL="http://${LOCAL_HOST}:${LOCAL_PORT}/health"
LOCAL_DFLASH_AUTOSTART="${LOCAL_DFLASH_AUTOSTART:-1}"

ensure_local_server() {
  if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${LOCAL_DFLASH_AUTOSTART}" != "1" ]]; then
    echo "dflash: server not responding at ${HEALTH_URL}" >&2
    echo "dflash: run '${SCRIPT_DIR}/dflash.sh restart' first" >&2
    return 1
  fi

  echo "dflash: server unavailable, restarting wrapper in background..." >&2
  "${SCRIPT_DIR}/dflash.sh" restart >&2
}

ensure_local_server

run_watchdog() {
  exec python3 "${SCRIPT_DIR}/opencode_watchdog.py" --model "${OPEN_CODE_MODEL}" -- "$@"
}

if [[ "${1:-}" == "run-auto" || "${1:-}" == "run-supervised" ]]; then
  shift
  run_watchdog "$@"
fi

if [[ "${1:-}" == "run" && "${LOCAL_DFLASH_OPENCODE_AUTOPILOT:-0}" == "1" ]]; then
  shift
  run_watchdog "$@"
fi

if [[ "${1:-}" == "run" ]]; then
  shift
  exec opencode run -m "${OPEN_CODE_MODEL}" --dangerously-skip-permissions "$@"
fi

exec opencode --model "${OPEN_CODE_MODEL}" "$@"
