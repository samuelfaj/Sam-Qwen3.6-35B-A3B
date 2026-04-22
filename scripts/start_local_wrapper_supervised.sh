#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESTART_DELAY_SECONDS="${LOCAL_DFLASH_RESTART_DELAY_SECONDS:-2}"
STOP_REQUESTED=0
CHILD_PID=""

forward_stop() {
  STOP_REQUESTED=1
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
  fi
}

trap forward_stop INT TERM HUP

while true; do
  "${SCRIPT_DIR}/start_local_wrapper.sh" "$@" &
  CHILD_PID=$!

  set +e
  wait "${CHILD_PID}"
  EXIT_CODE=$?
  set -e
  CHILD_PID=""

  if (( STOP_REQUESTED )); then
    exit 0
  fi

  printf '%s dflash-supervisor: wrapper exited with code %s; restarting in %ss\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "${EXIT_CODE}" \
    "${RESTART_DELAY_SECONDS}" \
    >&2

  sleep "${RESTART_DELAY_SECONDS}"
done
