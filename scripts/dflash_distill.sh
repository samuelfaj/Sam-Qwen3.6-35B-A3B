#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${REPO_ROOT}/.dflash-distill.pid"
LOG_FILE="${REPO_ROOT}/dflash-distill.log"
PORT="${LOCAL_DFLASH_DISTILL_PORT:-8011}"
HOST="${LOCAL_DFLASH_DISTILL_HOST:-127.0.0.1}"
HEALTH_URL="http://${HOST}:${PORT}/health"

get_running_pid() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "${pid}"
      return 0
    fi
    rm -f "${PID_FILE}"
  fi
  { lsof -ti:"${PORT}" -sTCP:LISTEN 2>/dev/null || true; } | head -1 || true
}

kill_tree() {
  local pid="$1"
  local sig="$2"
  local pgid
  pgid="$(ps -o pgid= "${pid}" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "${pgid}" ]]; then
    kill -"${sig}" -"${pgid}" 2>/dev/null || kill -"${sig}" "${pid}" 2>/dev/null || true
  else
    kill -"${sig}" "${pid}" 2>/dev/null || true
  fi
}

cmd_status() {
  local pid
  pid="$(get_running_pid)"
  if [[ -z "${pid}" ]]; then
    echo "dflash-distill: stopped"
    return 1
  fi
  echo "dflash-distill: running (pid ${pid}, port ${PORT})"
  if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "health: ok (${HEALTH_URL})"
    return 0
  fi
  echo "health: NOT responding at ${HEALTH_URL}"
  return 2
}

cmd_start() {
  local pid
  pid="$(get_running_pid)"
  if [[ -n "${pid}" ]]; then
    echo "dflash-distill: already running (pid ${pid})"
    return 0
  fi
  echo "dflash-distill: starting, logging to ${LOG_FILE}"
  nohup "${SCRIPT_DIR}/start_distill_wrapper_supervised.sh" "$@" >"${LOG_FILE}" 2>&1 &
  local new_pid=$!
  echo "${new_pid}" > "${PID_FILE}"
  disown "${new_pid}" 2>/dev/null || true
  echo "dflash-distill: pid ${new_pid}"
  local waited=0
  while (( waited < 120 )); do
    if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
      echo "dflash-distill: ready (${HEALTH_URL})"
      return 0
    fi
    if ! kill -0 "${new_pid}" 2>/dev/null; then
      echo "dflash-distill: process died during startup; see ${LOG_FILE}"
      rm -f "${PID_FILE}"
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "dflash-distill: started but /health not responding after 120s; check ${LOG_FILE}"
  return 1
}

cmd_stop() {
  local pid
  pid="$(get_running_pid)"
  if [[ -z "${pid}" ]]; then
    echo "dflash-distill: not running"
    rm -f "${PID_FILE}"
    return 0
  fi
  echo "dflash-distill: stopping pid ${pid} (SIGTERM)"
  kill_tree "${pid}" TERM
  local waited=0
  while (( waited < 30 )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${PID_FILE}"
      echo "dflash-distill: stopped"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "dflash-distill: still alive after 30s; use '$0 kill' to force"
  return 1
}

cmd_kill() {
  local pid
  pid="$(get_running_pid)"
  if [[ -z "${pid}" ]]; then
    echo "dflash-distill: not running"
    rm -f "${PID_FILE}"
    return 0
  fi
  echo "dflash-distill: force-killing pid ${pid} (SIGKILL)"
  kill_tree "${pid}" KILL
  rm -f "${PID_FILE}"
  local stragglers
  stragglers="$(lsof -ti:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${stragglers}" ]]; then
    echo "dflash-distill: killing port squatters: ${stragglers}"
    # shellcheck disable=SC2086
    kill -KILL ${stragglers} 2>/dev/null || true
  fi
  echo "dflash-distill: killed"
}

cmd_restart() {
  cmd_stop || true
  cmd_start "$@"
}

cmd_logs() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "dflash-distill: no log file at ${LOG_FILE}"
    return 1
  fi
  exec tail -f "${LOG_FILE}"
}

warn_if_not_ready() {
  if ! curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "dflash-distill: server not responding at ${HEALTH_URL}" >&2
    echo "dflash-distill: run '$0 start' first (or '$0 status' to inspect)" >&2
  fi
}

cmd_distill() {
  warn_if_not_ready
  exec "${SCRIPT_DIR}/run_distill_local.sh" "$@"
}

usage() {
  cat <<EOF
usage: $(basename "$0") <command> [args...]

server commands:
  start     start the stateless distill server in background, wait for /health (up to 120s)
  stop      graceful SIGTERM, wait up to 30s
  restart   stop then start
  status    check running process and /health
  kill      SIGKILL the process group, clear port squatters
  logs      tail -f ${LOG_FILE}

client commands:
  distill   launch distill against the stateless local server (forwards extra args)

env:
  LOCAL_DFLASH_DISTILL_HOST=${HOST}
  LOCAL_DFLASH_DISTILL_PORT=${PORT}

paths:
  pid file: ${PID_FILE}
  log file: ${LOG_FILE}
EOF
}

main() {
  local cmd="${1:-}"
  shift || true
  case "${cmd}" in
    start)             cmd_start "$@" ;;
    stop)              cmd_stop ;;
    restart)           cmd_restart "$@" ;;
    status)            cmd_status ;;
    kill)              cmd_kill ;;
    logs)              cmd_logs ;;
    distill)           cmd_distill "$@" ;;
    ""|-h|--help|help) usage ;;
    *) echo "unknown command: ${cmd}" >&2; usage >&2; exit 2 ;;
  esac
}

main "$@"
