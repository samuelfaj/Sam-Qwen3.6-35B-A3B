#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${REPO_ROOT}/.dflash.pid"
LOG_FILE="${REPO_ROOT}/dflash.log"
PORT="${LOCAL_DFLASH_PORT:-8010}"
HOST="${LOCAL_DFLASH_HOST:-127.0.0.1}"
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
    echo "dflash: stopped"
    return 1
  fi
  echo "dflash: running (pid ${pid}, port ${PORT})"
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
    echo "dflash: already running (pid ${pid})"
    return 0
  fi
  echo "dflash: starting, logging to ${LOG_FILE}"
  local caffeinate_wrap=()
  if [[ "${DFLASH_NO_CAFFEINATE:-0}" != "1" ]] && command -v caffeinate >/dev/null 2>&1; then
    # -d prevent display sleep, -i prevent idle sleep, -m prevent disk sleep,
    # -s prevent system sleep on AC, -u declare user activity.
    caffeinate_wrap=(caffeinate -dimsu)
  fi
  nohup "${caffeinate_wrap[@]}" "${SCRIPT_DIR}/start_local_wrapper_supervised.sh" "$@" >"${LOG_FILE}" 2>&1 &
  local new_pid=$!
  echo "${new_pid}" > "${PID_FILE}"
  disown "${new_pid}" 2>/dev/null || true
  echo "dflash: pid ${new_pid}"
  local waited=0
  while (( waited < 120 )); do
    if curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
      echo "dflash: ready (${HEALTH_URL})"
      return 0
    fi
    if ! kill -0 "${new_pid}" 2>/dev/null; then
      echo "dflash: process died during startup; see ${LOG_FILE}"
      rm -f "${PID_FILE}"
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "dflash: started but /health not responding after 120s; check ${LOG_FILE}"
  return 1
}

cmd_stop() {
  local pid
  pid="$(get_running_pid)"
  if [[ -z "${pid}" ]]; then
    echo "dflash: not running"
    rm -f "${PID_FILE}"
    return 0
  fi
  echo "dflash: stopping pid ${pid} (SIGTERM)"
  kill_tree "${pid}" TERM
  local waited=0
  while (( waited < 30 )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${PID_FILE}"
      echo "dflash: stopped"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "dflash: still alive after 30s; use '$0 kill' to force"
  return 1
}

cmd_kill() {
  local pid
  pid="$(get_running_pid)"
  if [[ -z "${pid}" ]]; then
    echo "dflash: not running"
    rm -f "${PID_FILE}"
    return 0
  fi
  echo "dflash: force-killing pid ${pid} (SIGKILL)"
  kill_tree "${pid}" KILL
  rm -f "${PID_FILE}"
  local stragglers
  stragglers="$(lsof -ti:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${stragglers}" ]]; then
    echo "dflash: killing port squatters: ${stragglers}"
    # shellcheck disable=SC2086
    kill -KILL ${stragglers} 2>/dev/null || true
  fi
  echo "dflash: killed"
}

cmd_restart() {
  cmd_stop || true
  cmd_start "$@"
}

cmd_logs() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "dflash: no log file at ${LOG_FILE}"
    return 1
  fi
  exec tail -f "${LOG_FILE}"
}

warn_if_not_ready() {
  if ! curl -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "dflash: server not responding at ${HEALTH_URL}" >&2
    echo "dflash: run '$0 start' first (or '$0 status' to inspect)" >&2
  fi
}

cmd_opencode() {
  warn_if_not_ready
  exec "${SCRIPT_DIR}/run_opencode_local.sh" "$@"
}

cmd_opencode_auto() {
  warn_if_not_ready
  exec "${SCRIPT_DIR}/run_opencode_local.sh" run-auto "$@"
}

cmd_codex() {
  warn_if_not_ready
  exec "${SCRIPT_DIR}/run_codex_local.sh" "$@"
}

cmd_queue() {
  warn_if_not_ready
  exec python3 "${SCRIPT_DIR}/agent_queue.py" run "$@"
}

cmd_queue_resume() {
  warn_if_not_ready
  exec python3 "${SCRIPT_DIR}/agent_queue.py" resume "$@"
}

cmd_queue_status() {
  exec python3 "${SCRIPT_DIR}/agent_queue.py" status "$@"
}

cmd_queue_plan() {
  warn_if_not_ready
  exec python3 "${SCRIPT_DIR}/agent_queue.py" plan "$@"
}

cmd_install_launchd() {
  local plist_dir="${HOME}/Library/LaunchAgents"
  local plist_path="${plist_dir}/dev.dflash.plist"
  mkdir -p "${plist_dir}"
  cat > "${plist_path}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>dev.dflash</string>
    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>-lc</string>
      <string>exec ${REPO_ROOT}/scripts/dflash.sh start</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${REPO_ROOT}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
      <key>SuccessfulExit</key>
      <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>${LOG_FILE}</string>
    <key>StandardErrorPath</key>
    <string>${LOG_FILE}</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>PATH</key>
      <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
  </dict>
</plist>
EOF
  launchctl unload "${plist_path}" 2>/dev/null || true
  launchctl load -w "${plist_path}"
  echo "dflash: launchd agent installed at ${plist_path}"
  echo "        manage via: launchctl {load,unload,start,stop} ${plist_path}"
}

cmd_uninstall_launchd() {
  local plist_path="${HOME}/Library/LaunchAgents/dev.dflash.plist"
  if [[ ! -f "${plist_path}" ]]; then
    echo "dflash: no launchd agent installed at ${plist_path}"
    return 0
  fi
  launchctl unload -w "${plist_path}" 2>/dev/null || true
  rm -f "${plist_path}"
  echo "dflash: launchd agent removed (${plist_path})"
}

cmd_thermal() {
  # Prints current macOS thermal state. Exits non-zero if the CPU is being throttled
  # (CPU_Speed_Limit < 90). Useful for autonomy loops that should back off when hot.
  if ! command -v pmset >/dev/null 2>&1; then
    echo "dflash: pmset not available; cannot read thermal state" >&2
    return 2
  fi
  local therm_output
  therm_output="$(pmset -g therm 2>&1)"
  echo "${therm_output}"
  local limit
  limit="$(echo "${therm_output}" | awk -F'=' '/CPU_Speed_Limit/ {gsub(/[^0-9]/, "", $2); print $2; exit}')"
  if [[ -z "${limit}" ]]; then
    return 0
  fi
  if (( limit < 90 )); then
    echo "dflash: CPU thermal-throttled (CPU_Speed_Limit=${limit})" >&2
    return 3
  fi
  return 0
}

usage() {
  cat <<EOF
usage: $(basename "$0") <command> [args...]

server commands:
  start     start the server in background, wait for /health (up to 120s)
  stop      graceful SIGTERM, wait up to 30s
  restart   stop then start
  status    check running process and /health
  kill      SIGKILL the process group, clear port squatters
  logs      tail -f ${LOG_FILE}

client commands:
  opencode       launch OpenCode against the local server (forwards extra args)
  opencode-auto  launch supervised OpenCode one-shot mode with watchdog/restarts
  codex          launch Codex against the local server (forwards extra args)

autonomous queue:
  queue --dir <workdir> "<goal>"
                 plan a goal into sub-tasks with DoD checks and execute them
  queue-resume --dir <workdir> [--retry-failed] [--retry-blocked]
                 resume an existing queue
  queue-status --dir <workdir>
                 show queue state
  queue-plan "<goal>"
                 run the planner only and print JSON to stdout

system integration:
  install-launchd    register dev.dflash LaunchAgent (auto-start on login, KeepAlive)
  uninstall-launchd  remove the LaunchAgent
  thermal            print pmset -g therm; exit non-zero when CPU_Speed_Limit < 90

env:
  LOCAL_DFLASH_HOST=${HOST}
  LOCAL_DFLASH_PORT=${PORT}

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
    opencode)          cmd_opencode "$@" ;;
    opencode-auto)     cmd_opencode_auto "$@" ;;
    codex)             cmd_codex "$@" ;;
    queue)             cmd_queue "$@" ;;
    queue-resume)      cmd_queue_resume "$@" ;;
    queue-status)      cmd_queue_status "$@" ;;
    queue-plan)        cmd_queue_plan "$@" ;;
    install-launchd)   cmd_install_launchd ;;
    uninstall-launchd) cmd_uninstall_launchd ;;
    thermal)           cmd_thermal ;;
    ""|-h|--help|help) usage ;;
    *) echo "unknown command: ${cmd}" >&2; usage >&2; exit 2 ;;
  esac
}

main "$@"
