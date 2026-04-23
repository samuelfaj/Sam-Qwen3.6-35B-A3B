#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${REPO_ROOT}/.dflash.pid"
LOG_FILE="${REPO_ROOT}/dflash.log"
PORT="${LOCAL_DFLASH_PORT:-8010}"
HOST="${LOCAL_DFLASH_HOST:-127.0.0.1}"
HEALTH_URL="http://${HOST}:${PORT}/health"
METRICS_URL="http://${HOST}:${PORT}/metrics"

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

cmd_monitor() {
  local interval="${1:-${LOCAL_DFLASH_MONITOR_INTERVAL_SECONDS:-1}}"
  if ! [[ "${interval}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "dflash: monitor interval must be numeric seconds" >&2
    return 2
  fi

  local pid
  pid="$(get_running_pid)"
  if [[ -z "${pid}" ]]; then
    echo "dflash: stopped"
    echo "dflash: run '$0 start' first"
    return 1
  fi

  local python_bin="${REPO_ROOT}/.venv/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    python_bin="$(command -v python3 || true)"
  fi
  if [[ -z "${python_bin}" ]]; then
    echo "dflash: python3 not found; cannot format monitor output" >&2
    return 2
  fi

  echo "dflash monitor: ${HEALTH_URL} (pid ${pid}, interval ${interval}s, Ctrl-C to stop)"
  while true; do
    local health_json metrics_json
    if ! health_json="$(curl -fsS --max-time 2 "${HEALTH_URL}" 2>/dev/null)"; then
      printf '%s health=down pid=%s port=%s\n' "$(date '+%H:%M:%S')" "${pid}" "${PORT}"
      sleep "${interval}"
      continue
    fi
    metrics_json="$(curl -fsS --max-time 2 "${METRICS_URL}" 2>/dev/null || printf '{}')"
    DFLASH_HEALTH_JSON="${health_json}" DFLASH_METRICS_JSON="${metrics_json}" "${python_bin}" - <<'PY'
import json
import os
import time


def load_env_json(name):
    try:
        return json.loads(os.environ.get(name, "{}"))
    except Exception:
        return {}


def gb_from_bytes(value):
    try:
        return float(value) / (1024 ** 3)
    except Exception:
        return 0.0


def fmt_gb(value):
    return f"{float(value):.2f}GB"


health = load_env_json("DFLASH_HEALTH_JSON")
metrics = load_env_json("DFLASH_METRICS_JSON")
now = time.strftime("%H:%M:%S")
loaded = "yes" if int(metrics.get("dflash_model_loaded", 1 if health.get("loaded") else 0) or 0) else "no"
active = int(metrics.get("dflash_active_generation_requests", health.get("active_generation_requests", 0)) or 0)
queued = int(metrics.get("dflash_queued_generation_requests", health.get("queued_generation_requests", 0)) or 0)
age = float(metrics.get("dflash_active_ticket_age_seconds", 0.0) or 0.0)
uptime = float(metrics.get("dflash_uptime_seconds", 0.0) or 0.0)
active_gb = gb_from_bytes(metrics.get("mlx_active_memory_bytes", 0.0))
cache_gb = gb_from_bytes(metrics.get("mlx_cache_memory_bytes", 0.0))
peak_gb = gb_from_bytes(metrics.get("mlx_peak_memory_bytes", 0.0))
history = int(metrics.get("dflash_response_history_entries", health.get("response_history_entries", 0)) or 0)
prefix_entries = int(metrics.get("dflash_prefix_cache_entries", health.get("prefix_cache_entries", 0)) or 0)
global_entries = int(metrics.get("dflash_global_prefix_cache_entries", health.get("global_prefix_cache_entries", 0)) or 0)
hits = int(metrics.get("dflash_global_prefix_cache_hits", health.get("global_prefix_cache_hits", 0)) or 0)
misses = int(metrics.get("dflash_global_prefix_cache_misses", health.get("global_prefix_cache_misses", 0)) or 0)
block_size = int(metrics.get("dflash_block_size", health.get("block_size", 0)) or 0)
context_window = int(metrics.get("dflash_context_window", health.get("context_window", 0)) or 0)
max_tokens = int(metrics.get("dflash_max_tokens_limit", health.get("max_tokens_limit", 0)) or 0)
last = health.get("last_request_metrics") or {}
last_prompt_tps = float(metrics.get("dflash_last_request_prompt_tps", last.get("prompt_tps", 0.0)) or 0.0)
last_generation_tps = float(metrics.get("dflash_last_request_generation_tps", last.get("generation_tps", 0.0)) or 0.0)
last_prompt_tokens = int(metrics.get("dflash_last_request_prompt_tokens", last.get("prompt_tokens", 0)) or 0)
last_generated_tokens = int(metrics.get("dflash_last_request_generated_tokens", last.get("generated_tokens", 0)) or 0)
last_elapsed = float(metrics.get("dflash_last_request_elapsed", last.get("elapsed", 0.0)) or 0.0)
last_acceptance = float(metrics.get("dflash_last_request_avg_acceptance_ratio", last.get("avg_acceptance_ratio", 0.0)) or 0.0)

print(
    f"{now} loaded={loaded} active={active} queued={queued} active_age={age:.1f}s "
    f"mem={fmt_gb(active_gb)} cache={fmt_gb(cache_gb)} peak={fmt_gb(peak_gb)} "
    f"last={last_generated_tokens}tok/{last_elapsed:.1f}s gen={last_generation_tps:.1f}tok/s "
    f"prompt={last_prompt_tokens}tok@{last_prompt_tps:.1f}tok/s accept={last_acceptance:.2f} "
    f"history={history} prefix={prefix_entries}/{global_entries} global_hit/miss={hits}/{misses} "
    f"block={block_size} ctx={context_window} max={max_tokens} uptime={uptime:.0f}s",
    flush=True,
)
PY
    sleep "${interval}"
  done
}

cmd_monitor_tui() {
  local interval="${1:-${LOCAL_DFLASH_MONITOR_INTERVAL_SECONDS:-1}}"
  if ! [[ "${interval}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "dflash: monitor-tui interval must be numeric seconds" >&2
    return 2
  fi

  local pid
  pid="$(get_running_pid)"
  if [[ -z "${pid}" ]]; then
    echo "dflash: stopped"
    echo "dflash: run '$0 start' first"
    return 1
  fi

  local python_bin="${REPO_ROOT}/.venv/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    python_bin="$(command -v python3 || true)"
  fi
  if [[ -z "${python_bin}" ]]; then
    echo "dflash: python3 not found; cannot render monitor TUI" >&2
    return 2
  fi

  DFLASH_HEALTH_URL="${HEALTH_URL}" \
  DFLASH_METRICS_URL="${METRICS_URL}" \
  DFLASH_REQUESTS_URL="http://${HOST}:${PORT}/requests?limit=8" \
  DFLASH_MONITOR_INTERVAL="${interval}" \
  DFLASH_MONITOR_PID="${pid}" \
  "${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os
import select
import shutil
import sys
import termios
import time
import tty
import urllib.error
import urllib.request


HEALTH_URL = os.environ["DFLASH_HEALTH_URL"]
METRICS_URL = os.environ["DFLASH_METRICS_URL"]
REQUESTS_URL = os.environ["DFLASH_REQUESTS_URL"]
INTERVAL = max(0.1, float(os.environ.get("DFLASH_MONITOR_INTERVAL", "1")))
SERVER_PID = os.environ.get("DFLASH_MONITOR_PID", "?")
TTY = sys.stdout.isatty()

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bg": "\033[48;5;236m",
}


def c(name: str, text: str) -> str:
    if not TTY:
        return text
    return f"{COLORS.get(name, '')}{text}{COLORS['reset']}"


def fetch_json(url: str, timeout: float = 2.0) -> tuple[dict, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except Exception as exc:
        return {}, str(exc)


def num(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def integer(value, default=0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def gb_from_bytes(value) -> float:
    return num(value) / (1024 ** 3)


def fmt_seconds(value) -> str:
    seconds = max(0.0, num(value))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{rem:02d}s"
    hours = minutes // 60
    minutes %= 60
    return f"{hours}h{minutes:02d}m"


def fmt_gb(value) -> str:
    return f"{num(value):.2f} GB"


def clamp_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    text = str(text)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def bar(value: float, limit: float, width: int = 18) -> str:
    if width <= 0:
        return ""
    ratio = 0.0 if limit <= 0 else max(0.0, min(1.0, value / limit))
    fill = int(round(ratio * width))
    return "[" + "#" * fill + "-" * (width - fill) + "]"


def line(width: int, char: str = "-") -> str:
    return char * max(0, width)


def metric_row(label: str, value: str, width: int, color: str = "white") -> str:
    label_width = min(18, max(10, width // 4))
    value_width = max(0, width - label_width - 1)
    return f"{c('dim', label.ljust(label_width))} {c(color, clamp_text(value, value_width))}"


def build_screen(health: dict, metrics: dict, requests: dict, errors: list[str]) -> str:
    width, height = shutil.get_terminal_size((110, 32))
    width = max(80, width)
    height = max(24, height)

    loaded = integer(metrics.get("dflash_model_loaded", 1 if health.get("loaded") else 0))
    active = integer(metrics.get("dflash_active_generation_requests", health.get("active_generation_requests", 0)))
    queued = integer(metrics.get("dflash_queued_generation_requests", health.get("queued_generation_requests", 0)))
    age = num(metrics.get("dflash_active_ticket_age_seconds", 0.0))
    uptime = num(metrics.get("dflash_uptime_seconds", 0.0))

    active_gb = gb_from_bytes(metrics.get("mlx_active_memory_bytes", 0.0))
    cache_gb = gb_from_bytes(metrics.get("mlx_cache_memory_bytes", 0.0))
    peak_gb = gb_from_bytes(metrics.get("mlx_peak_memory_bytes", 0.0))

    max_tokens = integer(metrics.get("dflash_max_tokens_limit", health.get("max_tokens_limit", 0)))
    context = integer(metrics.get("dflash_context_window", health.get("context_window", 0)))
    block = integer(metrics.get("dflash_block_size", health.get("block_size", 0)))
    history = integer(metrics.get("dflash_response_history_entries", health.get("response_history_entries", 0)))
    prefix = integer(metrics.get("dflash_prefix_cache_entries", health.get("prefix_cache_entries", 0)))
    global_prefix = integer(metrics.get("dflash_global_prefix_cache_entries", health.get("global_prefix_cache_entries", 0)))
    hits = integer(metrics.get("dflash_global_prefix_cache_hits", health.get("global_prefix_cache_hits", 0)))
    misses = integer(metrics.get("dflash_global_prefix_cache_misses", health.get("global_prefix_cache_misses", 0)))

    last = health.get("last_request_metrics") or {}
    last_elapsed = num(metrics.get("dflash_last_request_elapsed", last.get("elapsed", 0.0)))
    last_prompt_tps = num(metrics.get("dflash_last_request_prompt_tps", last.get("prompt_tps", 0.0)))
    last_generation_tps = num(metrics.get("dflash_last_request_generation_tps", last.get("generation_tps", 0.0)))
    last_prompt_tokens = integer(metrics.get("dflash_last_request_prompt_tokens", last.get("prompt_tokens", 0)))
    last_generated_tokens = integer(metrics.get("dflash_last_request_generated_tokens", last.get("generated_tokens", 0)))
    last_accept = num(metrics.get("dflash_last_request_avg_acceptance_ratio", last.get("avg_acceptance_ratio", 0.0)))
    last_finish = last.get("finish_reason", "n/a")
    last_surface = last.get("surface", "n/a")

    status_text = "RUNNING" if loaded or active else "IDLE"
    status_color = "green" if active else ("cyan" if loaded else "yellow")
    if errors:
        status_text = "DEGRADED"
        status_color = "red"

    left = 38
    mid = 38
    gap = "  "
    right = max(24, width - left - mid - len(gap) * 2)

    rows: list[str] = []
    title = f" DFlash Agentic Local Server "
    subtitle = f"pid {SERVER_PID} | {HEALTH_URL} | refresh {INTERVAL:g}s | q quits"
    rows.append(c("bold", title) + c("dim", " " + subtitle))
    rows.append(line(width))
    rows.append(
        metric_row("status", status_text, left, status_color)
        + gap
        + metric_row("active/queued", f"{active}/{queued}  age {fmt_seconds(age)}", mid, "white")
        + gap
        + metric_row("uptime", fmt_seconds(uptime), right, "white")
    )
    rows.append(
        metric_row("memory active", fmt_gb(active_gb), left, "green")
        + gap
        + metric_row("cache", fmt_gb(cache_gb), mid, "yellow")
        + gap
        + metric_row("peak", fmt_gb(peak_gb), right, "magenta")
    )
    rows.append(
        metric_row("context", f"{context} ctx / {max_tokens} max", left, "white")
        + gap
        + metric_row("block", str(block), mid, "white")
        + gap
        + metric_row("history", f"{history} responses", right, "white")
    )
    rows.append(
        metric_row("prefix cache", f"{prefix} response / {global_prefix} global", left, "cyan")
        + gap
        + metric_row("global hit/miss", f"{hits}/{misses}", mid, "cyan")
        + gap
        + metric_row("last surface", str(last_surface), right, "white")
    )
    rows.append(line(width))

    rows.append(c("bold", "Last request"))
    rows.append(
        metric_row("tokens", f"{last_generated_tokens} out / {last_prompt_tokens} prompt", left, "white")
        + gap
        + metric_row("elapsed", fmt_seconds(last_elapsed), mid, "white")
        + gap
        + metric_row("finish", str(last_finish), right, "white")
    )
    rows.append(
        metric_row("generation", f"{last_generation_tps:.1f} tok/s {bar(last_generation_tps, 60, 12)}", left, "green")
        + gap
        + metric_row("prefill", f"{last_prompt_tps:.1f} tok/s {bar(last_prompt_tps, 2500, 12)}", mid, "cyan")
        + gap
        + metric_row("accept", f"{last_accept:.2f} {bar(last_accept, 1.0, 12)}", right, "yellow")
    )
    rows.append(line(width))

    rows.append(c("bold", "Recent requests"))
    entries = list((requests or {}).get("entries") or [])[-8:]
    if not entries:
        rows.append(c("dim", "  no completed request metrics yet"))
    else:
        header = "  time      surface    out  prompt  gen tok/s  prefill tok/s  accept  finish"
        rows.append(c("dim", clamp_text(header, width)))
        for item in reversed(entries):
            ts = item.get("finished_at") or item.get("ts") or 0
            try:
                when = time.strftime("%H:%M:%S", time.localtime(float(ts)))
            except Exception:
                when = "--:--:--"
            row = (
                f"  {when}  "
                f"{str(item.get('surface', 'n/a'))[:9].ljust(9)} "
                f"{integer(item.get('generated_tokens', 0)):>5} "
                f"{integer(item.get('prompt_tokens', 0)):>7} "
                f"{num(item.get('generation_tps', 0.0)):>9.1f} "
                f"{num(item.get('prompt_tps', 0.0)):>13.1f} "
                f"{num(item.get('avg_acceptance_ratio', 0.0)):>6.2f} "
                f"{str(item.get('finish_reason', 'n/a'))[:12]}"
            )
            rows.append(clamp_text(row, width))

    if errors:
        rows.append(line(width))
        rows.append(c("red", "Errors"))
        for error in errors[-3:]:
            rows.append(c("red", "  " + clamp_text(error, width - 2)))

    rows.append("")
    rows.append(c("dim", "Tip: run a Codex/OpenCode request in another terminal; this screen updates when generation finishes."))

    return "\n".join(rows[:height])


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    try:
        return sys.stdin.read(1)
    except Exception:
        return None


def main() -> int:
    old_term = None
    if TTY and sys.stdin.isatty():
        old_term = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin)
    try:
        if TTY:
            sys.stdout.write("\033[?1049h\033[?25l")
            sys.stdout.flush()
        while True:
            health, health_err = fetch_json(HEALTH_URL)
            metrics, metrics_err = fetch_json(METRICS_URL)
            requests, requests_err = fetch_json(REQUESTS_URL)
            errors = [err for err in (health_err, metrics_err, requests_err) if err]
            screen = build_screen(health, metrics, requests, errors)
            if TTY:
                sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(screen + "\n")
            sys.stdout.flush()
            deadline = time.time() + INTERVAL
            while time.time() < deadline:
                key = read_key()
                if key in {"q", "Q", "\x03"}:
                    return 0
                time.sleep(0.05)
            if not TTY:
                sys.stdout.write("\n")
    except KeyboardInterrupt:
        return 0
    finally:
        if old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        if TTY:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
    return 0


raise SystemExit(main())
PY
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
  monitor   poll /health and /metrics live (optional interval seconds)
  monitor-tui
            full-screen dynamic monitor (optional interval seconds, q quits)

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
    monitor)           cmd_monitor "$@" ;;
    monitor-tui)       cmd_monitor_tui "$@" ;;
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
