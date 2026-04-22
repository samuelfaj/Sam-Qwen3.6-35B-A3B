#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from queue import Empty, Queue
from typing import Any


ACTION_INTENT_MARKERS = (
    "i'll",
    "i will",
    "let me",
    "now i'll",
    "next i'll",
    "vou",
    "agora vou",
    "deixa eu",
)
PROGRESS_EVENT_TYPES = {"step_start", "text", "tool_use", "step_finish"}
DEFAULT_STALL_TIMEOUT_SECONDS = 900
DEFAULT_RESTART_DELAY_SECONDS = 5
DEFAULT_MAX_RESTARTS = 12
DEFAULT_LOOP_REPEAT_THRESHOLD = 3
DEFAULT_RECENT_EVENT_LIMIT = 64
DEFAULT_RECENT_TEXT_LIMIT = 8
DEFAULT_RECENT_TOOL_LIMIT = 8


def _env_non_negative_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return default if value < 0 else value


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return default if value <= 0 else value


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _looks_like_action_intent(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(marker in normalized for marker in ACTION_INTENT_MARKERS)


def _truncate(text: str, limit: int = 280) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _parse_workdir(run_args: list[str]) -> str | None:
    for idx, arg in enumerate(run_args):
        if arg == "--dir" and idx + 1 < len(run_args):
            return run_args[idx + 1]
        if arg.startswith("--dir="):
            return arg.split("=", 1)[1]
    return None


def _extract_task(run_args: list[str]) -> str | None:
    positional: list[str] = []
    skip_next = False
    for arg in run_args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--dir", "--format", "--model"}:
            skip_next = True
            continue
        if arg.startswith("--dir=") or arg.startswith("--format=") or arg.startswith("--model="):
            continue
        if arg.startswith("-"):
            continue
        positional.append(arg)
    return positional[-1] if positional else None


def _ensure_json_run_args(run_args: list[str]) -> list[str]:
    args = list(run_args)
    format_value: str | None = None
    for idx, arg in enumerate(args):
        if arg == "--format" and idx + 1 < len(args):
            format_value = args[idx + 1]
            break
        if arg.startswith("--format="):
            format_value = arg.split("=", 1)[1]
            break

    if format_value is None:
        args.extend(["--format", "json"])
    elif format_value != "json":
        raise SystemExit("opencode watchdog requires '--format json'")

    if "--print-logs" not in args:
        args.append("--print-logs")
    return args


def _build_checkpoint_path(checkpoint_dir: Path, workdir: str, started_at: int) -> Path:
    slug = Path(workdir).name or "workspace"
    return checkpoint_dir / f"{started_at}-{slug}.json"


def _tool_signature(part: dict[str, Any]) -> str:
    tool_name = str(part.get("tool") or "")
    state = part.get("state") or {}
    tool_input = state.get("input")
    try:
        encoded_input = json.dumps(tool_input, sort_keys=True, ensure_ascii=False)
    except TypeError:
        encoded_input = repr(tool_input)
    return f"{tool_name}:{encoded_input}"


def _append_recent(buffer: deque[str], value: str, limit: int) -> None:
    if len(buffer) >= limit:
        buffer.popleft()
    buffer.append(value)


def _repeat_count(values: deque[str]) -> int:
    if not values:
        return 0
    counts = Counter(values)
    return max(counts.values())


def _build_resume_prompt(state: dict[str, Any]) -> str:
    original_task = state.get("original_task") or "Continue the previous task."
    failure_reason = state.get("last_failure_reason") or "previous run stopped unexpectedly"
    attempt = state.get("attempt", 1)
    recent_texts = state.get("recent_texts") or []
    recent_tools = state.get("recent_tool_signatures") or []
    lines = [
        "Resume the interrupted long-running software task from the current filesystem state.",
        f"Original objective: {original_task}",
        f"Restart attempt: {attempt}",
        f"Previous stop reason: {failure_reason}",
        "Rules:",
        "- Do not restart from scratch.",
        "- Do not repeat completed edits or repeated diagnoses unless verification requires it.",
        "- First inspect the current files/tests, then continue from the latest incomplete step.",
        "- If a tool is needed, call it immediately instead of only describing the next action.",
    ]
    if recent_texts:
        lines.append("Recent assistant outputs:")
        lines.extend(f"- {_truncate(text)}" for text in recent_texts[-3:])
    if recent_tools:
        lines.append("Recent tool calls:")
        lines.extend(f"- {_truncate(tool)}" for tool in recent_tools[-3:])
    return "\n".join(lines)


def _read_stream(stream, source: str, queue: Queue) -> None:
    try:
        for line in iter(stream.readline, ""):
            queue.put((source, line))
    finally:
        queue.put((f"{source}_closed", None))


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 10
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value
        for key, value in state.items()
        if not key.endswith("_deque")
    }
    serializable["updated_at"] = int(time.time())
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _handle_json_event(event: dict[str, Any], state: dict[str, Any]) -> str | None:
    event_type = str(event.get("type") or "")
    if event_type in PROGRESS_EVENT_TYPES:
        state["last_progress_at"] = time.time()
        state["last_progress_event"] = event_type

    state["recent_events"].append(
        {
            "ts": int(time.time()),
            "type": event_type,
            "summary": _truncate(json.dumps(event, ensure_ascii=False)),
        }
    )
    if len(state["recent_events"]) > state["recent_event_limit"]:
        state["recent_events"] = state["recent_events"][-state["recent_event_limit"] :]

    part = event.get("part") or {}
    if event_type == "text":
        text = str(part.get("text") or "")
        normalized = _normalize_text(text)
        if normalized:
            _append_recent(state["recent_texts_deque"], text, state["recent_text_limit"])
            state["recent_texts"] = list(state["recent_texts_deque"])
            state["last_text"] = text
            state["last_text_at"] = time.time()
            if _repeat_count(state["recent_texts_deque"]) >= state["loop_repeat_threshold"]:
                return "repeated_text_loop"

    if event_type == "tool_use":
        signature = _tool_signature(part)
        if signature:
            _append_recent(state["recent_tool_signatures_deque"], signature, state["recent_tool_limit"])
            state["recent_tool_signatures"] = list(state["recent_tool_signatures_deque"])
            state["last_tool_signature"] = signature
            state["last_tool_at"] = time.time()
            if _repeat_count(state["recent_tool_signatures_deque"]) >= state["loop_repeat_threshold"]:
                return "repeated_tool_loop"

    if event_type == "step_finish":
        state["last_step_reason"] = str(part.get("reason") or "")
        state["last_step_finish_at"] = time.time()

    return None


def _should_retry_after_exit(returncode: int, state: dict[str, Any]) -> str | None:
    if returncode != 0:
        return f"exit_code_{returncode}"

    last_text = str(state.get("last_text") or "")
    last_text_at = float(state.get("last_text_at") or 0.0)
    last_tool_at = float(state.get("last_tool_at") or 0.0)
    last_step_reason = str(state.get("last_step_reason") or "")
    if last_step_reason == "stop" and _looks_like_action_intent(last_text) and last_text_at >= last_tool_at:
        return "action_only_stop"
    return None


def _monitor_attempt(
    model: str,
    run_args: list[str],
    state: dict[str, Any],
) -> tuple[int, str | None]:
    command = ["opencode", "run", "-m", model, "--dangerously-skip-permissions", *run_args]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    state["active_pid"] = process.pid
    state["last_progress_at"] = time.time()

    queue: Queue = Queue()
    stdout_thread = threading.Thread(target=_read_stream, args=(process.stdout, "stdout", queue), daemon=True)
    stderr_thread = threading.Thread(target=_read_stream, args=(process.stderr, "stderr", queue), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    stdout_closed = False
    stderr_closed = False
    loop_reason: str | None = None

    while True:
        try:
            source, payload = queue.get(timeout=1.0)
        except Empty:
            if process.poll() is not None and stdout_closed and stderr_closed:
                break
            if time.time() - float(state["last_progress_at"]) > state["stall_timeout_seconds"]:
                loop_reason = "stall_timeout"
                _kill_process_tree(process)
                break
            continue

        if source == "stdout_closed":
            stdout_closed = True
            continue
        if source == "stderr_closed":
            stderr_closed = True
            continue
        if payload is None:
            continue

        if source == "stdout":
            sys.stdout.write(payload)
            sys.stdout.flush()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            loop_reason = _handle_json_event(event, state)
            if loop_reason:
                _kill_process_tree(process)
                break
            continue

        sys.stderr.write(payload)
        sys.stderr.flush()

    returncode = process.wait()
    state["last_returncode"] = returncode
    state["active_pid"] = None
    if loop_reason is not None:
        return returncode, loop_reason
    return returncode, _should_retry_after_exit(returncode, state)


def _build_attempt_args(state: dict[str, Any], base_run_args: list[str]) -> list[str]:
    if state["attempt"] <= 1:
        return list(base_run_args)

    task = _extract_task(base_run_args)
    if task is None:
        return list(base_run_args)

    resume_prompt = _build_resume_prompt(state)
    rebuilt_args = list(base_run_args)
    for idx in range(len(rebuilt_args) - 1, -1, -1):
        if rebuilt_args[idx] == task:
            rebuilt_args[idx] = resume_prompt
            break
    state["last_resume_prompt"] = resume_prompt
    return rebuilt_args


def run_watchdog(args: argparse.Namespace) -> int:
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    started_at = int(time.time())
    workdir = _parse_workdir(args.run_args) or os.getcwd()
    checkpoint_path = _build_checkpoint_path(checkpoint_dir, workdir, started_at)

    state: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at,
        "updated_at": started_at,
        "workdir": workdir,
        "original_task": _extract_task(args.run_args),
        "attempt": 0,
        "max_restarts": args.max_restarts,
        "stall_timeout_seconds": args.stall_timeout_seconds,
        "restart_delay_seconds": args.restart_delay_seconds,
        "loop_repeat_threshold": args.loop_repeat_threshold,
        "recent_event_limit": DEFAULT_RECENT_EVENT_LIMIT,
        "recent_text_limit": DEFAULT_RECENT_TEXT_LIMIT,
        "recent_tool_limit": DEFAULT_RECENT_TOOL_LIMIT,
        "recent_events": [],
        "recent_texts": [],
        "recent_tool_signatures": [],
        "last_failure_reason": None,
        "last_resume_prompt": None,
        "last_progress_event": None,
        "last_progress_at": time.time(),
        "last_step_reason": None,
        "last_text": None,
        "last_text_at": 0.0,
        "last_tool_signature": None,
        "last_tool_at": 0.0,
        "last_step_finish_at": 0.0,
        "last_returncode": None,
        "active_pid": None,
        "completed": False,
        "base_run_args": list(args.run_args),
    }
    state["recent_texts_deque"] = deque(maxlen=DEFAULT_RECENT_TEXT_LIMIT)
    state["recent_tool_signatures_deque"] = deque(maxlen=DEFAULT_RECENT_TOOL_LIMIT)
    _write_checkpoint(checkpoint_path, state)

    exit_code = 1
    for attempt in range(1, args.max_restarts + 2):
        state["attempt"] = attempt
        run_args = _build_attempt_args(state, args.run_args)
        _write_checkpoint(checkpoint_path, state)
        returncode, retry_reason = _monitor_attempt(args.model, run_args, state)
        state["last_failure_reason"] = retry_reason
        _write_checkpoint(checkpoint_path, state)
        if retry_reason is None:
            state["completed"] = True
            _write_checkpoint(checkpoint_path, state)
            return returncode
        if attempt > args.max_restarts:
            exit_code = returncode if returncode != 0 else 1
            break
        time.sleep(args.restart_delay_seconds)

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervise long-running opencode run tasks with watchdogs and checkpoints.")
    parser.add_argument("--model", required=True, help="Fully qualified OpenCode model id.")
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("LOCAL_DFLASH_OPENCODE_CHECKPOINT_DIR", ".opencode-watchdog"),
        help="Directory where checkpoint JSON files are stored.",
    )
    parser.add_argument(
        "--stall-timeout-seconds",
        type=int,
        default=_env_positive_int("LOCAL_DFLASH_OPENCODE_STALL_TIMEOUT_SECONDS", DEFAULT_STALL_TIMEOUT_SECONDS),
        help="Restart if no progress events arrive within this many seconds.",
    )
    parser.add_argument(
        "--restart-delay-seconds",
        type=int,
        default=_env_positive_int("LOCAL_DFLASH_OPENCODE_RESTART_DELAY_SECONDS", DEFAULT_RESTART_DELAY_SECONDS),
        help="Seconds to wait before restarting after a stall or loop.",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=_env_non_negative_int("LOCAL_DFLASH_OPENCODE_MAX_RESTARTS", DEFAULT_MAX_RESTARTS),
        help="Maximum number of restart attempts after the first run.",
    )
    parser.add_argument(
        "--loop-repeat-threshold",
        type=int,
        default=_env_positive_int("LOCAL_DFLASH_OPENCODE_LOOP_REPEAT_THRESHOLD", DEFAULT_LOOP_REPEAT_THRESHOLD),
        help="Treat repeated identical text/tool events at or above this threshold as a loop.",
    )
    parser.add_argument("run_args", nargs=argparse.REMAINDER, help="Arguments forwarded to 'opencode run'.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    forwarded = list(args.run_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        parser.error("missing opencode run arguments")
    args.run_args = _ensure_json_run_args(forwarded)
    return run_watchdog(args)


if __name__ == "__main__":
    raise SystemExit(main())
