#!/usr/bin/env python3
"""Agent queue orchestrator for autonomous 24h runs.

Plans a high-level goal into ordered sub-tasks with programmatic
Definition-of-Done checks, executes each sub-task in a fresh opencode
session (via the existing run_opencode_local.sh watchdog), verifies
with bash checks and an AI judge, and advances the queue with retries.

Usage:
    python3 scripts/agent_queue.py run --dir <workdir> "<goal>"
    python3 scripts/agent_queue.py resume --dir <workdir>
    python3 scripts/agent_queue.py status --dir <workdir>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


LOCAL_HOST = os.environ.get("LOCAL_DFLASH_HOST", "127.0.0.1")
LOCAL_PORT = os.environ.get("LOCAL_DFLASH_PORT", "8010")
LOCAL_MODEL_NAME = os.environ.get(
    "LOCAL_DFLASH_MODEL_NAME", "qwen3.6-35b-a3b-dflash-local"
)
LOCAL_API_BASE = f"http://{LOCAL_HOST}:{LOCAL_PORT}/v1"
HEALTH_URL = f"http://{LOCAL_HOST}:{LOCAL_PORT}/health"

QUEUE_DIR_NAME = ".agent-queue"
STATE_FILENAME = "state.json"
LOG_DIRNAME = "logs"

MAX_ATTEMPTS = int(os.environ.get("LOCAL_DFLASH_QUEUE_MAX_ATTEMPTS", "3"))
DOD_CHECK_TIMEOUT = int(os.environ.get("LOCAL_DFLASH_QUEUE_DOD_TIMEOUT", "120"))
REQUEST_TIMEOUT = int(
    os.environ.get("LOCAL_DFLASH_QUEUE_REQUEST_TIMEOUT_SECONDS", "600")
)
PLANNER_MAX_TOKENS = int(
    os.environ.get("LOCAL_DFLASH_QUEUE_PLANNER_MAX_TOKENS", "4096")
)
JUDGE_MAX_TOKENS = int(os.environ.get("LOCAL_DFLASH_QUEUE_JUDGE_MAX_TOKENS", "128"))
PLANNER_RETRIES = int(os.environ.get("LOCAL_DFLASH_QUEUE_PLANNER_RETRIES", "2"))

# Phase 2B — 24h autonomy knobs.
WALLCLOCK_HOURS = float(os.environ.get("LOCAL_DFLASH_QUEUE_WALLCLOCK_HOURS", "24"))
TOKEN_BUDGET = int(os.environ.get("LOCAL_DFLASH_QUEUE_TOKEN_BUDGET", "0"))  # 0 = unbounded
EXECUTOR_TIMEOUT = int(os.environ.get("LOCAL_DFLASH_QUEUE_EXECUTOR_TIMEOUT", "3600"))
REPLAN_AFTER_FAILURES = int(os.environ.get("LOCAL_DFLASH_QUEUE_REPLAN_AFTER_FAILURES", "2"))
REPLAN_MAX = int(os.environ.get("LOCAL_DFLASH_QUEUE_REPLAN_MAX", "3"))
SHARED_MEMORY_MAX_BYTES = int(os.environ.get("LOCAL_DFLASH_QUEUE_SHARED_MEMORY_BYTES", "65536"))
HEARTBEAT_FILENAME = "heartbeat"
RUN_JSONL_FILENAME = "run.jsonl"
ROLLOUTS_DIRNAME = "rollouts"


PLANNER_SYSTEM = (
    "You are a planning module for a local coding agent. "
    "Decompose the user's high-level goal into an ordered list of small sub-tasks, "
    "each executable in a single fresh agent session with its own clean context. "
    "Return ONLY a JSON object. No prose, no markdown, no code fences.\n\n"
    "Schema:\n"
    '{"tasks":[{'
    '"id":"t1",'
    '"title":"short human-readable title",'
    '"instruction":"concrete self-contained instruction for the executor agent, '
    "including any critical context, file paths, and acceptance constraints\","
    '"dod_checks":['
    '{"type":"bash","command":"test -f package.json","expect_exit":0,'
    '"description":"package.json exists"}'
    "],"
    '"depends_on":[]'
    "}]}\n\n"
    "Rules:\n"
    "- Each task must be independently executable given the filesystem state "
    "left by its dependencies.\n"
    "- dod_checks are non-interactive bash one-liners run from the working directory. "
    "expect_exit=0 means the check passed.\n"
    "- Prefer standard unix tools (test, grep, jq, node, python, cat, diff, timeout).\n"
    "- Checks must be deterministic and mechanical. Never depend on human judgment.\n"
    "- Every task must have at least one check specific to that task.\n"
    "- Keep tasks small. 5 to 25 tasks is typical. Order them so dependencies come first.\n"
    "- The instruction field is read with no prior chat history. It must stand alone."
)

JUDGE_SYSTEM = (
    "You are a strict task-completion judge for a local coding agent. "
    "You are given a sub-task, the bash checks that define 'done', the mechanical "
    "results of those checks, and the executor agent's last text output. "
    "Respond with EXACTLY one uppercase token on the first line: DONE, CONTINUE, or BLOCKED.\n"
    "- DONE: all required checks pass and the work is verifiably complete.\n"
    "- CONTINUE: one or more checks fail but another attempt with a hint could succeed.\n"
    "- BLOCKED: cannot be completed without external intervention "
    "(missing dependency on the host, missing credentials, human decision, "
    "impossible requirement).\n"
    "When unsure between CONTINUE and DONE, answer CONTINUE.\n"
    "On the second line give one short sentence: a hint for the next attempt "
    "(when CONTINUE) or the reason (when BLOCKED). Leave it empty for DONE."
)

EXECUTOR_PROMPT_TEMPLATE = """\
You are working on a single sub-task of a larger project. Focus only on this sub-task.

Working directory: {workdir}

Objective: {title}

Detailed instruction:
{instruction}

Definition of Done (ALL of the following bash checks must exit with code 0 before you stop):
{dod_text}

Rules:
- Run the checks above yourself (via your shell tool) to verify before stopping.
- If a check fails after your edit, fix the problem and re-verify immediately.
- Do not end your turn by describing the next action. Execute tool calls instead.
- Ignore work outside this sub-task. The queue will handle other sub-tasks separately.
{hint_block}"""


@dataclasses.dataclass
class DoDCheck:
    type: str = "bash"
    command: str = ""
    expect_exit: int = 0
    description: str = ""


@dataclasses.dataclass
class Task:
    id: str
    title: str
    instruction: str
    dod_checks: list[DoDCheck] = dataclasses.field(default_factory=list)
    depends_on: list[str] = dataclasses.field(default_factory=list)
    attempts: int = 0
    state: str = "pending"
    last_hint: str = ""
    last_verdict: str = ""
    last_check_results: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    last_log_path: str = ""


@dataclasses.dataclass
class QueueState:
    goal: str
    workdir: str
    created_at: int
    updated_at: int
    tasks: list[Task]
    tokens_used: int = 0
    replan_count: int = 0
    consecutive_failures: int = 0
    shared_memory: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueueState":
        tasks: list[Task] = []
        for raw in data.get("tasks", []):
            raw = dict(raw)
            checks_raw = raw.pop("dod_checks", [])
            raw["dod_checks"] = [DoDCheck(**c) for c in checks_raw]
            tasks.append(Task(**raw))
        shared_memory = data.get("shared_memory") or {}
        if not isinstance(shared_memory, dict):
            shared_memory = {}
        return cls(
            goal=data.get("goal", ""),
            workdir=data.get("workdir", ""),
            created_at=data.get("created_at", int(time.time())),
            updated_at=data.get("updated_at", int(time.time())),
            tasks=tasks,
            tokens_used=int(data.get("tokens_used") or 0),
            replan_count=int(data.get("replan_count") or 0),
            consecutive_failures=int(data.get("consecutive_failures") or 0),
            shared_memory=shared_memory,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "workdir": self.workdir,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tasks": [dataclasses.asdict(t) for t in self.tasks],
            "tokens_used": self.tokens_used,
            "replan_count": self.replan_count,
            "consecutive_failures": self.consecutive_failures,
            "shared_memory": self.shared_memory,
        }


def _queue_dir(workdir: Path) -> Path:
    return workdir / QUEUE_DIR_NAME


def _state_path(workdir: Path) -> Path:
    return _queue_dir(workdir) / STATE_FILENAME


def _log_dir(workdir: Path) -> Path:
    return _queue_dir(workdir) / LOG_DIRNAME


def _rollouts_dir(workdir: Path) -> Path:
    return _queue_dir(workdir) / ROLLOUTS_DIRNAME


def _heartbeat_path(workdir: Path) -> Path:
    return _queue_dir(workdir) / HEARTBEAT_FILENAME


def _run_jsonl_path(workdir: Path) -> Path:
    return _queue_dir(workdir) / RUN_JSONL_FILENAME


def _touch_heartbeat(workdir: Path) -> None:
    path = _heartbeat_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    try:
        path.write_text(str(now), encoding="utf-8")
    except OSError:
        pass


def _record_run_event(workdir: Path, event: dict[str, Any]) -> None:
    path = _run_jsonl_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("ts", time.time())
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


def load_state(workdir: Path) -> QueueState | None:
    path = _state_path(workdir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"corrupt state file at {path}: {exc}") from exc
    return QueueState.from_dict(data)


def save_state(state: QueueState, workdir: Path) -> None:
    path = _state_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = int(time.time())
    path.write_text(
        json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ensure_local_server() -> None:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as resp:
            if 200 <= resp.status < 300:
                return
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        pass
    raise RuntimeError(
        f"Local DFlash server not responding at {HEALTH_URL}. "
        "Start it with: ./scripts/dflash.sh start"
    )


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{LOCAL_API_BASE}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _chat_completion(
    messages: list[dict[str, Any]], max_tokens: int, temperature: float = 0.0
) -> str:
    text, _ = _chat_completion_with_usage(messages, max_tokens, temperature)
    return text


def _chat_completion_with_usage(
    messages: list[dict[str, Any]], max_tokens: int, temperature: float = 0.0
) -> tuple[str, int]:
    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    result = _post_json("/chat/completions", payload)
    choices = result.get("choices") or []
    text = ""
    if choices:
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "")
    usage = result.get("usage") or {}
    tokens_used = int(usage.get("total_tokens") or 0)
    return text, tokens_used


def _extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None


def plan_tasks(goal: str) -> list[Task]:
    last_raw = ""
    for attempt in range(PLANNER_RETRIES + 1):
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": f"Goal:\n{goal}\n\nReturn the JSON decomposition now.",
            },
        ]
        if attempt > 0:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output was not valid JSON that matches the schema. "
                        "Return ONLY the JSON object, nothing else.\n"
                        f"Previous output started with:\n{last_raw[:400]}"
                    ),
                }
            )
        last_raw = _chat_completion(
            messages, max_tokens=PLANNER_MAX_TOKENS, temperature=0.0
        )
        parsed = _extract_json(last_raw)
        if not parsed or not isinstance(parsed.get("tasks"), list):
            continue
        tasks: list[Task] = []
        valid = True
        for idx, raw_task in enumerate(parsed["tasks"]):
            if not isinstance(raw_task, dict):
                valid = False
                break
            task_id = str(raw_task.get("id") or f"t{idx + 1}")
            checks: list[DoDCheck] = []
            for raw_check in raw_task.get("dod_checks") or []:
                if not isinstance(raw_check, dict):
                    continue
                command = str(raw_check.get("command") or "").strip()
                if not command:
                    continue
                try:
                    expect_exit = int(raw_check.get("expect_exit", 0))
                except (TypeError, ValueError):
                    expect_exit = 0
                checks.append(
                    DoDCheck(
                        type=str(raw_check.get("type") or "bash"),
                        command=command,
                        expect_exit=expect_exit,
                        description=str(raw_check.get("description") or ""),
                    )
                )
            tasks.append(
                Task(
                    id=task_id,
                    title=str(raw_task.get("title") or task_id),
                    instruction=str(raw_task.get("instruction") or ""),
                    dod_checks=checks,
                    depends_on=[
                        str(d) for d in raw_task.get("depends_on") or [] if d
                    ],
                )
            )
        if not valid or not tasks:
            continue
        return tasks
    raise RuntimeError(
        "Planner failed to return a valid JSON task list. "
        f"Last raw output (truncated):\n{last_raw[:1000]}"
    )


def run_dod_check(check: DoDCheck, workdir: Path) -> dict[str, Any]:
    check_record = dataclasses.asdict(check)
    if check.type != "bash":
        return {
            "check": check_record,
            "exit_code": -1,
            "passed": False,
            "stdout": "",
            "stderr": f"unsupported check type: {check.type}",
            "timed_out": False,
        }
    try:
        proc = subprocess.run(
            check.command,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=DOD_CHECK_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout) or ""
        return {
            "check": check_record,
            "exit_code": -1,
            "passed": False,
            "stdout": stdout[-2000:],
            "stderr": f"TIMEOUT after {DOD_CHECK_TIMEOUT}s",
            "timed_out": True,
        }
    return {
        "check": check_record,
        "exit_code": proc.returncode,
        "passed": proc.returncode == check.expect_exit,
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-2000:],
        "timed_out": False,
    }


def run_all_checks(checks: list[DoDCheck], workdir: Path) -> list[dict[str, Any]]:
    return [run_dod_check(c, workdir) for c in checks]


def _format_dod_for_prompt(checks: list[DoDCheck]) -> str:
    if not checks:
        return "(no mechanical checks defined; stop when you are confident the task is complete)"
    lines: list[str] = []
    for idx, c in enumerate(checks, 1):
        lines.append(f"{idx}. {c.description or c.command}")
        lines.append(f"   $ {c.command}")
        if c.expect_exit != 0:
            lines.append(f"   (expected exit code: {c.expect_exit})")
    return "\n".join(lines)


def _build_executor_hint_block(task: Task, shared_memory: dict[str, Any]) -> str:
    parts: list[str] = []
    if task.last_hint:
        parts.append(f"Previous attempt failed. Judge hint: {task.last_hint}")
    if shared_memory:
        mem_parts: list[str] = []
        for key in ("last_summary", "known_failing_paths", "files_touched"):
            val = shared_memory.get(key)
            if val:
                mem_parts.append(f"- {key}: {val}")
        if mem_parts:
            parts.append("Shared memory from prior tasks:\n" + "\n".join(mem_parts))
    if not parts:
        return ""
    return "\n" + "\n\n".join(parts) + "\n"


def execute_task(
    task: Task,
    workdir: Path,
    script_dir: Path,
    *,
    shared_memory: dict[str, Any] | None = None,
    timeout_seconds: int = EXECUTOR_TIMEOUT,
) -> tuple[Path, bool]:
    hint_block = _build_executor_hint_block(task, shared_memory or {})
    prompt = EXECUTOR_PROMPT_TEMPLATE.format(
        workdir=str(workdir),
        title=task.title,
        instruction=task.instruction,
        dod_text=_format_dod_for_prompt(task.dod_checks),
        hint_block=hint_block,
    )

    log_dir = _log_dir(workdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}-attempt-{task.attempts}.log"

    # Snapshot the executor log under .agent-queue/rollouts/ so retries can
    # see what the previous attempt tried. The main log stays truncated to
    # the latest attempt; rollouts keep history.
    rollouts_dir = _rollouts_dir(workdir)
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(script_dir / "run_opencode_local.sh"),
        "run-auto",
        "--dir",
        str(workdir),
        prompt,
    ]
    env = os.environ.copy()
    env.setdefault("LOCAL_DFLASH_OPENCODE_STALL_TIMEOUT_SECONDS", "600")
    env.setdefault("LOCAL_DFLASH_OPENCODE_MAX_RESTARTS", "3")
    env.setdefault("LOCAL_DFLASH_AUTOSTART", "1")

    timed_out = False
    with log_path.open("wb") as log_file:
        try:
            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
                timeout=timeout_seconds if timeout_seconds > 0 else None,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            log_file.write(
                f"\n[agent_queue] executor timed out after {timeout_seconds}s\n".encode()
            )
    # Save a rollout snapshot so future retries get the trace.
    try:
        rollout_path = rollouts_dir / f"{task.id}-attempt-{task.attempts}.log"
        rollout_path.write_bytes(log_path.read_bytes())
    except OSError:
        pass
    return log_path, timed_out


def extract_executor_final_text(log_path: Path, limit_chunks: int = 12) -> str:
    if not log_path.exists():
        return ""
    texts: list[str] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "text":
                part = event.get("part") or {}
                text = part.get("text")
                if text:
                    texts.append(str(text))
    return "\n".join(texts[-limit_chunks:])


def _fallback_log_tail(log_path: Path, limit_bytes: int = 4000) -> str:
    if not log_path.exists():
        return ""
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def judge_task(
    task: Task, check_results: list[dict[str, Any]], final_text: str
) -> tuple[str, str]:
    lines: list[str] = []
    for idx, result in enumerate(check_results, 1):
        check = result["check"]
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"{idx}. [{status}] exit={result['exit_code']} "
            f"expected={check['expect_exit']} | "
            f"{check.get('description') or check['command']}"
        )
        if not result["passed"]:
            stderr = (result.get("stderr") or "").strip().splitlines()
            if stderr:
                lines.append(f"   stderr: {stderr[-1][:200]}")
    checks_summary = "\n".join(lines) or "(no mechanical checks were configured)"

    user_content = (
        f"Sub-task id: {task.id}\n"
        f"Sub-task title: {task.title}\n\n"
        f"Instruction:\n{task.instruction}\n\n"
        f"Check results:\n{checks_summary}\n\n"
        f"Executor final text (last chunks):\n{final_text[-2000:] or '(empty)'}\n\n"
        "Verdict?"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        reply = _chat_completion(
            messages, max_tokens=JUDGE_MAX_TOKENS, temperature=0.0
        ).strip()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        return "CONTINUE", f"judge call failed: {exc}"
    if not reply:
        return "CONTINUE", "empty judge reply"
    reply_lines = [ln.strip() for ln in reply.splitlines() if ln.strip()]
    if not reply_lines:
        return "CONTINUE", "empty judge reply"
    verdict = reply_lines[0].upper().split()[0] if reply_lines[0] else ""
    if verdict not in {"DONE", "CONTINUE", "BLOCKED"}:
        verdict = "CONTINUE"
    hint = reply_lines[1] if len(reply_lines) > 1 else ""
    return verdict, hint


def _update_shared_memory(
    state: QueueState,
    task: Task,
    check_results: list[dict[str, Any]],
    final_text: str,
    verdict: str,
) -> None:
    mem = state.shared_memory
    if verdict == "DONE":
        summary = (
            final_text.strip().splitlines()[-5:] if final_text.strip() else []
        )
        mem["last_summary"] = f"[{task.id}] {task.title}: " + " | ".join(summary)[:500]
    failing = [
        str(r.get("check", {}).get("description") or r.get("check", {}).get("command"))
        for r in check_results
        if not r.get("passed")
    ]
    if failing:
        known = mem.setdefault("known_failing_paths", [])
        if isinstance(known, list):
            for item in failing[:3]:
                if item and item not in known:
                    known.append(item)
            del known[:max(0, len(known) - 25)]
    # Cap total size so shared memory doesn't grow unboundedly.
    serialized = json.dumps(mem, ensure_ascii=False)
    if len(serialized.encode("utf-8")) > SHARED_MEMORY_MAX_BYTES:
        mem.pop("known_failing_paths", None)


def _replan_remaining_tasks(state: QueueState) -> int:
    """When REPLAN_AFTER_FAILURES consecutive failures happen, ask the planner
    to rewrite the remaining tasks given the accumulated failure context.
    Returns the number of tasks replaced.
    """
    if state.replan_count >= REPLAN_MAX:
        return 0
    remaining = [t for t in state.tasks if t.state == "pending"]
    if not remaining:
        return 0
    failure_reasons: list[str] = []
    for t in state.tasks:
        if t.state in {"failed", "blocked"} and (t.last_hint or t.last_verdict):
            failure_reasons.append(
                f"[{t.state}] {t.id} ({t.title}): {t.last_hint or t.last_verdict}"
            )
    try:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original goal:\n{state.goal}\n\n"
                    f"Failed / blocked tasks so far:\n"
                    + ("\n".join(failure_reasons) or "(none)")
                    + "\n\nRemaining pending tasks:\n"
                    + "\n".join(f"- {t.id}: {t.title}" for t in remaining)
                    + "\n\nReturn a NEW JSON task list that replaces the remaining ones. "
                    "Keep the same id prefix discipline but feel free to restructure. "
                    "Acknowledge the failure reasons in your plan."
                ),
            },
        ]
        raw = _chat_completion(messages, max_tokens=PLANNER_MAX_TOKENS, temperature=0.0)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return 0
    parsed = _extract_json(raw)
    if not parsed or not isinstance(parsed.get("tasks"), list):
        return 0
    new_tasks: list[Task] = []
    for idx, raw_task in enumerate(parsed["tasks"]):
        if not isinstance(raw_task, dict):
            continue
        task_id = str(raw_task.get("id") or f"r{state.replan_count + 1}t{idx + 1}")
        checks: list[DoDCheck] = []
        for raw_check in raw_task.get("dod_checks") or []:
            if not isinstance(raw_check, dict):
                continue
            command = str(raw_check.get("command") or "").strip()
            if not command:
                continue
            try:
                expect_exit = int(raw_check.get("expect_exit", 0))
            except (TypeError, ValueError):
                expect_exit = 0
            checks.append(
                DoDCheck(
                    type=str(raw_check.get("type") or "bash"),
                    command=command,
                    expect_exit=expect_exit,
                    description=str(raw_check.get("description") or ""),
                )
            )
        new_tasks.append(
            Task(
                id=task_id,
                title=str(raw_task.get("title") or task_id),
                instruction=str(raw_task.get("instruction") or ""),
                dod_checks=checks,
                depends_on=[
                    str(d) for d in raw_task.get("depends_on") or [] if d
                ],
            )
        )
    if not new_tasks:
        return 0
    # Replace pending tasks with the new ones.
    state.tasks = [t for t in state.tasks if t.state != "pending"] + new_tasks
    state.replan_count += 1
    state.consecutive_failures = 0
    return len(new_tasks)


def _next_runnable_task(state: QueueState) -> Task | None:
    completed = {t.id for t in state.tasks if t.state == "completed"}
    for task in state.tasks:
        if task.state != "pending":
            continue
        if all(dep in completed for dep in task.depends_on):
            return task
    return None


def _mark_unreachable_as_skipped(state: QueueState) -> None:
    completed = {t.id for t in state.tasks if t.state == "completed"}
    terminal = {"completed", "blocked", "failed", "skipped"}
    for task in state.tasks:
        if task.state != "pending":
            continue
        missing = [d for d in task.depends_on if d not in completed]
        if missing:
            upstream_terminal = {d: next((x.state for x in state.tasks if x.id == d), "unknown") for d in missing}
            if all(state_value in terminal for state_value in upstream_terminal.values()):
                task.state = "skipped"
                task.last_hint = (
                    "skipped: upstream not completed: "
                    + ", ".join(f"{d}={s}" for d, s in upstream_terminal.items())
                )


def _budget_exhausted(state: QueueState, started_wallclock: float) -> tuple[bool, str]:
    if WALLCLOCK_HOURS > 0:
        elapsed_h = (time.time() - started_wallclock) / 3600.0
        if elapsed_h >= WALLCLOCK_HOURS:
            return True, f"wall-clock budget exhausted ({elapsed_h:.2f}h >= {WALLCLOCK_HOURS}h)"
    if TOKEN_BUDGET > 0 and state.tokens_used >= TOKEN_BUDGET:
        return True, f"token budget exhausted ({state.tokens_used} >= {TOKEN_BUDGET})"
    return False, ""


def run_queue(state: QueueState, workdir: Path, script_dir: Path) -> int:
    total = len(state.tasks)
    started_wallclock = time.time()
    _record_run_event(
        workdir,
        {"type": "queue_started", "goal": state.goal, "tasks": total},
    )
    _touch_heartbeat(workdir)

    while True:
        exhausted, why = _budget_exhausted(state, started_wallclock)
        if exhausted:
            print(f"[queue] {why}; saving state and exiting.", flush=True)
            _record_run_event(workdir, {"type": "budget_exhausted", "reason": why})
            break

        task = _next_runnable_task(state)
        if task is None:
            break

        task.state = "in_progress"
        task.attempts += 1
        save_state(state, workdir)
        _touch_heartbeat(workdir)

        print(
            f"\n=== [{task.id}] attempt {task.attempts}/{MAX_ATTEMPTS}: {task.title} "
            f"(tokens_used={state.tokens_used}, elapsed={int(time.time() - started_wallclock)}s) ===",
            flush=True,
        )
        _record_run_event(
            workdir,
            {"type": "task_start", "task_id": task.id, "attempt": task.attempts},
        )
        log_path, timed_out = execute_task(
            task,
            workdir,
            script_dir,
            shared_memory=state.shared_memory,
            timeout_seconds=EXECUTOR_TIMEOUT,
        )
        task.last_log_path = str(log_path)
        _touch_heartbeat(workdir)

        print(f"[{task.id}] running {len(task.dod_checks)} DoD check(s)...", flush=True)
        check_results = run_all_checks(task.dod_checks, workdir)
        task.last_check_results = check_results
        all_passed = (
            all(r["passed"] for r in check_results) if check_results else False
        )
        for r in check_results:
            check = r["check"]
            status = "PASS" if r["passed"] else "FAIL"
            label = check.get("description") or check["command"]
            print(f"  [{status}] {label}", flush=True)

        final_text = extract_executor_final_text(log_path) or _fallback_log_tail(log_path)
        if timed_out:
            final_text = (final_text or "") + "\n[agent_queue] executor timed out."
        try:
            verdict, hint = judge_task(task, check_results, final_text)
        except Exception as exc:
            verdict, hint = "CONTINUE", f"judge exception: {exc}"
        # Judge itself also burns tokens; track approximately.
        state.tokens_used += JUDGE_MAX_TOKENS
        task.last_verdict = verdict
        task.last_hint = hint
        print(
            f"[{task.id}] judge: {verdict}{(' — ' + hint) if hint else ''}",
            flush=True,
        )
        _update_shared_memory(state, task, check_results, final_text, verdict)

        if verdict == "DONE" and all_passed:
            task.state = "completed"
            state.consecutive_failures = 0
        elif verdict == "BLOCKED":
            task.state = "blocked"
            state.consecutive_failures += 1
        elif task.attempts >= MAX_ATTEMPTS:
            task.state = "failed"
            state.consecutive_failures += 1
        else:
            task.state = "pending"

        _record_run_event(
            workdir,
            {
                "type": "task_end",
                "task_id": task.id,
                "attempt": task.attempts,
                "state": task.state,
                "verdict": verdict,
                "timed_out": timed_out,
                "tokens_used_so_far": state.tokens_used,
            },
        )
        save_state(state, workdir)
        _touch_heartbeat(workdir)

        if (
            state.consecutive_failures >= REPLAN_AFTER_FAILURES
            and state.replan_count < REPLAN_MAX
        ):
            print(
                f"[queue] {state.consecutive_failures} consecutive failures; "
                f"triggering dynamic replan ({state.replan_count + 1}/{REPLAN_MAX}).",
                flush=True,
            )
            replaced = _replan_remaining_tasks(state)
            _record_run_event(
                workdir,
                {"type": "replan", "round": state.replan_count, "replaced": replaced},
            )
            save_state(state, workdir)

    _mark_unreachable_as_skipped(state)
    save_state(state, workdir)

    duration = int(time.time() - started_wallclock)
    completed = [t for t in state.tasks if t.state == "completed"]
    blocked = [t for t in state.tasks if t.state == "blocked"]
    failed = [t for t in state.tasks if t.state == "failed"]
    skipped = [t for t in state.tasks if t.state == "skipped"]
    print(
        f"\n=== queue finished in {duration}s: "
        f"{len(completed)}/{total} completed, "
        f"{len(blocked)} blocked, {len(failed)} failed, {len(skipped)} skipped ===",
        flush=True,
    )
    for t in blocked + failed + skipped:
        reason = t.last_hint or t.last_verdict or "(no reason)"
        print(f"  [{t.state}] {t.id}: {t.title} — {reason}", flush=True)
    return 0 if not (blocked or failed or skipped) else 1


def print_status(state: QueueState) -> None:
    counts: dict[str, int] = {}
    for t in state.tasks:
        counts[t.state] = counts.get(t.state, 0) + 1
    print(f"Goal: {state.goal}")
    print(f"Workdir: {state.workdir}")
    print(f"Total tasks: {len(state.tasks)}")
    for name in ("pending", "in_progress", "completed", "blocked", "failed", "skipped"):
        if name in counts:
            print(f"  {name}: {counts[name]}")
    print()
    for t in state.tasks:
        extra = ""
        if t.last_verdict and t.state != "completed":
            extra = f" — {t.last_verdict}"
            if t.last_hint:
                extra += f": {t.last_hint}"
        print(
            f"[{t.state:11s}] {t.id}: {t.title} (attempts={t.attempts}){extra}"
        )


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_fallback_plan(path: Path) -> list[Task]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(raw_tasks, list):
        raise RuntimeError(f"fallback plan at {path} must have a top-level 'tasks' array")
    tasks: list[Task] = []
    for idx, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict):
            continue
        task_id = str(raw_task.get("id") or f"f{idx + 1}")
        checks: list[DoDCheck] = []
        for raw_check in raw_task.get("dod_checks") or []:
            if not isinstance(raw_check, dict):
                continue
            command = str(raw_check.get("command") or "").strip()
            if not command:
                continue
            try:
                expect_exit = int(raw_check.get("expect_exit", 0))
            except (TypeError, ValueError):
                expect_exit = 0
            checks.append(
                DoDCheck(
                    type=str(raw_check.get("type") or "bash"),
                    command=command,
                    expect_exit=expect_exit,
                    description=str(raw_check.get("description") or ""),
                )
            )
        tasks.append(
            Task(
                id=task_id,
                title=str(raw_task.get("title") or task_id),
                instruction=str(raw_task.get("instruction") or ""),
                dod_checks=checks,
                depends_on=[
                    str(d) for d in raw_task.get("depends_on") or [] if d
                ],
            )
        )
    if not tasks:
        raise RuntimeError(f"fallback plan at {path} contained no valid tasks")
    return tasks


def cmd_run(args: argparse.Namespace) -> int:
    workdir = Path(args.dir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if load_state(workdir) is not None:
        print(
            f"queue already exists at {_queue_dir(workdir)}; "
            f"use 'resume' or delete the directory to start fresh",
            file=sys.stderr,
        )
        return 2
    fallback_path: Path | None = None
    if getattr(args, "fallback_plan", None):
        fallback_path = Path(args.fallback_plan).expanduser()
        if not fallback_path.exists():
            print(f"fallback plan not found at {fallback_path}", file=sys.stderr)
            return 2
    ensure_local_server()
    tasks: list[Task]
    try:
        print(f"planning goal against {LOCAL_API_BASE} ...", flush=True)
        tasks = plan_tasks(args.goal)
    except RuntimeError as exc:
        if fallback_path is None:
            raise
        print(f"planner failed ({exc}); loading fallback plan from {fallback_path}", file=sys.stderr)
        tasks = _load_fallback_plan(fallback_path)
    print(f"planner produced {len(tasks)} task(s).", flush=True)
    state = QueueState(
        goal=args.goal,
        workdir=str(workdir),
        created_at=int(time.time()),
        updated_at=int(time.time()),
        tasks=tasks,
    )
    save_state(state, workdir)
    return run_queue(state, workdir, _script_dir())


def cmd_resume(args: argparse.Namespace) -> int:
    workdir = Path(args.dir).expanduser().resolve()
    state = load_state(workdir)
    if state is None:
        print(f"no queue at {_state_path(workdir)}", file=sys.stderr)
        return 2
    ensure_local_server()
    for task in state.tasks:
        if task.state == "in_progress":
            task.state = "pending"
    if args.retry_blocked:
        for task in state.tasks:
            if task.state in {"blocked", "failed", "skipped"}:
                task.state = "pending"
                task.attempts = 0
    if args.retry_failed:
        for task in state.tasks:
            if task.state == "failed":
                task.state = "pending"
                task.attempts = 0
    save_state(state, workdir)
    return run_queue(state, workdir, _script_dir())


def cmd_status(args: argparse.Namespace) -> int:
    workdir = Path(args.dir).expanduser().resolve()
    state = load_state(workdir)
    if state is None:
        print(f"no queue at {_state_path(workdir)}", file=sys.stderr)
        return 2
    print_status(state)
    return 0


def cmd_plan_only(args: argparse.Namespace) -> int:
    ensure_local_server()
    tasks = plan_tasks(args.goal)
    print(json.dumps(
        {"tasks": [dataclasses.asdict(t) for t in tasks]},
        indent=2,
        ensure_ascii=False,
    ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and execute a queue of sub-tasks with Definition-of-Done."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Plan a new queue and execute it")
    p_run.add_argument("--dir", required=True, help="Working directory for the run")
    p_run.add_argument(
        "--fallback-plan",
        default=None,
        help="Path to a pre-written JSON plan used if the planner fails to produce valid JSON.",
    )
    p_run.add_argument("goal", help="High-level goal")
    p_run.set_defaults(func=cmd_run)

    p_resume = sub.add_parser("resume", help="Resume an existing queue")
    p_resume.add_argument("--dir", required=True, help="Working directory")
    p_resume.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also re-queue tasks in 'failed' state",
    )
    p_resume.add_argument(
        "--retry-blocked",
        action="store_true",
        help="Also re-queue tasks in 'blocked', 'failed', and 'skipped' states",
    )
    p_resume.set_defaults(func=cmd_resume)

    p_status = sub.add_parser("status", help="Show queue status")
    p_status.add_argument("--dir", required=True, help="Working directory")
    p_status.set_defaults(func=cmd_status)

    p_plan = sub.add_parser("plan", help="Only run the planner and print JSON to stdout")
    p_plan.add_argument("goal", help="High-level goal")
    p_plan.set_defaults(func=cmd_plan_only)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
