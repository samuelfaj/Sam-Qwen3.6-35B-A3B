#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
import re
import time
import uuid
from threading import Condition, RLock, Thread, Timer
from typing import Any, Iterator, Literal

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from dflash.ddtree_engine import generate_ddtree
from dflash.model_mlx import (
    AdaptiveBlockSizeConfig,
    GenerationResponse,
    PromptPrefillState,
    clone_prefill_state_for_reuse,
    derive_prefill_prefix_state,
    estimate_memory_bytes,
    load,
    load_draft,
    prefill_prompt,
    stream_generate,
    tokenize_prompt,
)


_logger = logging.getLogger("dflash.server")


DEFAULT_MODEL_PATH = "/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit"
DEFAULT_DRAFT_PATH = "/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash"
DEFAULT_MODEL_NAME = "qwen3.6-35b-a3b-dflash-local"
DEFAULT_TRACE_FILE = os.environ.get("LOCAL_DFLASH_TRACE_FILE")
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
TAGGED_TOOL_CALL_RE = re.compile(
    r"<(?P<tag>tool_call|tool_calls|function_call|function_calls)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    re.DOTALL,
)
FENCED_TOOL_CALL_RE = re.compile(
    r"```(?P<tag>tool_call|tool_calls|function_call|function_calls)\s*(?P<body>.*?)```",
    re.DOTALL,
)
PARAM_RE = re.compile(
    r"<parameter=(?P<name>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)
SPECIAL_TOKENS = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
)
THINK_BLOCK_RE = re.compile(r"<think>\s*(?P<reasoning>.*?)\s*</think>\s*", re.DOTALL)
TOOL_BLOCK_MARKERS = (
    ("<tool_call>", "</tool_call>"),
    ("<tool_calls>", "</tool_calls>"),
    ("<function_call>", "</function_call>"),
    ("<function_calls>", "</function_calls>"),
    ("```tool_call", "```"),
    ("```tool_calls", "```"),
    ("```function_call", "```"),
    ("```function_calls", "```"),
)
VISIBLE_HIDDEN_MARKERS = (("<think>", "</think>"), *TOOL_BLOCK_MARKERS)
VISIBLE_START_MARKERS = tuple(marker for marker, _ in VISIBLE_HIDDEN_MARKERS)
VISIBLE_PARTIAL_MARKERS = (*SPECIAL_TOKENS, *VISIBLE_START_MARKERS)
TOOL_NORMALIZATION_CACHE_LIMIT = 128
_ANTHROPIC_TOOL_NORMALIZATION_CACHE: dict[str, list[dict[str, Any]]] = {}


def _env_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return default if value <= 0 else value


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return default if value <= 0 else value


def _env_non_negative_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return default if value < 0 else value


def _env_non_negative_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return default if value < 0 else value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


STREAM_HEARTBEAT_SECONDS = _env_positive_float("LOCAL_DFLASH_STREAM_HEARTBEAT_SECONDS", 1.0)
STREAM_QUEUE_MAX_CHUNKS = _env_positive_int("LOCAL_DFLASH_STREAM_QUEUE_MAX_CHUNKS", 32)
STREAM_QUEUE_PUT_TIMEOUT_SECONDS = _env_positive_float("LOCAL_DFLASH_STREAM_QUEUE_PUT_TIMEOUT_SECONDS", 0.5)
RESPONSE_HISTORY_LIMIT = _env_non_negative_int("LOCAL_DFLASH_RESPONSE_HISTORY_LIMIT", 8)
REQUEST_METRICS_HISTORY_LIMIT = _env_non_negative_int("LOCAL_DFLASH_REQUEST_METRICS_HISTORY_LIMIT", 50)
PREFIX_CACHE_STATE_LIMIT = _env_non_negative_int("LOCAL_DFLASH_PREFIX_CACHE_STATE_LIMIT", 1)
PREFIX_CACHE_STATE_BYTE_LIMIT = _env_non_negative_int(
    "LOCAL_DFLASH_PREFIX_CACHE_STATE_BYTE_LIMIT",
    2 * 1024 * 1024 * 1024,
)
GLOBAL_PREFIX_CACHE_LIMIT = _env_non_negative_int("LOCAL_DFLASH_GLOBAL_PREFIX_CACHE_LIMIT", 1)
MIN_TOOL_RESPONSE_MAX_TOKENS = _env_positive_int("LOCAL_DFLASH_MIN_TOOL_RESPONSE_MAX_TOKENS", 8192)
RESPONSES_ACTION_FOLLOWUP_LIMIT = _env_non_negative_int("LOCAL_DFLASH_RESPONSES_ACTION_FOLLOWUP_LIMIT", 2)
ALLOW_APPLY_PATCH_TOOL = _env_bool("LOCAL_DFLASH_ALLOW_APPLY_PATCH_TOOL", False)
RESPONSES_CONTINUE_PROMPT = (
    "[System: Continue the previous incomplete response now. "
    "Do not repeat prior acknowledgements, summaries, or bullet lists. "
    "Execute the required tool call immediately. "
    "If a tool call was cut off, re-emit it in full.]"
)
RESPONSES_TOOL_RESULT_PROMPT = (
    "[System: The previous tool call already completed and its result is available above. "
    "Use that result to continue. "
    "Do not immediately repeat the exact same tool call with identical arguments. "
    "If the tool output shows a failure, read the error and choose a different corrective action. "
    "If a scaffold command reports Operation cancelled because the target already exists, inspect the "
    "existing files and continue from them instead of running the scaffold command again. "
    "If a dev-server command such as npm run dev, vite, or next dev times out after printing a ready/local URL, "
    "treat the server startup as verified; do not retry it, and continue with build/check or final. "
    "After tests pass for a generated app, still run npm run build before final.]"
)
CHAT_TOOL_RESULT_PROMPT = (
    "[System: The previous tool call already completed and its result is available above. "
    "Use that result to continue. Do not immediately repeat the exact same tool call or shell command with identical arguments. "
    "If a file write command succeeded, move to the next different required file, package script, test, build, or verification step. "
    "If the user asked for tests, create test files, add a finite test script such as vitest run, and run it. "
    "After tests pass for a generated app, still run npm run build before final. "
    "If a command fails, read the error and choose a different corrective action.]"
)
RESPONSES_ACTION_PROMPT = (
    "[System: You just stated the next action but did not execute it. "
    "Do not repeat or re-emit the plan; do not call update_plan again on this turn. "
    "If files need to change, use an available editing-capable tool or shell command now. "
    "If a command must run, call the available shell tool now. "
    "Only produce a final text answer if no tool call is possible. "
    "Do not stop after announcing what you will do next. "
    "For generated apps, do not finish until you have run the relevant build/check command or identified a real blocker. "
    "Prefer finite commands such as npm run build or npm test -- --run over long-running dev servers such as npm run dev. "
    "For Vitest projects, use npm test -- --run or npx vitest run; do not run bare npm test because it can enter watch mode. "
    "If tests passed but npm run build has not run yet, run npm run build next.]"
)
RESPONSES_EMPTY_OUTPUT_PROMPT = (
    "[System: Your previous response was empty or too short to be actionable. "
    "The task is not complete. Continue the agentic workflow now. "
    "If work remains, call the appropriate tool immediately. "
    "If you need context, inspect files or run commands. "
    "Do not emit an empty response, a single token, or a final answer unless the task is actually complete.]"
)
PLANNING_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "update_plan",
        "plan",
        "todo_write",
        "todo",
        "apply_plan",
        "planner",
    }
)
RESPONSES_FOLLOWUP_JUDGE_ENABLED = _env_bool("LOCAL_DFLASH_FOLLOWUP_JUDGE", True)
RESPONSES_FOLLOWUP_JUDGE_MAX_TOKENS = _env_positive_int(
    "LOCAL_DFLASH_FOLLOWUP_JUDGE_MAX_TOKENS", 128
)
RESPONSES_FOLLOWUP_JUDGE_LOGPROB_MARGIN = _env_positive_float(
    "LOCAL_DFLASH_FOLLOWUP_JUDGE_LOGPROB_MARGIN", 1.0
)
RESPONSES_FOLLOWUP_JUDGE_LOGPROB_SYSTEM_PROMPT = (
    "You are a strict turn-completion checker for an AI coding agent. "
    "The agent has function-calling tools to read files, edit code, run commands, and inspect results. "
    "Decide whether the agent's latest turn is finished or whether it still needs to act.\n\n"
    "Reply with EXACTLY one uppercase letter and nothing else.\n"
    "- Y: the agent delivered the final answer, reported a genuine blocker, or asked a real clarifying question.\n"
    "- N: the agent described a next step, a fix, or an action it should perform (in any language) but did not actually call any tool.\n"
    "- N: the agent already called `update_plan` with the same plan as a prior turn but did not actually invoke an action tool (`apply_patch`, `shell`, `container.exec`, etc.).\n"
    "- N: the agent emitted `update_plan` alone when the user asked for concrete work and there are action tools available.\n"
    "When unsure, answer N."
)
RESPONSES_FOLLOWUP_JUDGE_JSON_SYSTEM_PROMPT = (
    "You are a strict turn-completion checker for an AI coding agent. "
    "The agent has function-calling tools to read files, edit code, run commands, and inspect results. "
    "Decide whether the agent's latest turn is finished or whether it still needs to act.\n\n"
    "Respond with ONLY a single JSON object and nothing else, matching this schema:\n"
    '{"reason":"<one short sentence>","verdict":"COMPLETE"|"INCOMPLETE"}\n\n'
    "- COMPLETE: the agent delivered the final answer, reported a genuine blocker, or asked a real clarifying question.\n"
    "- INCOMPLETE: the agent described a next step, a fix, or an action it should perform (in any language) but did not actually call any tool.\n"
    "- INCOMPLETE: the agent re-emitted the same `update_plan` plan as a prior turn without actually invoking any action tool.\n"
    "- INCOMPLETE: the agent emitted `update_plan` alone when concrete action tools were available and the user's request required executing something.\n\n"
    "When unsure, answer INCOMPLETE."
)
TOOL_CALLING_RULES_PROMPT = (
    "Tool-calling rules (strict):\n"
    "- Function calls MUST be wrapped in <tool_call>...</tool_call>. The JSON body must parse as a single object with keys \"name\" and \"arguments\".\n"
    "- Do NOT omit the opening <tool_call> tag, even after prose.\n"
    "- You MAY write one short reasoning line BEFORE a tool call, but NEVER after announcing an action without executing it.\n"
    "- NEVER say \"I will now <do X>\" or \"Next, I'll call <tool>\" without emitting the <tool_call> in the same turn.\n"
    "- If you have already emitted a plan via update_plan this turn, do NOT re-emit it; proceed directly to an available file-editing or shell tool.\n"
    "- Use the <cwd> from environment_context as the project root for shell workdir and file paths. Never create or edit user projects under CODEX_HOME, /tmp/codex-local-dflash, /private/tmp/codex-local-dflash, plugin directories, or skill directories unless the user explicitly asks.\n"
    "- Required parameters MUST be present. Do not emit tool calls with \"arguments\": {} unless the schema has zero required fields.\n"
    "- When a prior tool_response is available, USE IT. Do not re-issue the identical call with the same arguments.\n"
    "- When a command fails, read the error and change strategy; do not retry the same failing command unchanged.\n"
    "- If a project scaffold command reports Operation cancelled because the target exists, inspect and modify the existing project instead of scaffolding again.\n"
    "- For buildable apps, run the relevant build/check command before the final answer unless a real blocker prevents it.\n"
    "- Do not use long-running dev servers such as npm run dev, vite, or next dev as the final verification step; prefer finite build/check/test commands.\n"
    "- For Vitest projects, run npm test -- --run or npx vitest run. Do not run bare npm test because it can enter watch mode.\n"
    "- Produce a final natural-language answer ONLY when no tool is applicable or the user's task is fully complete.\n"
    "- Decide and act in the same turn."
)
TOOL_CALLING_RULES_APPLY_PATCH_PROMPT = (
    "\n- The apply_patch tool is available. Use it for precise repository file edits when appropriate."
)
TOOL_CALLING_RULES_ENABLED = _env_bool("LOCAL_DFLASH_TOOL_CALLING_RULES", True)


DEFAULT_TEMPERATURE_NO_TOOLS = _env_positive_float("LOCAL_DFLASH_DEFAULT_TEMPERATURE", 0.6)
DEFAULT_TEMPERATURE_WITH_TOOLS = _env_positive_float("LOCAL_DFLASH_DEFAULT_TEMPERATURE_WITH_TOOLS", 0.3)
DEFAULT_TOP_P = _env_positive_float("LOCAL_DFLASH_DEFAULT_TOP_P", 0.8)
DEFAULT_TOP_K = _env_non_negative_int("LOCAL_DFLASH_DEFAULT_TOP_K", 20)
DEFAULT_MIN_P = _env_non_negative_float("LOCAL_DFLASH_DEFAULT_MIN_P", 0.0)
# Official Qwen3.6-35B-A3B Instruct defaults per the model card: presence_penalty=1.5
# "to reduce endless repetitions", repetition_penalty=1.0 (disabled — the team
# explicitly relies on presence_penalty, not repetition_penalty).
DEFAULT_PRESENCE_PENALTY = _env_non_negative_float("LOCAL_DFLASH_DEFAULT_PRESENCE_PENALTY", 1.5)
DEFAULT_REPETITION_PENALTY = _env_non_negative_float("LOCAL_DFLASH_DEFAULT_REPETITION_PENALTY", 1.0)
DEFAULT_FREQUENCY_PENALTY = _env_non_negative_float("LOCAL_DFLASH_DEFAULT_FREQUENCY_PENALTY", 0.0)
DEFAULT_MAX_TOKENS_FALLBACK = _env_positive_int("LOCAL_DFLASH_DEFAULT_MAX_TOKENS", 8192)
MIN_TEMPERATURE_WITH_TOOLS = _env_non_negative_float("LOCAL_DFLASH_MIN_TEMPERATURE_WITH_TOOLS", 0.0)


def _coerce_sampling_arg(sampling: Any, temperature: float | None) -> "SamplingParams":
    """Back-compat helper: accept either a ready `SamplingParams` OR a plain
    `temperature` float on the old call sites. Any missing fields fall back
    to the no-tools defaults.
    """
    if isinstance(sampling, SamplingParams):
        return sampling
    if isinstance(sampling, (int, float)):
        return SamplingParams.for_request(
            temperature=float(sampling),
            top_p=None, top_k=None, min_p=None,
            presence_penalty=None, repetition_penalty=None, frequency_penalty=None,
            has_tools=False,
        )
    return SamplingParams.for_request(
        temperature=temperature,
        top_p=None, top_k=None, min_p=None,
        presence_penalty=None, repetition_penalty=None, frequency_penalty=None,
        has_tools=False,
    )


@dataclass
class SamplingParams:
    temperature: float = DEFAULT_TEMPERATURE_NO_TOOLS
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    min_p: float = DEFAULT_MIN_P
    presence_penalty: float = 0.0
    repetition_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_context_size: int = 20
    presence_context_size: int = 20

    @classmethod
    def for_request(
        cls,
        *,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        min_p: float | None,
        presence_penalty: float | None,
        repetition_penalty: float | None,
        frequency_penalty: float | None,
        has_tools: bool,
    ) -> "SamplingParams":
        if has_tools:
            temp = DEFAULT_TEMPERATURE_WITH_TOOLS if temperature is None else float(temperature)
            if temp < MIN_TEMPERATURE_WITH_TOOLS:
                temp = DEFAULT_TEMPERATURE_WITH_TOOLS
            pp = DEFAULT_PRESENCE_PENALTY if presence_penalty is None else float(presence_penalty)
            rp = DEFAULT_REPETITION_PENALTY if repetition_penalty is None else float(repetition_penalty)
        else:
            temp = DEFAULT_TEMPERATURE_NO_TOOLS if temperature is None else float(temperature)
            pp = 0.0 if presence_penalty is None else float(presence_penalty)
            rp = 0.0 if repetition_penalty is None else float(repetition_penalty)
        return cls(
            temperature=max(0.0, temp),
            top_p=DEFAULT_TOP_P if top_p is None else float(top_p),
            top_k=DEFAULT_TOP_K if top_k is None else int(top_k),
            min_p=DEFAULT_MIN_P if min_p is None else float(min_p),
            presence_penalty=pp,
            repetition_penalty=rp,
            frequency_penalty=DEFAULT_FREQUENCY_PENALTY if frequency_penalty is None else float(frequency_penalty),
        )


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] | str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    stream: bool = False
    keep_alive: str | int | float | None = None
    model_config = ConfigDict(extra="ignore")


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: Any = None
    tool_use_id: str | None = None
    content: Any = None
    is_error: bool | None = None
    model_config = ConfigDict(extra="allow")


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"] | str
    content: str | list[AnthropicContentBlock]


class AnthropicRequest(BaseModel):
    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS_FALLBACK
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    keep_alive: str | int | float | None = None
    model_config = ConfigDict(extra="ignore")


class AnthropicCountTokensRequest(BaseModel):
    model: str
    system: str | list[AnthropicContentBlock] | None = None
    messages: list[AnthropicMessage]
    model_config = ConfigDict(extra="ignore")


class ResponsesRequest(BaseModel):
    model: str
    input: str | list[Any]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    prompt_cache_key: str | None = None
    store: bool | None = None
    service_tier: str | None = None
    previous_response_id: str | None = None
    include: list[str] | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    client_metadata: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    truncation: Any = None
    keep_alive: str | int | float | None = None
    model_config = ConfigDict(extra="ignore")


class PromptTooLargeError(ValueError):
    pass


class UnknownPreviousResponseError(LookupError):
    pass


TRACE_ROTATE_MAX_BYTES = _env_positive_int("LOCAL_DFLASH_TRACE_ROTATE_MAX_BYTES", 100 * 1024 * 1024)
TRACE_ROTATE_MAX_AGE_SECONDS = _env_positive_int("LOCAL_DFLASH_TRACE_ROTATE_MAX_AGE_SECONDS", 4 * 60 * 60)
TRACE_ROTATE_KEEP = _env_non_negative_int("LOCAL_DFLASH_TRACE_ROTATE_KEEP", 5)
_TRACE_ROTATION_STATE: dict[str, Any] = {"opened_at": None}


def _maybe_rotate_trace_file(path: str) -> None:
    """Rotate the trace file when it grows past `TRACE_ROTATE_MAX_BYTES`
    or has been open for more than `TRACE_ROTATE_MAX_AGE_SECONDS`. Keeps the
    last `TRACE_ROTATE_KEEP` rotations.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    opened_at = _TRACE_ROTATION_STATE.get("opened_at")
    age_exceeded = opened_at is not None and (time.time() - opened_at) >= TRACE_ROTATE_MAX_AGE_SECONDS
    if size < TRACE_ROTATE_MAX_BYTES and not age_exceeded:
        if opened_at is None:
            _TRACE_ROTATION_STATE["opened_at"] = time.time()
        return
    # Shift .N → .N+1 up to keep limit; the oldest gets dropped.
    for idx in range(TRACE_ROTATE_KEEP - 1, 0, -1):
        src = f"{path}.{idx}"
        dst = f"{path}.{idx + 1}"
        try:
            if os.path.exists(src):
                os.replace(src, dst)
        except OSError:
            pass
    try:
        if os.path.exists(path):
            os.replace(path, f"{path}.1")
    except OSError:
        pass
    _TRACE_ROTATION_STATE["opened_at"] = time.time()


def _trace_event(kind: str, payload: dict[str, Any]) -> None:
    if not DEFAULT_TRACE_FILE:
        return
    event = {
        "ts": time.time(),
        "kind": kind,
        "payload": payload,
    }
    _maybe_rotate_trace_file(DEFAULT_TRACE_FILE)
    try:
        with open(DEFAULT_TRACE_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        _logger.warning("trace write failed: %s", exc)


def _trace_request(kind: str, payload: dict[str, Any]) -> None:
    _trace_event(kind, payload)


def _json_line(event: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"


def _data_line(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False)
    return f"data: {body}\n\n"


def _done_line() -> str:
    return "data: [DONE]\n\n"


def _make_stream_queue() -> Queue:
    return Queue(maxsize=max(1, STREAM_QUEUE_MAX_CHUNKS))


def _queue_put(queue: Queue, item: Any, stop_event: Any = None) -> bool:
    while True:
        if stop_event is not None:
            try:
                if stop_event.is_set():
                    return False
            except Exception:
                return False
        try:
            queue.put(item, timeout=STREAM_QUEUE_PUT_TIMEOUT_SECONDS)
            return True
        except Full:
            continue


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, BaseModel):
                item = item.model_dump(mode="json")
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"input_text", "output_text", "text", "thinking"}:
                parts.append(_coerce_text(item.get("text")))
            elif item_type == "tool_result":
                parts.append(_extract_text_from_content(item.get("content")))
            elif "text" in item:
                parts.append(_coerce_text(item.get("text")))
        return "".join(parts)
    if isinstance(content, BaseModel):
        return _extract_text_from_content(content.model_dump(mode="json"))
    if isinstance(content, dict):
        return _coerce_text(content.get("text") or content.get("content") or content)
    return _coerce_text(content)


def _coerce_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"input": stripped}
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    return {"input": value}


def _canonical_tool_arguments(value: Any) -> str:
    return json.dumps(_coerce_tool_arguments(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _make_tool_message(content: Any, *, tool_call_id: str | None = None, name: str | None = None) -> dict[str, Any]:
    message = {
        "role": "tool",
        "content": _extract_text_from_content(content),
    }
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    if name:
        message["name"] = name
    return message


def _parse_param_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _clean_output_text(text: str) -> str:
    cleaned = text
    for token in SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def _longest_partial_marker_suffix(text: str, markers: tuple[str, ...]) -> int:
    best = 0
    for marker in markers:
        upper = min(len(text), len(marker) - 1)
        for size in range(upper, 0, -1):
            if size <= best:
                break
            if text.endswith(marker[:size]):
                best = size
                break
    return best


def _next_visible_marker(text: str) -> tuple[int, str, str | None] | None:
    best: tuple[int, str, str | None] | None = None

    for marker in SPECIAL_TOKENS:
        idx = text.find(marker)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, marker, None)

    for marker, end_marker in VISIBLE_HIDDEN_MARKERS:
        idx = text.find(marker)
        if idx != -1 and (best is None or idx < best[0]):
            best = (idx, marker, end_marker)

    return best


class _IncrementalVisibleTextExtractor:
    def __init__(self) -> None:
        self._visible_buffer = ""
        self._hidden_buffer = ""
        self._hidden_end_marker: str | None = None

    def feed(self, text: str, *, final: bool = False) -> str:
        if text:
            if self._hidden_end_marker is None:
                self._visible_buffer += text
            else:
                self._hidden_buffer += text

        visible_parts: list[str] = []

        while True:
            if self._hidden_end_marker is not None:
                end_idx = self._hidden_buffer.find(self._hidden_end_marker)
                if end_idx == -1:
                    keep = 0 if final else _longest_partial_marker_suffix(
                        self._hidden_buffer,
                        (self._hidden_end_marker,),
                    )
                    self._hidden_buffer = self._hidden_buffer[-keep:] if keep > 0 else ""
                    break

                self._hidden_buffer = self._hidden_buffer[end_idx + len(self._hidden_end_marker):]
                self._hidden_end_marker = None
                self._visible_buffer = self._hidden_buffer + self._visible_buffer
                self._hidden_buffer = ""
                continue

            match = _next_visible_marker(self._visible_buffer)
            if match is None:
                keep = 0 if final else _longest_partial_marker_suffix(
                    self._visible_buffer,
                    VISIBLE_PARTIAL_MARKERS,
                )
                if keep > 0:
                    visible_parts.append(self._visible_buffer[:-keep])
                    self._visible_buffer = self._visible_buffer[-keep:]
                else:
                    visible_parts.append(self._visible_buffer)
                    self._visible_buffer = ""
                break

            idx, marker, end_marker = match
            if idx > 0:
                visible_parts.append(self._visible_buffer[:idx])
            self._visible_buffer = self._visible_buffer[idx + len(marker):]
            if end_marker is None:
                continue
            self._hidden_end_marker = end_marker
            self._hidden_buffer = self._visible_buffer
            self._visible_buffer = ""

        return "".join(visible_parts)


class _IncrementalVisibleTextStream:
    def __init__(self, *, strip_edges: bool) -> None:
        self._extractor = _IncrementalVisibleTextExtractor()
        self._strip_edges = strip_edges
        self._emitted_non_whitespace = False
        self._trailing_whitespace = ""

    def _strip_delta(self, text: str) -> str:
        if not text:
            return ""

        if not self._emitted_non_whitespace:
            text = text.lstrip()
            if not text:
                return ""
            self._emitted_non_whitespace = True

        if self._trailing_whitespace:
            text = self._trailing_whitespace + text
            self._trailing_whitespace = ""

        stripped = text.rstrip()
        trailing_len = len(text) - len(stripped)
        if trailing_len > 0:
            self._trailing_whitespace = text[-trailing_len:]
            text = stripped

        return text

    def feed(self, text: str, *, final: bool = False) -> str:
        delta = self._extractor.feed(text, final=final)
        if not self._strip_edges:
            return delta
        return self._strip_delta(delta)


def _extract_visible_text(text: str) -> str:
    extractor = _IncrementalVisibleTextExtractor()
    return extractor.feed(text, final=True)


def _strip_reasoning_blocks(text: str) -> tuple[str, str]:
    cleaned = _clean_output_text(text)
    reasoning_parts: list[str] = []

    if "</think>" in cleaned:
        leading_reasoning, cleaned = cleaned.split("</think>", 1)
        leading_reasoning = leading_reasoning.replace("<think>", "")
        leading_reasoning = _clean_output_text(leading_reasoning)
        if leading_reasoning:
            reasoning_parts.append(leading_reasoning)
        cleaned = _clean_output_text(cleaned)

    def _replace(match: re.Match[str]) -> str:
        reasoning = _clean_output_text(match.group("reasoning"))
        if reasoning:
            reasoning_parts.append(reasoning)
        return ""

    visible = THINK_BLOCK_RE.sub(_replace, cleaned)
    return "\n\n".join(reasoning_parts), _clean_output_text(visible)


def _make_function_call_item(
    name: str,
    arguments: Any,
    *,
    call_id: str | None = None,
    item_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id or f"fc_{uuid.uuid4().hex}",
        "call_id": call_id or f"call_{uuid.uuid4().hex}",
        "name": name,
        "arguments": json.dumps(_coerce_tool_arguments(arguments), ensure_ascii=False),
        "status": "completed",
    }


def _make_custom_tool_call_item(
    name: str,
    raw_input: str,
    *,
    call_id: str | None = None,
    item_id: str | None = None,
) -> dict[str, Any]:
    """Emit Codex's `custom_tool_call` shape for tools registered as `type:"custom"`
    (e.g. freeform `apply_patch`). The raw payload travels in `input` as-is;
    Codex does NOT JSON-decode it."""
    return {
        "type": "custom_tool_call",
        "id": item_id or f"ctc_{uuid.uuid4().hex}",
        "call_id": call_id or f"call_{uuid.uuid4().hex}",
        "name": name,
        "input": raw_input if isinstance(raw_input, str) else json.dumps(raw_input, ensure_ascii=False),
        "status": "completed",
    }


def _custom_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    """Collect the names of every tool that was registered with `type:"custom"`."""
    if not tools:
        return set()
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type != "custom":
            continue
        candidate = tool.get("name")
        if not candidate:
            fn = tool.get("function")
            if isinstance(fn, dict):
                candidate = fn.get("name")
        if candidate:
            names.add(str(candidate))
    return names


def _convert_items_for_custom_tools(
    items: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Rewrite `function_call` items into `custom_tool_call` when the named
    tool was registered with `type:"custom"`. Required by Codex 0.122 for
    freeform `apply_patch`: sending a `function_call` for a custom tool makes
    `ResponseItem` deserialization fail and the turn stalls."""
    custom_names = _custom_tool_names(tools)
    if not custom_names:
        return items
    converted: list[dict[str, Any]] = []
    for item in items:
        if (
            item.get("type") == "function_call"
            and str(item.get("name") or "") in custom_names
        ):
            raw_args = item.get("arguments")
            raw_input: str
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    raw_input = raw_args
                else:
                    if isinstance(parsed, dict):
                        for key in ("input", "patch", "text", "content"):
                            if key in parsed and isinstance(parsed[key], str):
                                raw_input = parsed[key]
                                break
                        else:
                            raw_input = raw_args
                    else:
                        raw_input = raw_args
            else:
                raw_input = json.dumps(raw_args, ensure_ascii=False)
            converted.append(
                _make_custom_tool_call_item(
                    name=str(item.get("name") or ""),
                    raw_input=raw_input,
                    call_id=item.get("call_id"),
                    item_id=item.get("id"),
                )
            )
        else:
            converted.append(item)
    return converted


def _tool_call_items_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return []
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return []

    if isinstance(payload, list):
        items: list[dict[str, Any]] = []
        for entry in payload:
            items.extend(_tool_call_items_from_payload(entry))
        return items

    if not isinstance(payload, dict):
        return []

    nested_keys = ("tool_calls", "function_calls", "calls")
    for key in nested_keys:
        if key in payload:
            return _tool_call_items_from_payload(payload.get(key))

    function_payload = payload.get("function")
    if isinstance(function_payload, dict):
        name = (
            function_payload.get("name")
            or payload.get("name")
            or payload.get("tool_name")
            or payload.get("recipient_name")
            or payload.get("action")
        )
        arguments = function_payload.get("arguments")
        if arguments is None:
            arguments = payload.get("arguments")
        if arguments is None:
            arguments = payload.get("input")
        if arguments is None:
            arguments = payload.get("parameters")
        if name:
            return [
                _make_function_call_item(
                    str(name),
                    arguments,
                    call_id=payload.get("call_id") or payload.get("id"),
                    item_id=payload.get("item_id"),
                )
            ]

    name = (
        payload.get("name")
        or payload.get("tool_name")
        or payload.get("recipient_name")
        or payload.get("action")
    )
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("input")
    if arguments is None:
        arguments = payload.get("parameters")

    if name and (arguments is not None or payload.get("type") in {"function_call", "custom_tool_call", "tool_use"}):
        return [
            _make_function_call_item(
                str(name),
                arguments,
                call_id=payload.get("call_id") or payload.get("id"),
                item_id=payload.get("item_id"),
            )
        ]

    return []


def _parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    cleaned = _clean_output_text(text)
    # Track consumed (start, end) spans so we don't double-parse a single
    # `<tool_call>` that contains both nested `<function=...>` XML (Qwen3-Coder)
    # and a bare JSON body the outer TAGGED pattern also matches.
    consumed_spans: list[tuple[int, int]] = []
    parsed_blocks: list[tuple[int, int, list[dict[str, Any]]]] = []

    def _span_overlaps(start: int, end: int) -> bool:
        for s, e in consumed_spans:
            if start < e and end > s:
                return True
        return False

    for match in TOOL_CALL_RE.finditer(cleaned):
        params: dict[str, Any] = {}
        for param_match in PARAM_RE.finditer(match.group("body")):
            params[param_match.group("name").strip()] = _parse_param_value(param_match.group("value"))
        consumed_spans.append((match.start(), match.end()))
        parsed_blocks.append(
            (
                match.start(),
                match.end(),
                [
                    _make_function_call_item(
                        match.group("name").strip(),
                        params,
                    )
                ],
            )
        )

    for match in TAGGED_TOOL_CALL_RE.finditer(cleaned):
        if _span_overlaps(match.start(), match.end()):
            continue
        parsed = _tool_call_items_from_payload(match.group("body"))
        if parsed:
            consumed_spans.append((match.start(), match.end()))
            parsed_blocks.append((match.start(), match.end(), parsed))

    for match in FENCED_TOOL_CALL_RE.finditer(cleaned):
        if _span_overlaps(match.start(), match.end()):
            continue
        parsed = _tool_call_items_from_payload(match.group("body"))
        if parsed:
            consumed_spans.append((match.start(), match.end()))
            parsed_blocks.append((match.start(), match.end(), parsed))

    tool_calls: list[dict[str, Any]] = []
    for _, _, items in sorted(parsed_blocks, key=lambda entry: entry[0]):
        tool_calls.extend(items)

    visible_text = _extract_visible_text(cleaned)
    return _clean_output_text(visible_text), tool_calls


def _make_internal_tool_call(name: str, arguments: Any, *, call_id: str | None = None) -> dict[str, Any]:
    tool_call_id = call_id or f"call_{uuid.uuid4().hex}"
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _coerce_tool_arguments(arguments),
        }
    }


def _approx_tokens_bytes(tokens: Any) -> int:
    """Rough byte accounting for a tuple/list of ints. 8 bytes / token is
    more than a 32-bit int actually costs on CPython but leaves headroom for
    the surrounding tuple header."""
    try:
        return 8 * len(tokens)
    except Exception:
        return 0


def _response_usage(result: dict[str, Any]) -> dict[str, Any]:
    input_tokens = int(result.get("prompt_tokens") or 0)
    output_tokens = int(result.get("generated_tokens") or 0)
    cached_input_tokens = int(result.get("reused_prefix_tokens") or 0)
    reasoning_output_tokens = int(result.get("reasoning_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "input_tokens_details": {
            "cached_tokens": cached_input_tokens,
        },
        "output_tokens_details": {
            "reasoning_tokens": reasoning_output_tokens,
        },
    }


def _response_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "prefill_seconds": result["prefill_seconds"],
        "decode_seconds": result["decode_seconds"],
        "prompt_tps": result["prompt_tps"],
        "generation_tps": result["generation_tps"],
        "reused_prefix_tokens": result.get("reused_prefix_tokens", 0),
        "prefix_cache_source": result.get("prefix_cache_source", "none"),
        "speculative_steps": result.get("speculative_steps", 0),
        "proposed_tokens": result.get("proposed_tokens", 0),
        "accepted_tokens": result.get("accepted_tokens", 0),
        "avg_acceptance_length": result.get("avg_acceptance_length", 0.0),
        "avg_acceptance_ratio": result.get("avg_acceptance_ratio", 0.0),
        "acceptance_lengths": result.get("acceptance_lengths", []),
        "acceptance_ratios": result.get("acceptance_ratios", []),
        "block_size_history": result.get("block_size_history", []),
        "adaptive_block_size": result.get("adaptive_block_size", False),
        "peak_memory_gb": result["peak_memory_gb"],
        "elapsed": result["elapsed"],
    }


def _has_unterminated_tool_call_markup(text: str) -> bool:
    cleaned = _clean_output_text(text)

    if not cleaned:
        return False

    paired_markers = (
        ("<tool_call>", "</tool_call>"),
        ("<tool_calls>", "</tool_calls>"),
        ("<function_call>", "</function_call>"),
        ("<function_calls>", "</function_calls>"),
    )
    for start_marker, end_marker in paired_markers:
        if cleaned.count(start_marker) > cleaned.count(end_marker):
            return True

    if len(re.findall(r"<function=[^>\n]+>", cleaned)) > cleaned.count("</function>"):
        return True

    if len(re.findall(r"<parameter=[^>\n]+>", cleaned)) > cleaned.count("</parameter>"):
        return True

    fenced_tags = ("tool_call", "tool_calls", "function_call", "function_calls")
    for tag in fenced_tags:
        start_marker = f"```{tag}"
        start_idx = cleaned.rfind(start_marker)
        if start_idx == -1:
            continue
        if cleaned.find("```", start_idx + 3) == -1:
            return True

    return False


def _response_completion_state(result: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    if _has_unterminated_tool_call_markup(result.get("text", "")):
        return "incomplete", {"reason": "truncated_tool_call"}

    finish_reason = _coerce_text(result.get("finish_reason")).strip().lower()
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        return "incomplete", {"reason": "max_output_tokens"}

    return "completed", None


def _classify_error_code(message: str, exc: Exception | None = None) -> str:
    """Map a raw error message/exception to a Codex-recognized error code.

    Codex parses `error.code` to decide retry behavior; without a recognized
    code it falls back to Retryable which can mask real failures and cause
    infinite retry loops.
    """
    if exc is not None:
        if isinstance(exc, PromptTooLargeError):
            return "context_length_exceeded"
        if isinstance(exc, UnknownPreviousResponseError):
            return "invalid_prompt"
    lowered = (message or "").lower()
    if "context" in lowered and ("length" in lowered or "window" in lowered):
        return "context_length_exceeded"
    if "prompt" in lowered and ("too large" in lowered or "too long" in lowered):
        return "context_length_exceeded"
    if "quota" in lowered or "rate limit" in lowered:
        return "rate_limit_exceeded"
    if "oom" in lowered or "out of memory" in lowered or "insufficient memory" in lowered:
        return "server_overloaded"
    if "invalid prompt" in lowered or "invalid input" in lowered:
        return "invalid_prompt"
    return "server_error"


def _is_planning_only_function_call(item: dict[str, Any]) -> bool:
    if item.get("type") not in {"function_call", "custom_tool_call"}:
        return False
    name = str(item.get("name") or "").strip().lower()
    return name in PLANNING_ONLY_TOOL_NAMES


def _response_is_followup_candidate(
    result: dict[str, Any],
    output_items: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> bool:
    """Cheap gate: returns True only if it is worth asking the model-based judge."""
    if not tools:
        return False

    response_status, _ = _response_completion_state(result)
    if response_status != "completed":
        return False

    has_action_call = any(
        item.get("type") in {"function_call", "custom_tool_call"}
        and not _is_planning_only_function_call(item)
        for item in output_items
    )
    if has_action_call:
        return False

    has_planning_call = any(
        _is_planning_only_function_call(item) for item in output_items
    )
    assistant_text = _output_text_from_items(output_items).strip()
    if not assistant_text and not has_planning_call:
        return False

    return True


def _response_needs_empty_output_followup(
    result: dict[str, Any],
    output_items: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> bool:
    if not tools:
        return False
    response_status, _ = _response_completion_state(result)
    if response_status != "completed":
        return False
    if any(item.get("type") in {"function_call", "custom_tool_call"} for item in output_items):
        return False
    assistant_text = _output_text_from_items(output_items).strip()
    generated_tokens = int(result.get("generated_tokens", 0) or 0)
    return generated_tokens <= 2 or len(assistant_text) <= 2


def _assistant_text_promises_action(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return False

    promise_patterns = (
        r"\blet me\b",
        r"\bi(?:'ll| will)\b",
        r"\bi(?:'m| am) going to\b",
        r"\bi need to\b",
        r"\bnow i(?:'ll| will)\b",
        r"\bnext i(?:'ll| will)\b",
        r"\bvou\b",
        r"\bvou agora\b",
        r"\bdeixe-me\b",
        r"\bagora vou\b",
    )
    if not any(re.search(pattern, normalized) for pattern in promise_patterns):
        return False

    action_patterns = (
        r"\b(add|build|building|check|create|edit|explore|fix|implement|inspect|install|modify|read|run|start|test|update|verify|write)\b",
        r"\b(criar|editar|executar|implementar|instalar|ler|modificar|rodar|verificar)\b",
    )
    return any(re.search(pattern, normalized) for pattern in action_patterns)


def _chat_result_needs_action_followup(
    result: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    messages: list[dict[str, Any]] | None = None,
) -> bool:
    if not tools:
        return False
    if str(result.get("finish_reason") or "").strip() != "stop":
        return False
    visible_text, tool_calls = _parse_tool_calls(_coerce_text(result.get("text", "")))
    if tool_calls:
        return False
    return _assistant_text_promises_action(
        visible_text
    ) or _task_history_requires_completion_followup(messages or [], visible_text)


def _task_history_requires_completion_followup(
    messages: list[dict[str, Any]],
    assistant_text: str,
) -> bool:
    last_user_text = _last_user_message_text(messages).lower()
    if not last_user_text:
        return False
    blob = (
        json.dumps(messages, ensure_ascii=False, default=str)
        + "\n"
        + assistant_text
    ).lower()
    wants_buildable_app = any(
        marker in last_user_text
        for marker in ("react", "vite", "typescript", "app", "game")
    )
    if wants_buildable_app and not any(
        marker in blob for marker in ("npm run build", "vite build")
    ):
        return True
    wants_tests = "test" in last_user_text
    if wants_tests and not any(
        marker in blob
        for marker in ("vitest run", "npm test -- --run", "npm run test -- --run")
    ):
        return True
    return False


def _massage_chat_tool_result_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return messages
    last = messages[-1]
    if str(last.get("role") or "").lower() != "tool":
        return messages
    return [
        *messages,
        {
            "role": "user",
            "content": CHAT_TOOL_RESULT_PROMPT,
        },
    ]


def _last_user_message_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = _extract_text_from_content(message.get("content"))
        if text.strip():
            return text
    return ""


def _summarize_tool_names(tools: list[dict[str, Any]] | None) -> str:
    if not tools:
        return "(none)"
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        candidate = tool.get("name")
        if not candidate:
            fn = tool.get("function")
            if isinstance(fn, dict):
                candidate = fn.get("name")
        if candidate:
            names.append(str(candidate))
    return ", ".join(names) if names else "(none)"


def _tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    candidate = tool.get("name")
    if not candidate:
        fn = tool.get("function")
        if isinstance(fn, dict):
            candidate = fn.get("name")
    return str(candidate) if candidate else None


def _available_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    if not tools:
        return set()
    names: set[str] = set()
    for tool in tools:
        candidate = _tool_name(tool)
        if candidate:
            names.add(candidate)
    return names


def _filter_disabled_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if ALLOW_APPLY_PATCH_TOOL:
        return tools
    return [tool for tool in tools if _tool_name(tool) != "apply_patch"]


def _filter_disabled_tool_calls(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if ALLOW_APPLY_PATCH_TOOL:
        return items
    filtered: list[dict[str, Any]] = []
    for item in items:
        if item.get("type") in {"function_call", "custom_tool_call"} and item.get("name") == "apply_patch":
            continue
        filtered.append(item)
    return filtered


def _tool_calling_rules_prompt(tools: list[dict[str, Any]] | None) -> str:
    prompt = TOOL_CALLING_RULES_PROMPT
    if "apply_patch" in _available_tool_names(tools):
        prompt += TOOL_CALLING_RULES_APPLY_PATCH_PROMPT
    return prompt


def _responses_action_prompt(tools: list[dict[str, Any]] | None) -> str:
    prompt = RESPONSES_ACTION_PROMPT
    if "apply_patch" in _available_tool_names(tools):
        prompt += " [System: The apply_patch tool is available for precise file edits.]"
    return prompt


def _first_encoded_token_id(tokenizer: Any, text: str) -> int | None:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokens = tokenizer.encode(text)
    if not tokens:
        return None
    return int(tokens[0])


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        candidate = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return candidate if isinstance(candidate, dict) else None


def _make_message_item(text: str) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def _output_text_from_items(items: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for item in items:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                texts.append(_coerce_text(content.get("text")))
    return "".join(texts)


def _make_reasoning_item(summary_text: str) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": f"rs_{uuid.uuid4().hex}",
        "summary": [
            {
                "type": "summary_text",
                "text": summary_text,
            }
        ],
    }


def _build_output_items(full_text: str) -> list[dict[str, Any]]:
    reasoning_text, visible_text = _strip_reasoning_blocks(full_text)
    assistant_text, tool_calls = _parse_tool_calls(visible_text)
    tool_calls = _filter_disabled_tool_calls(tool_calls)
    deduped_tool_calls: list[dict[str, Any]] = []
    previous_signature: tuple[str, str] | None = None
    for tool_call in tool_calls:
        signature = (
            _coerce_text(tool_call.get("name") or "tool"),
            _canonical_tool_arguments(tool_call.get("arguments")),
        )
        if signature == previous_signature:
            continue
        deduped_tool_calls.append(tool_call)
        previous_signature = signature
    items: list[dict[str, Any]] = []
    if reasoning_text:
        items.append(_make_reasoning_item(reasoning_text))
    if assistant_text:
        items.append(_make_message_item(assistant_text))
    items.extend(deduped_tool_calls)
    if not items:
        items.append(_make_message_item(""))
    return items


def _compact_output_items_for_replay(output_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in output_items:
        item_type = item.get("type")
        if item_type == "message":
            compact.append(_make_message_item(_extract_text_from_content(item.get("content"))))
            continue
        if item_type == "function_call":
            compact.append(
                _make_function_call_item(
                    str(item.get("name") or "tool"),
                    item.get("arguments"),
                    call_id=item.get("call_id"),
                    item_id=item.get("id"),
                )
            )
            continue
        if item_type == "custom_tool_call":
            compact.append(
                _make_custom_tool_call_item(
                    str(item.get("name") or "tool"),
                    _coerce_text(item.get("input")),
                    call_id=item.get("call_id"),
                    item_id=item.get("id"),
                )
            )
    if not compact:
        compact.append(_make_message_item(""))
    return compact


def _messages_from_output_items(output_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in output_items:
        item_type = item.get("type")
        if item_type == "message":
            messages.append(
                {
                    "role": "assistant",
                    "content": _extract_text_from_content(item.get("content")),
                }
            )
            continue

        if item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _make_internal_tool_call(
                            item.get("name") or "tool",
                            item.get("arguments"),
                            call_id=item.get("call_id"),
                        )
                    ],
                }
            )
            continue

        if item_type == "custom_tool_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _make_internal_tool_call(
                            item.get("name") or "tool",
                            {"input": _coerce_text(item.get("input"))},
                            call_id=item.get("call_id"),
                        )
                    ],
                }
            )

    return messages


def _merge_message_context(
    base_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    system_parts: list[str] = []

    base_idx = 0
    while base_idx < len(base_messages) and base_messages[base_idx].get("role") == "system":
        content = _coerce_text(base_messages[base_idx].get("content")).strip()
        if content:
            system_parts.append(content)
        base_idx += 1

    new_idx = 0
    while new_idx < len(new_messages) and new_messages[new_idx].get("role") == "system":
        content = _coerce_text(new_messages[new_idx].get("content")).strip()
        if content:
            system_parts.append(content)
        new_idx += 1

    merged: list[dict[str, Any]] = []
    if system_parts:
        merged.append({"role": "system", "content": "\n\n".join(system_parts)})
    merged.extend(base_messages[base_idx:])
    merged.extend(new_messages[new_idx:])
    return merged


def _massage_responses_continuation_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages or messages[-1].get("role") != "assistant":
        return messages

    trailing_start = len(messages) - 1
    while trailing_start > 0:
        candidate = messages[trailing_start - 1]
        if candidate.get("role") != "assistant":
            break
        if candidate.get("tool_calls"):
            break
        trailing_start -= 1

    trailing_assistants = messages[trailing_start:]
    if not trailing_assistants:
        return messages

    if any(message.get("tool_calls") for message in trailing_assistants):
        return messages

    last_assistant = copy.deepcopy(trailing_assistants[-1])
    continue_message = {
        "role": "user",
        "content": RESPONSES_CONTINUE_PROMPT,
    }
    return [*messages[:trailing_start], last_assistant, continue_message]


def _massage_responses_tool_result_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages or messages[-1].get("role") != "tool":
        return messages

    tool_result_start = len(messages) - 1
    while tool_result_start > 0 and messages[tool_result_start - 1].get("role") == "tool":
        tool_result_start -= 1

    if tool_result_start == 0:
        return messages

    preceding_message = messages[tool_result_start - 1]
    if preceding_message.get("role") != "assistant" or not preceding_message.get("tool_calls"):
        return messages

    return [
        *messages,
        {
            "role": "user",
            "content": RESPONSES_TOOL_RESULT_PROMPT,
        },
    ]


def _synthesize_orphan_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Qwen3 chat templates REQUIRE every assistant `tool_calls` block to be
    followed by a matching `tool` / `<tool_response>` message. If the last
    assistant message emits tool_calls with no tool_result, Qwen drops into a
    prefill-only mode (no `<|im_start|>assistant` generation prompt), which
    surfaces as the "empty-args loop" from turn 3+.

    This function injects synthetic `{error: "tool_result_missing"}` responses
    for any dangling tool_calls, keeping the conversation template valid so
    the model resumes generation correctly.
    """
    if not messages:
        return messages
    last = messages[-1]
    if last.get("role") != "assistant":
        return messages
    tool_calls = last.get("tool_calls") or []
    if not tool_calls:
        return messages
    patched = list(messages)
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        call_id = call.get("id") or call.get("call_id")
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = call.get("name") or (fn.get("name") if isinstance(fn, dict) else None)
        patched.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps({"error": "tool_result_missing"}, ensure_ascii=False),
            }
        )
    patched.append(
        {
            "role": "user",
            "content": (
                "[System: The previous assistant turn emitted tool calls that were not "
                "delivered to the environment. Treat the tool results above as authoritative. "
                "Do not repeat the identical tool calls; decide the next action based on the error.]"
            ),
        }
    )
    return patched


def _leading_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        idx += 1
    return messages[:idx]


def _responses_max_tokens(requested_max_tokens: int | None, tools: list[dict[str, Any]] | None) -> int:
    max_tokens = requested_max_tokens or DEFAULT_MAX_TOKENS_FALLBACK
    if tools:
        max_tokens = max(max_tokens, MIN_TOOL_RESPONSE_MAX_TOKENS)
    return max_tokens


def _longest_common_prefix_tokens(left: list[int], right: list[int]) -> tuple[int, ...]:
    size = min(len(left), len(right))
    idx = 0
    while idx < size and left[idx] == right[idx]:
        idx += 1
    return tuple(left[:idx])


def _shared_prefix_length(left: list[int], right: tuple[int, ...]) -> int:
    size = min(len(left), len(right))
    idx = 0
    while idx < size and left[idx] == right[idx]:
        idx += 1
    return idx


def _prompt_startswith(prompt_tokens: list[int], prefix_tokens: tuple[int, ...]) -> bool:
    if len(prefix_tokens) > len(prompt_tokens):
        return False
    return tuple(prompt_tokens[: len(prefix_tokens)]) == prefix_tokens


def _hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    cache_key = _hash_json_payload(tools)
    cached = _ANTHROPIC_TOOL_NORMALIZATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name") or tool.get("tool_name")
        if not name:
            continue
        function_def: dict[str, Any] = {
            "name": name,
            "parameters": tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}},
        }
        description = _coerce_text(tool.get("description")).strip()
        if description:
            function_def["description"] = description
        normalized.append(
            {
                "type": "function",
                "function": function_def,
            }
        )
    if len(_ANTHROPIC_TOOL_NORMALIZATION_CACHE) >= TOOL_NORMALIZATION_CACHE_LIMIT:
        _ANTHROPIC_TOOL_NORMALIZATION_CACHE.pop(next(iter(_ANTHROPIC_TOOL_NORMALIZATION_CACHE)), None)
    _ANTHROPIC_TOOL_NORMALIZATION_CACHE[cache_key] = normalized
    return normalized


def _normalize_anthropic_messages(req: AnthropicRequest | AnthropicCountTokensRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    tools = _normalize_anthropic_tools(getattr(req, "tools", None))

    system_text = _extract_text_from_content(req.system)
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for msg in req.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
            continue

        if msg.role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in msg.content:
                block_data = block.model_dump(mode="json")
                block_type = block_data.get("type")
                if block_type == "tool_use":
                    tool_calls.append(
                        _make_internal_tool_call(
                            block_data.get("name") or "tool",
                            block_data.get("input"),
                            call_id=block_data.get("id"),
                        )
                    )
                    continue
                text = _extract_text_from_content(block_data)
                if text:
                    text_parts.append(text)

            if text_parts or tool_calls:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(text_parts),
                }
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)
            continue

        text_parts = []
        for block in msg.content:
            block_data = block.model_dump(mode="json")
            block_type = block_data.get("type")
            if block_type == "tool_result":
                if text_parts:
                    messages.append({"role": msg.role, "content": "".join(text_parts)})
                    text_parts = []
                messages.append(
                    _make_tool_message(
                        block_data.get("content"),
                        tool_call_id=block_data.get("tool_use_id"),
                        name=block_data.get("name"),
                    )
                )
                continue

            text = _extract_text_from_content(block_data)
            if text:
                text_parts.append(text)

        if text_parts:
            messages.append({"role": msg.role, "content": "".join(text_parts)})

    return messages, tools


def _anthropic_stop_reason(result: dict[str, Any], content_blocks: list[dict[str, Any]]) -> str:
    if any(block.get("type") == "tool_use" for block in content_blocks):
        return "tool_use"
    finish_reason = result.get("finish_reason")
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "stop_sequence":
        return "stop_sequence"
    if finish_reason == "refusal":
        return "refusal"
    return "end_turn"


def _build_anthropic_content_blocks(full_text: str) -> list[dict[str, Any]]:
    _, visible_text = _strip_reasoning_blocks(full_text)
    assistant_text, tool_calls = _parse_tool_calls(visible_text)
    content: list[dict[str, Any]] = []
    if assistant_text:
        content.append(
            {
                "type": "text",
                "text": assistant_text,
            }
        )
    for tool_call in tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": tool_call["call_id"],
                "name": tool_call["name"],
                "input": _coerce_tool_arguments(tool_call["arguments"]),
            }
        )
    if not content:
        content.append(
            {
                "type": "text",
                "text": "",
            }
        )
    return content


def _build_anthropic_message_payload(
    message_id: str,
    model_name: str,
    result: dict[str, Any],
    content_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": content_blocks,
        "stop_reason": _anthropic_stop_reason(result, content_blocks),
        "stop_sequence": None,
        "usage": {
            "input_tokens": result["prompt_tokens"],
            "output_tokens": result["generated_tokens"],
        },
        "metrics": _response_metrics(result),
    }


def _build_response_payload(
    response_id: str,
    model_name: str,
    result: dict[str, Any],
    output_items: list[dict[str, Any]],
    status: str,
    previous_response_id: str | None = None,
    incomplete_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model_name,
        "output": output_items,
        "output_text": _output_text_from_items(output_items),
        "previous_response_id": previous_response_id,
        "usage": _response_usage(result),
        "metrics": _response_metrics(result),
    }
    if incomplete_details is not None:
        payload["incomplete_details"] = incomplete_details
    return payload


def _model_detail_payload(server: "LocalModelServer") -> dict[str, Any]:
    context_length = server.context_window
    max_tokens = server.max_tokens_limit
    return {
        "id": server.model_name,
        "object": "model",
        "owned_by": "local",
        "context_length": context_length,
        "max_model_len": context_length,
        "max_tokens": max_tokens,
    }


def _lm_studio_model_payload(server: "LocalModelServer") -> dict[str, Any]:
    context_length = server.context_window
    model = _model_detail_payload(server)
    return {
        "id": server.model_name,
        "key": server.model_name,
        "context_length": context_length,
        "max_context_length": context_length,
        "loaded_instances": [
            {
                "identifier": server.model_name,
                "config": {
                    "context_length": context_length,
                },
            }
        ],
        **model,
    }


def _ollama_model_payload(server: "LocalModelServer") -> dict[str, Any]:
    context_length = server.context_window
    return {
        "name": server.model_name,
        "model": server.model_name,
        "modified_at": None,
        "size": 0,
        "digest": "local-dflash",
        "details": {
            "context_length": context_length,
            "max_model_len": context_length,
        },
    }


def _llamacpp_props_payload(server: "LocalModelServer") -> dict[str, Any]:
    context_length = server.context_window
    max_tokens = server.max_tokens_limit
    return {
        "model_path": server.model_path,
        "model": server.model_name,
        "default_generation_settings": {
            "n_ctx": context_length,
            "context_length": context_length,
            "n_predict": max_tokens,
        },
        "context_length": context_length,
        "chat_template": "qwen-tool-use",
    }


def _detect_context_window(model_path: str) -> int | None:
    config_path = os.path.join(model_path, "config.json")
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")

    for path in (config_path, tokenizer_config_path):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload.get("text_config"), dict):
            value = payload["text_config"].get("max_position_embeddings")
            if isinstance(value, int) and value > 0:
                return value
        for key in ("max_position_embeddings", "model_max_length"):
            value = payload.get(key)
            if isinstance(value, int) and value > 0:
                return value

    return None


def _parse_keep_alive(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if value < 0 else float(value)

    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"inf", "infinite", "forever"}:
        return None

    try:
        numeric = float(text)
    except ValueError:
        match = re.fullmatch(r"(-?\d+(?:\.\d+)?)([smh])", text)
        if not match:
            raise ValueError(f"Unsupported keep-alive value: {value}")
        numeric = float(match.group(1))
        unit = match.group(2)
        if unit == "m":
            numeric *= 60
        elif unit == "h":
            numeric *= 3600

    return None if numeric < 0 else numeric


def _gb_to_bytes(limit_gb: float | None) -> int | None:
    if limit_gb is None:
        return None
    if limit_gb < 0:
        return None
    return int(limit_gb * (1024 ** 3))


def _comment_line(comment: str = "heartbeat") -> str:
    return f": {comment}\n\n"


def _responses_heartbeat_line(response_id: str, model_name: str) -> str:
    # Typed heartbeat for the Responses SSE stream. Codex's `stream_idle_timeout`
    # resets on any SSE event frame, NOT on SSE comment lines, so emitting a
    # `response.in_progress` event (which Codex already handles) keeps the
    # stream alive during long prefills / tool-call generations.
    payload = {
        "type": "response.in_progress",
        "response": {
            "id": response_id,
            "object": "response",
            "status": "in_progress",
            "model": model_name,
        },
    }
    return _json_line("response.in_progress", payload)


def _chat_heartbeat_line(completion_id: str, created: int, model_name: str) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": None,
            }
        ],
    }
    return _data_line(payload)


def _anthropic_heartbeat_line() -> str:
    # Anthropic Messages SSE has a first-class `ping` event.
    return _json_line("ping", {"type": "ping"})


def _normalize_responses_input(req: ResponsesRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    tools = _filter_disabled_tools(req.tools or [])
    system_parts: list[str] = []

    if req.instructions:
        system_parts.append(req.instructions)

    # Inject the "Tool-calling rules (strict)" block whenever tools are
    # registered. This single block removes most of Qwen3's "plan-and-announce"
    # failure mode in agentic Codex runs. Opt out via
    # LOCAL_DFLASH_TOOL_CALLING_RULES=0 for diagnostics.
    if tools and TOOL_CALLING_RULES_ENABLED:
        system_parts.append(_tool_calling_rules_prompt(tools))

    if isinstance(req.input, str):
        messages.append({"role": "user", "content": req.input})
        return messages, tools

    for item in req.input:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        role = item.get("role", "user")
        if role == "developer":
            role = "system"

        if item_type == "message":
            content = _extract_text_from_content(item.get("content"))
            if role == "system":
                if content:
                    system_parts.append(content)
            else:
                messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )
            continue

        if item_type in {"function_call", "custom_tool_call"}:
            tool_name = item.get("name") or item.get("tool_name") or item.get("action") or "tool"
            call_id = item.get("call_id") or item.get("tool_call_id") or item.get("id")
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _make_internal_tool_call(
                            tool_name,
                            item.get("arguments"),
                            call_id=call_id,
                        )
                    ],
                }
            )
            continue

        if item_type in {"function_call_output", "custom_tool_call_output"}:
            output = item.get("output")
            if not output:
                output = item.get("content")
            messages.append(
                _make_tool_message(
                    output,
                    tool_call_id=item.get("call_id") or item.get("tool_call_id") or item.get("id"),
                    name=item.get("name") or item.get("tool_name"),
                )
            )
            continue

        if item_type == "reasoning":
            # Preserve any encrypted_content / summary text so it can be
            # replayed if we ever gain access to the target model's cached
            # thinking. For now we keep the item as a hint in the assistant
            # history (wrapped in <think>…</think>) so Qwen sees the previous
            # deliberation when continuing a multi-turn reasoning turn.
            summary_parts: list[str] = []
            for content_block in item.get("summary") or []:
                if isinstance(content_block, dict):
                    text = content_block.get("text")
                    if text:
                        summary_parts.append(str(text))
            for content_block in item.get("content") or []:
                if isinstance(content_block, dict):
                    text = content_block.get("text")
                    if text:
                        summary_parts.append(str(text))
            encrypted = item.get("encrypted_content")
            if encrypted and not summary_parts:
                # Nothing readable to splice in; drop it (agrees with prior behavior).
                continue
            if summary_parts:
                think_block = "<think>\n" + "\n\n".join(summary_parts) + "\n</think>"
                messages.append({"role": "assistant", "content": think_block})
            continue

        content = _extract_text_from_content(item.get("content"))
        if content or role == "assistant":
            messages.append({"role": role, "content": content})

    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(part for part in system_parts if part)})

    return messages, tools


class LocalModelServer:
    def __init__(
        self,
        model_path: str,
        draft_path: str,
        model_name: str,
        block_size: int,
        disable_thinking: bool,
        sliding_window_size: int | None,
        max_tokens_limit: int | None,
        context_window: int | None,
        context_reserve: int,
        keep_alive_seconds: float | None,
        target_turboquant_bits: float | None,
        rotating_keep_tokens: int = 0,
        draft_turboquant_bits: float | None = None,
        adaptive_block_size_config: AdaptiveBlockSizeConfig | None = None,
        generation_engine: str = "dflash",
        ddtree_tree_budget: int = 4,
        ddtree_target_turboquant_bits: float | None = None,
        ddtree_fallback_to_dflash: bool = True,
        ddtree_retry_without_turboquant: bool = True,
        global_prefix_cache_limit: int = GLOBAL_PREFIX_CACHE_LIMIT,
        prefix_cache_state_byte_limit: int = PREFIX_CACHE_STATE_BYTE_LIMIT,
        global_prefix_cache_byte_limit: int | None = None,
        stable_prefix_tokens_byte_limit: int | None = None,
        mlx_clear_cache_threshold: float | None = 0.9,
    ) -> None:
        self.model_path = model_path
        self.draft_path = draft_path
        self.model_name = model_name
        self.block_size = block_size
        self.disable_thinking = disable_thinking
        self.sliding_window_size = sliding_window_size
        self.rotating_keep_tokens = max(0, int(rotating_keep_tokens or 0))
        self.max_tokens_limit = max_tokens_limit
        self.context_window = context_window
        self.context_reserve = context_reserve
        self.keep_alive_seconds = keep_alive_seconds
        self.target_turboquant_bits = (
            None if target_turboquant_bits is not None and target_turboquant_bits <= 0 else target_turboquant_bits
        )
        self.draft_turboquant_bits = (
            None if draft_turboquant_bits is not None and draft_turboquant_bits <= 0 else draft_turboquant_bits
        )
        normalized_engine = (generation_engine or "dflash").strip().lower()
        if normalized_engine not in {"dflash", "ddtree"}:
            raise ValueError(f"Unsupported generation_engine: {generation_engine}")
        self.generation_engine = normalized_engine
        self.ddtree_tree_budget = max(1, int(ddtree_tree_budget or 1))
        self.ddtree_target_turboquant_bits = (
            None
            if ddtree_target_turboquant_bits is not None and ddtree_target_turboquant_bits <= 0
            else ddtree_target_turboquant_bits
        )
        self.ddtree_fallback_to_dflash = bool(ddtree_fallback_to_dflash)
        self.ddtree_retry_without_turboquant = bool(ddtree_retry_without_turboquant)
        self._ddtree_target_turboquant_failed = False
        self.adaptive_block_size_config = adaptive_block_size_config or AdaptiveBlockSizeConfig(
            enabled=False,
            min_block_size=max(1, block_size),
            max_block_size=max(1, block_size),
        )
        self._lock = RLock()
        self._model = None
        self._draft = None
        self._tokenizer = None
        self._unload_timer: Timer | None = None
        self._last_used_at: float | None = None
        self._response_states: dict[str, dict[str, Any]] = {}
        self._response_order: deque[str] = deque()
        self._request_metrics_order: deque[dict[str, Any]] = deque(maxlen=max(1, REQUEST_METRICS_HISTORY_LIMIT))
        self._last_request_metrics: dict[str, Any] | None = None
        self._prefix_state_order: deque[str] = deque()
        self.response_history_limit = RESPONSE_HISTORY_LIMIT
        self.prefix_cache_state_limit = PREFIX_CACHE_STATE_LIMIT
        self.prefix_cache_state_byte_limit = max(0, int(prefix_cache_state_byte_limit or 0))
        self.global_prefix_cache_limit = global_prefix_cache_limit
        self._global_prefix_states: dict[str, PromptPrefillState] = {}
        self._global_prefix_order: deque[str] = deque()
        self._stable_prefix_tokens_by_key: dict[str, tuple[int, ...]] = {}
        self._global_prefix_cache_hits = 0
        self._global_prefix_cache_misses = 0
        self._generation_turn = Condition()
        self._next_generation_ticket = 0
        self._active_generation_ticket: int | None = None
        self._queued_generation_tickets: deque[int] = deque()
        self.mlx_clear_cache_threshold = mlx_clear_cache_threshold
        self.global_prefix_cache_byte_limit = max(0, int(global_prefix_cache_byte_limit or 0))
        self.stable_prefix_tokens_byte_limit = max(0, int(stable_prefix_tokens_byte_limit or 0))
        self._global_prefix_cache_bytes = 0
        self._global_prefix_state_bytes: dict[str, int] = {}
        self._stable_prefix_tokens_bytes = 0
        self._response_prefix_cache_bytes = 0
        self._prefix_state_bytes: dict[str, int] = {}

    def _clear_hidden_states_locked(self) -> None:
        hidden_states = getattr(self._model, "_hidden_states", None)
        if isinstance(hidden_states, list):
            hidden_states[:] = [None] * len(hidden_states)

    def _acquire_generation_turn(self) -> int:
        with self._generation_turn:
            ticket = self._next_generation_ticket
            self._next_generation_ticket += 1
            self._queued_generation_tickets.append(ticket)
            while self._active_generation_ticket is not None or self._queued_generation_tickets[0] != ticket:
                self._generation_turn.wait()
            self._queued_generation_tickets.popleft()
            self._active_generation_ticket = ticket
            return ticket

    def _release_generation_turn(self, ticket: int | None) -> None:
        if ticket is None:
            return
        with self._generation_turn:
            if self._active_generation_ticket == ticket:
                self._active_generation_ticket = None
            self._generation_turn.notify_all()

    def _record_generation_metrics(self, result: dict[str, Any], *, surface: str) -> None:
        metrics = {
            "surface": surface,
            "finished_at": time.time(),
            "finish_reason": result.get("finish_reason", "stop"),
            "elapsed": float(result.get("elapsed", 0.0) or 0.0),
            "prefill_seconds": float(result.get("prefill_seconds", 0.0) or 0.0),
            "decode_seconds": float(result.get("decode_seconds", 0.0) or 0.0),
            "prompt_tps": float(result.get("prompt_tps", 0.0) or 0.0),
            "generation_tps": float(result.get("generation_tps", 0.0) or 0.0),
            "prompt_tokens": int(result.get("prompt_tokens", 0) or 0),
            "generated_tokens": int(result.get("generated_tokens", 0) or 0),
            "reused_prefix_tokens": int(result.get("reused_prefix_tokens", 0) or 0),
            "speculative_steps": int(result.get("speculative_steps", 0) or 0),
            "proposed_tokens": int(result.get("proposed_tokens", 0) or 0),
            "accepted_tokens": int(result.get("accepted_tokens", 0) or 0),
            "avg_acceptance_length": float(result.get("avg_acceptance_length", 0.0) or 0.0),
            "avg_acceptance_ratio": float(result.get("avg_acceptance_ratio", 0.0) or 0.0),
            "peak_memory_gb": float(result.get("peak_memory_gb", 0.0) or 0.0),
            "prefix_cache_source": result.get("prefix_cache_source", "none"),
            "engine": result.get("engine", self.generation_engine),
            "ddtree_commit": result.get("ddtree_commit"),
            "tree_budget": int(result.get("tree_budget", 0) or 0),
            "target_turboquant_bits": float(result.get("target_turboquant_bits", 0.0) or 0.0),
            "ddtree_fast_path_ratio": float(result.get("ddtree_fast_path_ratio", 0.0) or 0.0),
            "avg_tree_node_count": float(result.get("avg_tree_node_count", 0.0) or 0.0),
            "max_tree_node_count": int(result.get("max_tree_node_count", 0) or 0),
        }
        phase_timings = result.get("ddtree_phase_timings_us") or {}
        if isinstance(phase_timings, dict):
            phase_total = 0.0
            for key, value in phase_timings.items():
                try:
                    numeric = float(value or 0.0)
                except Exception:
                    numeric = 0.0
                metrics[f"ddtree_phase_{key}_us"] = numeric
                phase_total += numeric
            metrics["ddtree_phase_total_us"] = phase_total
        block_size_history = result.get("block_size_history") or []
        if block_size_history:
            metrics["final_block_size"] = int(block_size_history[-1])
            metrics["min_observed_block_size"] = int(min(block_size_history))
            metrics["max_observed_block_size"] = int(max(block_size_history))
        with self._lock:
            self._last_request_metrics = metrics
            self._request_metrics_order.append(metrics)

    def _clear_request_state_locked(self) -> None:
        self._clear_hidden_states_locked()

    def _drop_prefix_cache_state_locked(self, response_id: str) -> None:
        state = self._response_states.get(response_id)
        if state is not None:
            state["prompt_cache_state"] = None
        stale_bytes = self._prefix_state_bytes.pop(response_id, 0)
        self._response_prefix_cache_bytes = max(
            0,
            self._response_prefix_cache_bytes - stale_bytes,
        )

    def _clear_cached_prefix_states_locked(self) -> None:
        while self._prefix_state_order:
            response_id = self._prefix_state_order.popleft()
            self._drop_prefix_cache_state_locked(response_id)

    def _clear_global_prefix_cache_locked(self) -> None:
        self._global_prefix_states.clear()
        self._global_prefix_order.clear()
        self._stable_prefix_tokens_by_key.clear()
        self._global_prefix_state_bytes.clear()
        self._global_prefix_cache_bytes = 0
        self._stable_prefix_tokens_bytes = 0

    def _reset_loaded_state_locked(self) -> None:
        self._clear_hidden_states_locked()
        self._clear_cached_prefix_states_locked()
        self._clear_global_prefix_cache_locked()
        self._model = None
        self._draft = None
        self._tokenizer = None
        gc.collect()
        mx.clear_cache()

    def _cancel_unload_timer_locked(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _unload_from_timer(self, scheduled_timer: Timer) -> None:
        # Timer callbacks run in background threads. Grab the lock and only
        # proceed if the pending timer is still us — otherwise a newer
        # `finish_request` has already rescheduled and we must not double-
        # unload.
        with self._lock:
            if self._unload_timer is not scheduled_timer:
                return
            self._cancel_unload_timer_locked()
            self._reset_loaded_state_locked()

    def unload(self) -> None:
        with self._lock:
            self._cancel_unload_timer_locked()
            self._reset_loaded_state_locked()

    def _schedule_unload_locked(self, keep_alive_seconds: float | None) -> None:
        # Called under `self._lock`. Creating and starting the Timer here
        # (rather than in a separate method) avoids a race where two
        # concurrent `finish_request` calls each fire a fresh Timer.
        self._cancel_unload_timer_locked()
        if self._model is None:
            return
        if keep_alive_seconds is None:
            return
        if keep_alive_seconds <= 0:
            self._reset_loaded_state_locked()
            return

        timer: Timer | None = None

        def _fire() -> None:
            if timer is not None:
                self._unload_from_timer(timer)

        timer = Timer(keep_alive_seconds, _fire)
        timer.daemon = True
        self._unload_timer = timer
        timer.start()

    def finish_request(self, keep_alive_override: Any = None) -> None:
        keep_alive_seconds = (
            self.keep_alive_seconds if keep_alive_override is None else _parse_keep_alive(keep_alive_override)
        )
        with self._lock:
            self._last_used_at = time.time()
            # Keep the warm model resident, but drop per-request tensors and free-list
            # allocations so memory returns close to the loaded baseline after each turn.
            self._clear_request_state_locked()
            if keep_alive_seconds is not None and keep_alive_seconds <= 0:
                self._clear_global_prefix_cache_locked()
            self._maybe_clear_mlx_cache_locked()
            self._schedule_unload_locked(keep_alive_seconds)

    def _maybe_clear_mlx_cache_locked(self) -> None:
        """Called between requests. When the Metal allocator cache exceeds
        `mlx_clear_cache_threshold * cache_limit`, flush it so long-running
        24h sessions don't fragment into OOM. Also resets the peak-memory
        counter so `/metrics` reports the recent window, not lifetime.
        """
        threshold = getattr(self, "mlx_clear_cache_threshold", None)
        if threshold is None or threshold <= 0:
            return
        try:
            cache_bytes = int(mx.get_cache_memory())
            cache_limit_bytes = int(mx.metal.get_cache_limit()) if hasattr(mx, "metal") and hasattr(mx.metal, "get_cache_limit") else 0
        except Exception:
            cache_bytes = 0
            cache_limit_bytes = 0
        if cache_limit_bytes <= 0:
            return
        if cache_bytes >= threshold * cache_limit_bytes:
            try:
                mx.clear_cache()
                if hasattr(mx, "reset_peak_memory"):
                    mx.reset_peak_memory()
                elif hasattr(mx.metal, "reset_peak_memory"):
                    mx.metal.reset_peak_memory()
            except Exception as exc:
                _logger.debug("mx.clear_cache() failed: %s", exc)

    def ensure_loaded(self) -> None:
        self._cancel_unload_timer_locked()
        if self._model is not None and self._draft is not None and self._tokenizer is not None:
            return
        if any(component is not None for component in (self._model, self._draft, self._tokenizer)):
            _logger.warning("partial loaded state detected; resetting model and prefix caches before reload")
            self._reset_loaded_state_locked()
        elif self._global_prefix_states or self._stable_prefix_tokens_by_key or self._global_prefix_state_bytes:
            _logger.warning("stale prefix cache detected without a loaded model; clearing cached prefix state")
            self._clear_cached_prefix_states_locked()
            self._clear_global_prefix_cache_locked()
        self._model, self._tokenizer = load(self.model_path)
        self._draft = load_draft(
            self.draft_path,
            sliding_window_size=self.sliding_window_size,
            turboquant_bits=self.draft_turboquant_bits,
            rotating_keep_tokens=self.rotating_keep_tokens,
        )
        # Make sure the Qwen3 stop tokens `<|im_end|>` and `<|endoftext|>` are
        # in the EOS set. Most Qwen3 chat tokenizers only expose `<|im_end|>`
        # by default, which means a runaway <|endoftext|> isn't caught. We do
        # NOT add `</tool_call>` — the agent audit flagged that as a footgun
        # (it would chop off the closing tag and trip unterminated-call logic).
        try:
            existing = set(self._tokenizer.eos_token_ids or [])
            for literal in ("<|im_end|>", "<|endoftext|>"):
                try:
                    tok = self._tokenizer.encode(literal, add_special_tokens=False)
                except TypeError:
                    tok = self._tokenizer.encode(literal)
                if tok and len(tok) == 1:
                    existing.add(int(tok[0]))
            self._tokenizer.eos_token_ids = sorted(existing)
        except Exception as exc:
            _logger.debug("failed to augment eos_token_ids: %s", exc)

    def _prune_prefix_cache_states_locked(self) -> None:
        while self._prefix_state_order and (
            len(self._prefix_state_order) > self.prefix_cache_state_limit
            or (
                self.prefix_cache_state_byte_limit > 0
                and self._response_prefix_cache_bytes > self.prefix_cache_state_byte_limit
            )
        ):
            stale_response_id = self._prefix_state_order.popleft()
            self._drop_prefix_cache_state_locked(stale_response_id)

    def _prune_global_prefix_states_locked(self) -> None:
        while len(self._global_prefix_order) > self.global_prefix_cache_limit:
            stale_key = self._global_prefix_order.popleft()
            self._global_prefix_states.pop(stale_key, None)
            stale_bytes = self._global_prefix_state_bytes.pop(stale_key, 0)
            self._global_prefix_cache_bytes = max(0, self._global_prefix_cache_bytes - stale_bytes)
            stale_tokens = self._stable_prefix_tokens_by_key.pop(stale_key, None)
            if stale_tokens is not None:
                self._stable_prefix_tokens_bytes = max(
                    0,
                    self._stable_prefix_tokens_bytes - _approx_tokens_bytes(stale_tokens),
                )
        if self.global_prefix_cache_byte_limit > 0:
            while (
                self._global_prefix_cache_bytes > self.global_prefix_cache_byte_limit
                and self._global_prefix_order
            ):
                stale_key = self._global_prefix_order.popleft()
                self._global_prefix_states.pop(stale_key, None)
                stale_bytes = self._global_prefix_state_bytes.pop(stale_key, 0)
                self._global_prefix_cache_bytes = max(0, self._global_prefix_cache_bytes - stale_bytes)
                stale_tokens = self._stable_prefix_tokens_by_key.pop(stale_key, None)
                if stale_tokens is not None:
                    self._stable_prefix_tokens_bytes = max(
                        0,
                        self._stable_prefix_tokens_bytes - _approx_tokens_bytes(stale_tokens),
                    )

    def _prune_stable_prefix_tokens_locked(self) -> None:
        if self.stable_prefix_tokens_byte_limit <= 0:
            return
        while (
            self._stable_prefix_tokens_bytes > self.stable_prefix_tokens_byte_limit
            and self._stable_prefix_tokens_by_key
        ):
            # Drop the first-inserted key we still have.
            try:
                stale_key = next(iter(self._stable_prefix_tokens_by_key))
            except StopIteration:
                break
            stale_tokens = self._stable_prefix_tokens_by_key.pop(stale_key, None)
            if stale_tokens is not None:
                self._stable_prefix_tokens_bytes = max(
                    0,
                    self._stable_prefix_tokens_bytes - _approx_tokens_bytes(stale_tokens),
                )

    def _prune_response_states_locked(self) -> None:
        while len(self._response_order) > self.response_history_limit:
            stale_response_id = self._response_order.popleft()
            stale_state = self._response_states.pop(stale_response_id, None)
            self._drop_prefix_cache_state_locked(stale_response_id)
            stale_parent_id = stale_state.get("previous_response_id") if stale_state is not None else None
            stale_tools = stale_state.get("tools") or [] if stale_state is not None else []
            for state in self._response_states.values():
                if state.get("previous_response_id") == stale_response_id:
                    state["previous_response_id"] = stale_parent_id
                    if stale_tools and not state.get("tools"):
                        state["tools"] = list(stale_tools)
            try:
                self._prefix_state_order.remove(stale_response_id)
            except ValueError:
                pass

    def _conversation_from_response_locked(
        self,
        response_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        states: list[dict[str, Any]] = []
        cursor = response_id
        seen: set[str] = set()

        while cursor:
            if cursor in seen:
                raise UnknownPreviousResponseError(f"Response history loop detected for {response_id}")
            seen.add(cursor)

            state = self._response_states.get(cursor)
            if state is None:
                raise UnknownPreviousResponseError(
                    f"Unknown previous_response_id: {response_id}. "
                    "This local server only remembers in-process response history."
                )

            states.append(state)
            cursor = state.get("previous_response_id")

        states.reverse()

        messages: list[dict[str, Any]] = []
        tools: list[dict[str, Any]] = []
        for state in states:
            messages = _merge_message_context(messages, state.get("input_messages", []))
            messages.extend(_messages_from_output_items(state.get("output_items", [])))
            state_tools = state.get("tools") or []
            if state_tools:
                tools = state_tools

        return messages, tools

    def resolve_responses_context(
        self,
        request_messages: list[dict[str, Any]],
        request_tools: list[dict[str, Any]],
        previous_response_id: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not previous_response_id:
            return request_messages, request_tools

        with self._lock:
            base_messages, inherited_tools = self._conversation_from_response_locked(previous_response_id)

        messages = _merge_message_context(base_messages, request_messages)
        tools = request_tools or inherited_tools
        return messages, tools

    def remember_response(
        self,
        response_id: str,
        previous_response_id: str | None,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        output_items: list[dict[str, Any]],
        prompt_cache_state: PromptPrefillState | None = None,
    ) -> None:
        with self._lock:
            stored_prompt_cache_state = prompt_cache_state if self.prefix_cache_state_limit > 0 else None
            stored_prompt_cache_state_bytes = 0
            if stored_prompt_cache_state is not None:
                try:
                    stored_prompt_cache_state_bytes = int(estimate_memory_bytes(stored_prompt_cache_state))
                except Exception:
                    stored_prompt_cache_state_bytes = 0
                if (
                    self.prefix_cache_state_byte_limit > 0
                    and stored_prompt_cache_state_bytes > self.prefix_cache_state_byte_limit
                ):
                    stored_prompt_cache_state = None
                    stored_prompt_cache_state_bytes = 0
            previous_tools = []
            if previous_response_id:
                previous_state = self._response_states.get(previous_response_id)
                if previous_state is not None:
                    previous_tools = previous_state.get("tools") or []
            stored_tools = list(tools) if tools and previous_tools != tools else []
            self._drop_prefix_cache_state_locked(response_id)
            self._response_states[response_id] = {
                "previous_response_id": previous_response_id,
                "input_messages": list(request_messages),
                "tools": stored_tools,
                "output_items": _compact_output_items_for_replay(output_items),
                "prompt_cache_state": stored_prompt_cache_state,
            }
            self._response_order.append(response_id)
            if stored_prompt_cache_state is not None:
                self._prefix_state_bytes[response_id] = stored_prompt_cache_state_bytes
                self._response_prefix_cache_bytes += stored_prompt_cache_state_bytes
                self._prefix_state_order.append(response_id)
                self._prune_prefix_cache_states_locked()
            self._prune_response_states_locked()

    def _prefix_cache_state_for_response_locked(self, response_id: str | None) -> PromptPrefillState | None:
        if not response_id:
            return None
        state = self._response_states.get(response_id)
        if state is None:
            return None
        prompt_cache_state = state.get("prompt_cache_state")
        if prompt_cache_state is None:
            return None
        return prompt_cache_state

    def build_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            kwargs["tools"] = tools
        enable_thinking = not self.disable_thinking
        # `preserve_thinking=True` keeps the last assistant `<think>` block on
        # Qwen3 thinking-mode templates; stripping it from turn 3+ causes the
        # known empty-args tool-call loop. Older templates don't accept the
        # kwarg, so we fall back in stages.
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                preserve_thinking=True,
                **kwargs,
            )
        except TypeError:
            pass
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def tokenize_prompt(self, prompt: str) -> list[int]:
        return tokenize_prompt(self._tokenizer, prompt).tolist()

    def _stable_prefix_key(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        stable_messages = _leading_system_messages(messages)
        if not stable_messages and not tools:
            return "stable-prefix::empty"
        payload = {
            "messages": stable_messages,
            "tools": tools or [],
        }
        return f"stable-prefix::{_hash_json_payload(payload)}"

    def _stable_prefix_tokens_locked(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[int, ...]:
        if self.global_prefix_cache_limit <= 0:
            return ()
        stable_prefix_key = self._stable_prefix_key(messages, tools=tools)
        if stable_prefix_key is None:
            return ()
        cached_tokens = self._stable_prefix_tokens_by_key.get(stable_prefix_key)
        if cached_tokens is not None:
            return cached_tokens

        stable_messages = _leading_system_messages(messages)
        probe_a = stable_messages + [{"role": "user", "content": "__LOCAL_DFLASH_STABLE_PREFIX_A__"}]
        probe_b = stable_messages + [{"role": "user", "content": "__LOCAL_DFLASH_STABLE_PREFIX_B__"}]
        prompt_a = self.build_prompt(probe_a, tools=tools, add_generation_prompt=False)
        prompt_b = self.build_prompt(probe_b, tools=tools, add_generation_prompt=False)
        prefix_tokens = _longest_common_prefix_tokens(
            self.tokenize_prompt(prompt_a),
            self.tokenize_prompt(prompt_b),
        )
        existing = self._stable_prefix_tokens_by_key.get(stable_prefix_key)
        if existing is not None:
            self._stable_prefix_tokens_bytes = max(
                0, self._stable_prefix_tokens_bytes - _approx_tokens_bytes(existing)
            )
        self._stable_prefix_tokens_by_key[stable_prefix_key] = prefix_tokens
        self._stable_prefix_tokens_bytes += _approx_tokens_bytes(prefix_tokens)
        self._prune_stable_prefix_tokens_locked()
        return prefix_tokens

    def _global_prefix_state_for_key_locked(self, stable_prefix_key: str | None) -> PromptPrefillState | None:
        if stable_prefix_key is None:
            return None
        state = self._global_prefix_states.get(stable_prefix_key)
        if state is None:
            return None
        return state

    def _reusable_prefix_state_for_prompt(
        self,
        prompt_tokens: list[int],
        prefix_state: PromptPrefillState | None,
    ) -> tuple[PromptPrefillState | None, int]:
        if prefix_state is None:
            return None, 0
        shared_tokens = _shared_prefix_length(prompt_tokens, prefix_state.prompt_tokens)
        if shared_tokens <= 0:
            return None, 0

        cached_tokens = len(prefix_state.prompt_tokens)
        prompt_length = len(prompt_tokens)
        if shared_tokens == cached_tokens:
            if shared_tokens == prompt_length and prefix_state.last_logits is None:
                return None, 0
            return clone_prefill_state_for_reuse(prefix_state), shared_tokens
        if shared_tokens >= prompt_length:
            return None, 0

        derived_state = derive_prefill_prefix_state(prefix_state, shared_tokens)
        if derived_state is None:
            return None, 0
        return derived_state, shared_tokens

    def _select_prefix_state_locked(
        self,
        prompt_tokens: list[int],
        previous_response_id: str | None,
        stable_prefix_key: str | None,
    ) -> tuple[PromptPrefillState | None, str]:
        best_state: PromptPrefillState | None = None
        best_source = "none"
        best_tokens = 0

        response_state, response_tokens = self._reusable_prefix_state_for_prompt(
            prompt_tokens,
            self._prefix_cache_state_for_response_locked(previous_response_id),
        )
        if response_state is not None:
            best_state = response_state
            best_source = "response"
            best_tokens = response_tokens

        global_state, global_tokens = self._reusable_prefix_state_for_prompt(
            prompt_tokens,
            self._global_prefix_state_for_key_locked(stable_prefix_key),
        )
        if global_state is not None:
            if best_state is None or global_tokens > best_tokens:
                best_state = global_state
                best_source = "global"
                best_tokens = global_tokens
            self._global_prefix_cache_hits += 1
        elif stable_prefix_key is not None and self.global_prefix_cache_limit > 0:
            self._global_prefix_cache_misses += 1

        return best_state, best_source

    def _remember_global_prefix_state_locked(
        self,
        stable_prefix_key: str | None,
        stable_prefix_tokens: tuple[int, ...],
        prompt_cache_state: PromptPrefillState | None,
    ) -> None:
        if (
            stable_prefix_key is None
            or not stable_prefix_tokens
            or self.global_prefix_cache_limit <= 0
            or self._model is None
            or self._tokenizer is None
        ):
            return

        derived_state = None
        if prompt_cache_state is not None and _prompt_startswith(list(prompt_cache_state.prompt_tokens), stable_prefix_tokens):
            derived_state = derive_prefill_prefix_state(prompt_cache_state, len(stable_prefix_tokens))

        if derived_state is None:
            prefill = prefill_prompt(
                self._model,
                self._tokenizer,
                list(stable_prefix_tokens),
                target_turboquant_bits=self.target_turboquant_bits,
                capture_prefill_state=True,
            )
            derived_state = prefill.prefill_state

        if derived_state is None:
            return

        try:
            state_bytes = int(estimate_memory_bytes(derived_state))
        except Exception:
            state_bytes = 0
        if (
            self.global_prefix_cache_byte_limit > 0
            and state_bytes > self.global_prefix_cache_byte_limit
        ):
            return

        previous = self._global_prefix_states.get(stable_prefix_key)
        if previous is not None:
            prev_bytes = self._global_prefix_state_bytes.pop(stable_prefix_key, 0)
            self._global_prefix_cache_bytes = max(0, self._global_prefix_cache_bytes - prev_bytes)
        self._global_prefix_states[stable_prefix_key] = derived_state
        self._global_prefix_state_bytes[stable_prefix_key] = state_bytes
        self._global_prefix_cache_bytes += state_bytes
        try:
            self._global_prefix_order.remove(stable_prefix_key)
        except ValueError:
            pass
        self._global_prefix_order.append(stable_prefix_key)
        self._prune_global_prefix_states_locked()

    def _should_capture_prompt_cache_state(self, stable_prefix_tokens: tuple[int, ...]) -> bool:
        return self.prefix_cache_state_limit > 0

    def _effective_max_tokens(self, requested_max_tokens: int, prompt_tokens: int) -> int:
        candidates = [max(1, requested_max_tokens)]

        if self.max_tokens_limit is not None:
            candidates.append(max(1, self.max_tokens_limit))

        if self.context_window is not None:
            max_prompt_tokens = max(1, self.context_window - self.context_reserve)
            if prompt_tokens > max_prompt_tokens:
                raise PromptTooLargeError(
                    "Prompt too large for configured context window: "
                    f"{prompt_tokens} tokens > {max_prompt_tokens} token limit"
                )
            available_context = self.context_window - prompt_tokens - self.context_reserve
            candidates.append(max(1, available_context))

        return min(candidates)

    def _generate_ddtree_result_locked(
        self,
        messages: list[dict[str, Any]],
        requested_max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        capture_prompt_cache_state: bool = False,
    ) -> tuple[dict[str, Any], int, float]:
        self.ensure_loaded()
        prompt = self.build_prompt(messages, tools=tools)
        prompt_tokens_list = self.tokenize_prompt(prompt)
        prompt_tokens = len(prompt_tokens_list)
        max_tokens = self._effective_max_tokens(requested_max_tokens, prompt_tokens)
        started = time.time()
        stable_prefix_key = self._stable_prefix_key(messages, tools=tools)
        stable_prefix_tokens = self._stable_prefix_tokens_locked(messages, tools=tools)
        capture_prefill_state = capture_prompt_cache_state and self._should_capture_prompt_cache_state(stable_prefix_tokens)
        prefix_state, prefix_cache_source = self._select_prefix_state_locked(
            prompt_tokens_list,
            previous_response_id,
            stable_prefix_key,
        )

        target_turboquant_bits = (
            None if self._ddtree_target_turboquant_failed else self.ddtree_target_turboquant_bits
        )
        attempts: list[tuple[str, float | None]] = [("ddtree", target_turboquant_bits)]
        if self.ddtree_retry_without_turboquant and target_turboquant_bits is not None:
            attempts.append(("ddtree-no-target-turboquant", None))

        last_error: Exception | None = None
        for engine_name, target_turboquant_bits in attempts:
            try:
                result = generate_ddtree(
                    target_model=self._model,
                    draft_model=self._draft,
                    tokenizer=self._tokenizer,
                    prompt_tokens=prompt_tokens_list,
                    max_new_tokens=max_tokens,
                    tree_budget=self.ddtree_tree_budget,
                    block_size=self.block_size,
                    adaptive_block_size=self.adaptive_block_size_config,
                    prefix_state=prefix_state,
                    capture_prefill_state=capture_prefill_state,
                    target_turboquant_bits=target_turboquant_bits,
                )
                result.update(
                    {
                        "engine": engine_name,
                        "prompt_tokens": prompt_tokens,
                        "prefix_cache_source": prefix_cache_source,
                        "prefill_hidden_bytes": int(result.get("prefill_hidden_bytes", 0) or 0),
                        "prefill_target_cache_bytes": int(result.get("prefill_target_cache_bytes", 0) or 0),
                        "prefill_logits_bytes": int(result.get("prefill_logits_bytes", 0) or 0),
                        "prefill_working_set_bytes": int(result.get("prefill_working_set_bytes", 0) or 0),
                        "prompt_cache_state_bytes": int(result.get("prompt_cache_state_bytes", 0) or 0),
                        "prompt_cache_state": result.get("prompt_cache_state") if capture_prefill_state else None,
                        "elapsed": time.time() - started,
                    }
                )
                if stable_prefix_tokens and result.get("prompt_cache_state") is not None:
                    self._remember_global_prefix_state_locked(
                        stable_prefix_key,
                        stable_prefix_tokens,
                        result.get("prompt_cache_state"),
                    )
                return result, prompt_tokens, started
            except Exception as exc:
                last_error = exc
                _logger.warning("%s generation failed: %s", engine_name, exc)
                if engine_name == "ddtree" and target_turboquant_bits is not None:
                    self._ddtree_target_turboquant_failed = True

        if last_error is not None:
            raise last_error
        raise RuntimeError("DDTree generation failed without an exception")

    @staticmethod
    def _ddtree_result_to_response(result: dict[str, Any]) -> GenerationResponse:
        response = GenerationResponse(
            text=str(result.get("text", "")),
            tokens=[],
            accepted=int(result.get("accepted_tokens", 0) or 0),
            prompt_tokens=int(result.get("prompt_tokens", 0) or 0),
            prefill_seconds=float(result.get("prefill_seconds", 0.0) or 0.0),
            reused_prefix_tokens=int(result.get("reused_prefix_tokens", 0) or 0),
            prompt_tps=float(result.get("prompt_tps", 0.0) or 0.0),
            generation_tokens=int(result.get("generated_tokens", 0) or 0),
            decode_seconds=float(result.get("decode_seconds", 0.0) or 0.0),
            generation_tps=float(result.get("generation_tps", 0.0) or 0.0),
            peak_memory=float(result.get("peak_memory_gb", 0.0) or 0.0),
            speculative_steps=int(result.get("speculative_steps", 0) or 0),
            proposed_tokens=int(result.get("proposed_tokens", 0) or 0),
            accepted_tokens=int(result.get("accepted_tokens", 0) or 0),
            avg_acceptance_length=float(result.get("avg_acceptance_length", 0.0) or 0.0),
            avg_acceptance_ratio=float(result.get("avg_acceptance_ratio", 0.0) or 0.0),
            acceptance_lengths=tuple(result.get("acceptance_lengths", ()) or ()),
            acceptance_ratios=tuple(result.get("acceptance_ratios", ()) or ()),
            block_size_history=tuple(result.get("block_size_history", ()) or ()),
            adaptive_block_size=False,
            finish_reason=str(result.get("finish_reason", "stop")),
            prefill_state=result.get("prompt_cache_state"),
        )
        response.extra_result_fields = {
            key: value
            for key, value in result.items()
            if key.startswith("ddtree_")
            or key
            in {
                "engine",
                "tree_budget",
                "target_turboquant_bits",
                "tree_node_count_history",
                "avg_tree_node_count",
                "max_tree_node_count",
            }
        }
        return response

    def _generate_locked(
        self,
        messages: list[dict[str, Any]],
        requested_max_tokens: int,
        sampling: "SamplingParams | float | None" = None,
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        capture_prompt_cache_state: bool = False,
        should_stop: Any = None,
        *,
        temperature: float | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        sampling = _coerce_sampling_arg(sampling, temperature)
        if self.generation_engine == "ddtree":
            try:
                result, _, _ = self._generate_ddtree_result_locked(
                    messages,
                    requested_max_tokens,
                    tools=tools,
                    previous_response_id=previous_response_id,
                    capture_prompt_cache_state=capture_prompt_cache_state,
                )
                return [str(result.get("text", ""))], result
            except Exception:
                if not self.ddtree_fallback_to_dflash:
                    raise
                _logger.exception("DDTree generation failed; falling back to DFlash")

        self.ensure_loaded()
        prompt = self.build_prompt(messages, tools=tools)
        prompt_tokens_list = self.tokenize_prompt(prompt)
        prompt_tokens = len(prompt_tokens_list)
        max_tokens = self._effective_max_tokens(requested_max_tokens, prompt_tokens)
        stable_prefix_key = self._stable_prefix_key(messages, tools=tools)
        stable_prefix_tokens = self._stable_prefix_tokens_locked(messages, tools=tools)
        capture_prefill_state = capture_prompt_cache_state and self._should_capture_prompt_cache_state(stable_prefix_tokens)
        prefix_state, prefix_cache_source = self._select_prefix_state_locked(
            prompt_tokens_list,
            previous_response_id,
            stable_prefix_key,
        )
        text_parts: list[str] = []
        final = None
        started = time.time()
        for chunk in stream_generate(
            self._model,
            self._draft,
            self._tokenizer,
            prompt_tokens_list,
            block_size=self.block_size,
            max_tokens=max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
            presence_penalty=sampling.presence_penalty,
            repetition_penalty=sampling.repetition_penalty,
            frequency_penalty=sampling.frequency_penalty,
            repetition_context_size=sampling.repetition_context_size,
            presence_context_size=sampling.presence_context_size,
            target_turboquant_bits=self.target_turboquant_bits,
            prefix_state=prefix_state,
            capture_prefill_state=capture_prefill_state,
            adaptive_block_size=self.adaptive_block_size_config,
            should_stop=should_stop,
        ):
            if chunk.text:
                text_parts.append(chunk.text)
            final = chunk
        if final is None:
            raise RuntimeError("Model returned no output")
        if stable_prefix_tokens:
            self._remember_global_prefix_state_locked(
                stable_prefix_key,
                stable_prefix_tokens,
                final.prefill_state if capture_prefill_state else None,
            )
        result = {
            "text": "".join(text_parts),
            "finish_reason": final.finish_reason or "stop",
            "prompt_tokens": prompt_tokens,
            "prefill_seconds": final.prefill_seconds,
            "prompt_tps": final.prompt_tps,
            "reused_prefix_tokens": final.reused_prefix_tokens,
            "decode_seconds": final.decode_seconds,
            "generation_tps": final.generation_tps,
            "generated_tokens": final.generation_tokens,
            "speculative_steps": final.speculative_steps,
            "proposed_tokens": final.proposed_tokens,
            "accepted_tokens": final.accepted_tokens,
            "avg_acceptance_length": final.avg_acceptance_length,
            "avg_acceptance_ratio": final.avg_acceptance_ratio,
            "acceptance_lengths": list(final.acceptance_lengths),
            "acceptance_ratios": list(final.acceptance_ratios),
            "block_size_history": list(final.block_size_history),
            "adaptive_block_size": final.adaptive_block_size,
            "prefix_cache_source": prefix_cache_source,
            "prefill_hidden_bytes": final.prefill_hidden_bytes,
            "prefill_target_cache_bytes": final.prefill_target_cache_bytes,
            "prefill_logits_bytes": final.prefill_logits_bytes,
            "prefill_working_set_bytes": final.prefill_working_set_bytes,
            "prompt_cache_state_bytes": final.prompt_cache_state_bytes,
            "peak_memory_gb": final.peak_memory,
            "elapsed": time.time() - started,
            "prompt_cache_state": final.prefill_state if capture_prefill_state else None,
            "engine": "dflash",
        }
        return text_parts, result

    def _stream_generate_locked(
        self,
        messages: list[dict[str, Any]],
        requested_max_tokens: int,
        sampling: "SamplingParams | float | None" = None,
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        capture_prompt_cache_state: bool = False,
        should_stop: Any = None,
        *,
        temperature: float | None = None,
    ) -> tuple[Iterator[Any], int, float, str | None, tuple[int, ...], str]:
        sampling = _coerce_sampling_arg(sampling, temperature)
        if self.generation_engine == "ddtree":
            try:
                result, prompt_tokens, started = self._generate_ddtree_result_locked(
                    messages,
                    requested_max_tokens,
                    tools=tools,
                    previous_response_id=previous_response_id,
                    capture_prompt_cache_state=capture_prompt_cache_state,
                )
                return (
                    iter((self._ddtree_result_to_response(result),)),
                    prompt_tokens,
                    started,
                    None,
                    (),
                    result.get("prefix_cache_source", "none"),
                )
            except Exception:
                if not self.ddtree_fallback_to_dflash:
                    raise
                _logger.exception("DDTree streaming generation failed; falling back to DFlash")

        self.ensure_loaded()
        prompt = self.build_prompt(messages, tools=tools)
        prompt_tokens_list = self.tokenize_prompt(prompt)
        prompt_tokens = len(prompt_tokens_list)
        max_tokens = self._effective_max_tokens(requested_max_tokens, prompt_tokens)
        started = time.time()
        stable_prefix_key = self._stable_prefix_key(messages, tools=tools)
        stable_prefix_tokens = self._stable_prefix_tokens_locked(messages, tools=tools)
        capture_prefill_state = capture_prompt_cache_state and self._should_capture_prompt_cache_state(stable_prefix_tokens)
        prefix_state, prefix_cache_source = self._select_prefix_state_locked(
            prompt_tokens_list,
            previous_response_id,
            stable_prefix_key,
        )
        iterator = stream_generate(
            self._model,
            self._draft,
            self._tokenizer,
            prompt_tokens_list,
            block_size=self.block_size,
            max_tokens=max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            min_p=sampling.min_p,
            presence_penalty=sampling.presence_penalty,
            repetition_penalty=sampling.repetition_penalty,
            frequency_penalty=sampling.frequency_penalty,
            repetition_context_size=sampling.repetition_context_size,
            presence_context_size=sampling.presence_context_size,
            target_turboquant_bits=self.target_turboquant_bits,
            prefix_state=prefix_state,
            capture_prefill_state=capture_prefill_state,
            adaptive_block_size=self.adaptive_block_size_config,
            should_stop=should_stop,
        )
        return iterator, prompt_tokens, started, stable_prefix_key, stable_prefix_tokens, prefix_cache_source

    def _generation_worker(
        self,
        queue: Queue,
        messages: list[dict[str, Any]],
        requested_max_tokens: int,
        sampling: SamplingParams,
        tools: list[dict[str, Any]] | None = None,
        keep_alive_override: Any = None,
        previous_response_id: str | None = None,
        capture_prompt_cache_state: bool = False,
        stop_event: Any = None,
    ) -> None:
        generation_ticket: int | None = None
        text_parts: list[str] = []
        final = None
        prompt_tokens = 0
        started = time.time()
        stable_prefix_key: str | None = None
        stable_prefix_tokens: tuple[int, ...] = ()
        prefix_cache_source = "none"
        try:
            generation_ticket = self._acquire_generation_turn()
            with self._lock:
                iterator, prompt_tokens, started, stable_prefix_key, stable_prefix_tokens, prefix_cache_source = self._stream_generate_locked(
                    messages,
                    requested_max_tokens,
                    sampling,
                    tools=tools,
                    previous_response_id=previous_response_id,
                    capture_prompt_cache_state=capture_prompt_cache_state,
                    should_stop=(stop_event.is_set if stop_event is not None else None),
                )
                for chunk in iterator:
                    if chunk.text:
                        text_parts.append(chunk.text)
                        if not _queue_put(queue, ("text", chunk.text), stop_event):
                            return
                    final = chunk

            if final is None:
                raise RuntimeError("Model returned no output")

            with self._lock:
                if stable_prefix_tokens:
                    self._remember_global_prefix_state_locked(
                        stable_prefix_key,
                        stable_prefix_tokens,
                        final.prefill_state if capture_prompt_cache_state else None,
                    )
            result = {
                "text": "".join(text_parts),
                "finish_reason": final.finish_reason or "stop",
                "prompt_tokens": prompt_tokens,
                "prefill_seconds": final.prefill_seconds,
                "prompt_tps": final.prompt_tps,
                "reused_prefix_tokens": final.reused_prefix_tokens,
                "decode_seconds": final.decode_seconds,
                "generation_tps": final.generation_tps,
                "generated_tokens": final.generation_tokens,
                "speculative_steps": final.speculative_steps,
                "proposed_tokens": final.proposed_tokens,
                "accepted_tokens": final.accepted_tokens,
                "avg_acceptance_length": final.avg_acceptance_length,
                "avg_acceptance_ratio": final.avg_acceptance_ratio,
                "acceptance_lengths": list(final.acceptance_lengths),
                "acceptance_ratios": list(final.acceptance_ratios),
                "block_size_history": list(final.block_size_history),
                "adaptive_block_size": final.adaptive_block_size,
                "prefix_cache_source": prefix_cache_source,
                "prefill_hidden_bytes": final.prefill_hidden_bytes,
                "prefill_target_cache_bytes": final.prefill_target_cache_bytes,
                "prefill_logits_bytes": final.prefill_logits_bytes,
                "prefill_working_set_bytes": final.prefill_working_set_bytes,
                "prompt_cache_state_bytes": final.prompt_cache_state_bytes,
                "peak_memory_gb": final.peak_memory,
                "elapsed": time.time() - started,
                "prompt_cache_state": final.prefill_state if capture_prompt_cache_state else None,
                "engine": "dflash",
            }
            extra_result_fields = getattr(final, "extra_result_fields", None)
            if isinstance(extra_result_fields, dict):
                result.update(extra_result_fields)
            self._record_generation_metrics(result, surface="stream")
            if not _queue_put(queue, ("result", result), stop_event):
                return
        except Exception as exc:
            _queue_put(queue, ("error", str(exc)), stop_event)
        finally:
            try:
                self.finish_request(keep_alive_override)
            finally:
                self._release_generation_turn(generation_ticket)
            _queue_put(queue, ("done", None), stop_event)

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        sampling: "SamplingParams | float | None" = None,
        tools: list[dict[str, Any]] | None = None,
        keep_alive_override: Any = None,
        previous_response_id: str | None = None,
        capture_prompt_cache_state: bool = False,
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        sampling = _coerce_sampling_arg(sampling, temperature)
        generation_ticket = self._acquire_generation_turn()
        try:
            current_messages = list(messages)
            remaining_max_tokens = max_tokens
            result: dict[str, Any] | None = None
            with self._lock:
                for _ in range(RESPONSES_ACTION_FOLLOWUP_LIMIT + 1):
                    _, result = self._generate_locked(
                        current_messages,
                        remaining_max_tokens,
                        sampling,
                        tools=tools,
                        previous_response_id=previous_response_id,
                        capture_prompt_cache_state=capture_prompt_cache_state,
                    )
                    remaining_max_tokens = max(
                        0,
                        remaining_max_tokens - max(0, int(result.get("generated_tokens", 0))),
                    )
                    previous_response_id = None
                    if (
                        remaining_max_tokens <= 0
                        or not _chat_result_needs_action_followup(
                            result, tools, current_messages
                        )
                    ):
                        break
                    visible_text, _ = _parse_tool_calls(_coerce_text(result.get("text", "")))
                    current_messages = [
                        *current_messages,
                        {"role": "assistant", "content": visible_text},
                        {"role": "user", "content": _responses_action_prompt(tools)},
                    ]
            if result is None:
                raise RuntimeError("Model returned no output")
            self._record_generation_metrics(result, surface="generate")
            return result
        finally:
            try:
                self.finish_request(keep_alive_override)
            finally:
                self._release_generation_turn(generation_ticket)

    def _generate_response_locked(
        self,
        messages: list[dict[str, Any]],
        requested_max_tokens: int,
        sampling: "SamplingParams | float | None" = None,
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        capture_prompt_cache_state: bool = False,
        should_stop: Any = None,
        *,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        sampling = _coerce_sampling_arg(sampling, temperature)
        current_messages = list(messages)
        current_previous_response_id = previous_response_id
        remaining_max_tokens = max(1, int(requested_max_tokens))

        for _ in range(RESPONSES_ACTION_FOLLOWUP_LIMIT + 1):
            if should_stop is not None:
                try:
                    if should_stop():
                        break
                except Exception:
                    pass
            _, result = self._generate_locked(
                current_messages,
                remaining_max_tokens,
                sampling,
                tools=tools,
                previous_response_id=current_previous_response_id,
                capture_prompt_cache_state=capture_prompt_cache_state,
                should_stop=should_stop,
            )
            remaining_max_tokens = max(
                0,
                remaining_max_tokens - max(0, int(result.get("generated_tokens", 0))),
            )
            output_items = _build_output_items(result["text"])
            output_items = _convert_items_for_custom_tools(output_items, tools)
            if _response_needs_empty_output_followup(result, output_items, tools):
                current_messages = [
                    *current_messages,
                    {
                        "role": "user",
                        "content": RESPONSES_EMPTY_OUTPUT_PROMPT,
                    },
                ]
                current_previous_response_id = None
                continue
            if not _response_is_followup_candidate(result, output_items, tools):
                return result, output_items
            if remaining_max_tokens <= 0:
                return result, output_items

            assistant_text = _output_text_from_items(output_items).strip()
            if _assistant_text_promises_action(assistant_text):
                need_followup = True
            elif _task_history_requires_completion_followup(
                current_messages, assistant_text
            ):
                need_followup = True
            else:
                need_followup = self._judge_response_needs_followup(
                    current_messages, assistant_text, tools
                )
            if not need_followup:
                return result, output_items

            carry_items = [
                item
                for item in output_items
                if not _is_planning_only_function_call(item)
            ]
            current_messages = [
                *current_messages,
                *_messages_from_output_items(carry_items),
                {
                    "role": "user",
                    "content": _responses_action_prompt(tools),
                },
            ]
            current_previous_response_id = None

        return result, output_items

    def _judge_response_needs_followup(
        self,
        messages: list[dict[str, Any]],
        assistant_text: str,
        tools: list[dict[str, Any]] | None,
    ) -> bool:
        if not RESPONSES_FOLLOWUP_JUDGE_ENABLED:
            return False

        last_user_text = _last_user_message_text(messages)
        tool_names = _summarize_tool_names(tools)

        logprob_verdict: bool | None = None
        logprob_margin = 0.0
        try:
            logprob_verdict, logprob_margin = self._judge_verdict_via_logprobs(
                last_user_text, tool_names, assistant_text
            )
        except Exception as exc:
            _logger.warning("judge logprob path failed: %s", exc)
            logprob_verdict, logprob_margin = None, 0.0

        if (
            logprob_verdict is not None
            and logprob_margin >= RESPONSES_FOLLOWUP_JUDGE_LOGPROB_MARGIN
        ):
            return logprob_verdict

        # Low-margin or no logprob signal — consult the reasoning judge.
        # `LOCAL_DFLASH_FOLLOWUP_JUDGE_TIEBREAK_VOTES` toggles best-of-N voting
        # (default 1 = legacy single-call behavior). Set to 3 for stronger
        # tie-breaking at the cost of extra tokens per judge invocation.
        votes_target = max(1, int(os.environ.get("LOCAL_DFLASH_FOLLOWUP_JUDGE_TIEBREAK_VOTES", "1")))
        reasoning_votes: list[bool] = []
        for _ in range(votes_target):
            verdict = self._judge_verdict_via_reasoning(
                last_user_text, tool_names, assistant_text
            )
            if verdict is None:
                break
            reasoning_votes.append(verdict)
            # Short-circuit: once the first 2 votes agree we already have a
            # majority for best-of-3.
            if (
                votes_target >= 3
                and len(reasoning_votes) == 2
                and reasoning_votes[0] == reasoning_votes[1]
            ):
                return reasoning_votes[0]
        if reasoning_votes:
            incomplete_count = sum(1 for v in reasoning_votes if v)
            complete_count = len(reasoning_votes) - incomplete_count
            if incomplete_count != complete_count:
                return incomplete_count > complete_count
            return reasoning_votes[0]

        if logprob_verdict is not None:
            return logprob_verdict
        return True

    def _judge_verdict_via_logprobs(
        self,
        last_user_text: str,
        tool_names: str,
        assistant_text: str,
    ) -> tuple[bool | None, float]:
        self.ensure_loaded()
        tokenizer = self._tokenizer
        tok_y = _first_encoded_token_id(tokenizer, "Y")
        tok_n = _first_encoded_token_id(tokenizer, "N")
        if tok_y is None or tok_n is None or tok_y == tok_n:
            return None, 0.0

        judge_messages = [
            {"role": "system", "content": RESPONSES_FOLLOWUP_JUDGE_LOGPROB_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Tools available: {tool_names}\n\n"
                    f"User's latest message:\n{last_user_text or '(not shown)'}\n\n"
                    f"Agent's latest response:\n{assistant_text}\n\n"
                    "Answer with exactly one letter: Y if the turn is complete, N if the agent still needs to act."
                ),
            },
        ]
        try:
            prompt_tokens = tokenizer.apply_chat_template(
                judge_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_tokens = tokenizer.apply_chat_template(
                judge_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        if hasattr(prompt_tokens, "tolist"):
            prompt_tokens = prompt_tokens.tolist()
        run = prefill_prompt(
            self._model,
            tokenizer,
            prompt_tokens,
            target_turboquant_bits=self.target_turboquant_bits,
            prefix_state=None,
            capture_prefill_state=False,
        )
        row = run.logits[0, -1, :].astype(mx.float32)
        mx.eval(row)
        logit_y = float(row[tok_y].item())
        logit_n = float(row[tok_n].item())
        margin = abs(logit_n - logit_y)
        is_incomplete = logit_n > logit_y
        return is_incomplete, margin

    def _judge_verdict_via_reasoning(
        self,
        last_user_text: str,
        tool_names: str,
        assistant_text: str,
    ) -> bool | None:
        judge_messages = [
            {"role": "system", "content": RESPONSES_FOLLOWUP_JUDGE_JSON_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Tools available: {tool_names}\n\n"
                    f"User's latest message:\n{last_user_text or '(not shown)'}\n\n"
                    f"Agent's latest response:\n{assistant_text}\n\n"
                    'Respond with ONLY: {"reason":"<one short sentence>","verdict":"COMPLETE"|"INCOMPLETE"}.'
                ),
            },
        ]
        try:
            _, judge_result = self._generate_locked(
                judge_messages,
                RESPONSES_FOLLOWUP_JUDGE_MAX_TOKENS,
                SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    presence_penalty=0.0,
                    repetition_penalty=0.0,
                    frequency_penalty=0.0,
                ),
            )
        except Exception as exc:
            _logger.warning("judge reasoning generation failed: %s", exc)
            return None
        raw = _coerce_text(judge_result.get("text", ""))
        _, visible = _strip_reasoning_blocks(raw)
        parsed = _extract_json_object(visible)
        if not parsed:
            return None
        verdict = str(parsed.get("verdict") or "").strip().upper()
        if verdict.startswith("INCOMPLETE"):
            return True
        if verdict.startswith("COMPLETE"):
            return False
        return None

    def _responses_generation_worker(
        self,
        queue: Queue,
        messages: list[dict[str, Any]],
        requested_max_tokens: int,
        sampling: SamplingParams,
        tools: list[dict[str, Any]] | None = None,
        keep_alive_override: Any = None,
        previous_response_id: str | None = None,
        capture_prompt_cache_state: bool = False,
        stop_event: Any = None,
    ) -> None:
        try:
            result, output_items = self.generate_response(
                messages,
                requested_max_tokens,
                sampling,
                tools=tools,
                keep_alive_override=keep_alive_override,
                previous_response_id=previous_response_id,
                capture_prompt_cache_state=capture_prompt_cache_state,
                should_stop=(stop_event.is_set if stop_event is not None else None),
            )
            _queue_put(queue, ("result", (result, output_items)), stop_event)
        except Exception as exc:
            _queue_put(queue, ("error", str(exc)), stop_event)
        finally:
            _queue_put(queue, ("done", None), stop_event)

    def generate_response(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        sampling: "SamplingParams | float | None" = None,
        tools: list[dict[str, Any]] | None = None,
        keep_alive_override: Any = None,
        previous_response_id: str | None = None,
        capture_prompt_cache_state: bool = False,
        should_stop: Any = None,
        *,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        sampling = _coerce_sampling_arg(sampling, temperature)
        generation_ticket = self._acquire_generation_turn()
        try:
            with self._lock:
                result, output_items = self._generate_response_locked(
                    messages,
                    max_tokens,
                    sampling,
                    tools=tools,
                    previous_response_id=previous_response_id,
                    capture_prompt_cache_state=capture_prompt_cache_state,
                    should_stop=should_stop,
                )
            self._record_generation_metrics(result, surface="responses")
            return result, output_items
        finally:
            try:
                self.finish_request(keep_alive_override)
            finally:
                self._release_generation_turn(generation_ticket)

    def stream_chat_completions(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        sampling: "SamplingParams | float | None" = None,
        tools: list[dict[str, Any]] | None = None,
        keep_alive_override: Any = None,
        *,
        temperature: float | None = None,
    ) -> Iterator[str]:
        sampling = _coerce_sampling_arg(sampling, temperature)
        from threading import Event as _Event
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        event_queue: Queue = _make_stream_queue()
        stop_event = _Event()
        worker = Thread(
            target=self._generation_worker,
            args=(event_queue, messages, max_tokens, sampling, tools, keep_alive_override, None, False, stop_event),
            daemon=True,
        )
        worker.start()

        try:
            visible_stream = _IncrementalVisibleTextStream(strip_edges=True)
            emitted_role = False
            result: dict[str, Any] | None = None
            done = False

            while not done:
                try:
                    kind, payload = event_queue.get(timeout=STREAM_HEARTBEAT_SECONDS)
                except Empty:
                    yield _chat_heartbeat_line(completion_id, created, self.model_name)
                    continue

                if kind == "text":
                    delta = visible_stream.feed(payload)
                    if not delta:
                        continue

                    chunk_delta: dict[str, Any] = {"content": delta}
                    if not emitted_role:
                        chunk_delta = {"role": "assistant", "content": delta}
                        emitted_role = True

                    yield _data_line(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": self.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": chunk_delta,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    continue

                if kind == "result":
                    result = payload
                    continue

                if kind == "error":
                    yield _data_line(
                        {
                            "error": {
                                "message": payload,
                            }
                        }
                    )
                    yield _done_line()
                    return

                if kind == "done":
                    done = True
        except GeneratorExit:
            # Client disconnected — signal worker to stop ASAP.
            stop_event.set()
            raise
        finally:
            stop_event.set()
            try:
                worker.join(timeout=30.0)
            except Exception:
                pass

        if result is None:
            yield _data_line(
                {
                    "error": {
                        "message": "Generation completed without a final result",
                    }
                }
            )
            yield _done_line()
            return

        final_delta = visible_stream.feed("", final=True)
        if final_delta:
            chunk_delta: dict[str, Any] = {"content": final_delta}
            if not emitted_role:
                chunk_delta = {"role": "assistant", "content": final_delta}
                emitted_role = True
            yield _data_line(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": chunk_delta,
                            "finish_reason": None,
                        }
                    ],
                }
            )

        if not emitted_role:
            yield _data_line(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )

        yield _data_line(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": result["finish_reason"],
                    }
                ],
            }
        )
        yield _done_line()

    def stream_response_events(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        sampling: "SamplingParams | float | None" = None,
        tools: list[dict[str, Any]] | None = None,
        request_messages: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        keep_alive_override: Any = None,
        *,
        temperature: float | None = None,
    ) -> Iterator[str]:
        sampling = _coerce_sampling_arg(sampling, temperature)
        response_id = f"resp_{uuid.uuid4().hex}"
        created_response = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": self.model_name,
            "output": [],
            "output_text": "",
            "previous_response_id": previous_response_id,
        }
        created = {
            "type": "response.created",
            "response": created_response,
        }
        yield _json_line("response.created", created)
        yield _json_line(
            "response.in_progress",
            {
                "type": "response.in_progress",
                "response": created_response,
            },
        )
        from threading import Event as _Event
        event_queue: Queue = _make_stream_queue()
        stop_event = _Event()
        worker = Thread(
            target=self._responses_generation_worker,
            args=(event_queue, messages, max_tokens, sampling, tools, keep_alive_override, previous_response_id, True, stop_event),
            daemon=True,
        )
        worker.start()
        try:
            yield from self._stream_response_events_body(
                response_id=response_id,
                previous_response_id=previous_response_id,
                event_queue=event_queue,
                messages=messages,
                max_tokens=max_tokens,
                sampling=sampling,
                tools=tools,
                request_messages=request_messages,
            )
        except GeneratorExit:
            stop_event.set()
            raise
        finally:
            stop_event.set()
            try:
                worker.join(timeout=30.0)
            except Exception:
                pass
        return

    def _stream_response_events_body(
        self,
        *,
        response_id: str,
        previous_response_id: str | None,
        event_queue: Queue,
        messages: list[dict[str, Any]],
        max_tokens: int,
        sampling: SamplingParams,
        tools: list[dict[str, Any]] | None,
        request_messages: list[dict[str, Any]] | None,
    ) -> Iterator[str]:
        message_item_id: str | None = None
        result: dict[str, Any] | None = None
        output_items: list[dict[str, Any]] | None = None
        done = False

        while not done:
            try:
                kind, payload = event_queue.get(timeout=STREAM_HEARTBEAT_SECONDS)
            except Empty:
                yield _responses_heartbeat_line(response_id, self.model_name)
                continue

            if kind == "text":
                continue

            if kind == "result":
                result, output_items = payload
                continue

            if kind == "error":
                err_message = str(payload)
                err_code = _classify_error_code(err_message)
                yield _json_line(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "status": "failed",
                            "model": self.model_name,
                            "error": {
                                "code": err_code,
                                "message": err_message,
                            },
                        },
                        "error": {
                            "code": err_code,
                            "message": err_message,
                        },
                    },
                )
                yield _done_line()
                return

            if kind == "done":
                done = True

        if result is None:
            err_message = "Generation completed without a final result"
            err_code = "server_error"
            yield _json_line(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "failed",
                        "model": self.model_name,
                        "error": {
                            "code": err_code,
                            "message": err_message,
                        },
                    },
                    "error": {
                        "code": err_code,
                        "message": err_message,
                    },
                },
            )
            yield _done_line()
            return

        assert output_items is not None
        final_output_text = _output_text_from_items(output_items)
        if final_output_text:
            if message_item_id is None:
                message_item_id = f"msg_{uuid.uuid4().hex}"
                yield _json_line(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "response_id": response_id,
                        "output_index": 0,
                        "item": {
                            "id": message_item_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": [],
                                    "logprobs": [],
                                }
                            ],
                        },
                    },
                )
                yield _json_line(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "response_id": response_id,
                        "output_index": 0,
                        "item_id": message_item_id,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    },
                )
            yield _json_line(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "output_index": 0,
                    "item_id": message_item_id,
                    "content_index": 0,
                    "delta": final_output_text,
                },
            )

        effective_request_messages = request_messages or messages
        self.remember_response(
            response_id=response_id,
            previous_response_id=previous_response_id,
            request_messages=effective_request_messages,
            tools=tools or [],
            output_items=output_items,
            prompt_cache_state=result.get("prompt_cache_state"),
        )

        next_output_index = 0
        for item in output_items:
            if item["type"] == "reasoning":
                reasoning_text = ""
                for part in item.get("summary") or []:
                    if isinstance(part, dict):
                        reasoning_text += str(part.get("text") or "")
                yield _json_line(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "response_id": response_id,
                        "output_index": next_output_index,
                        "item": {
                            "id": item["id"],
                            "type": "reasoning",
                            "summary": [],
                            "status": "in_progress",
                        },
                    },
                )
                if reasoning_text:
                    yield _json_line(
                        "response.reasoning_summary_part.added",
                        {
                            "type": "response.reasoning_summary_part.added",
                            "response_id": response_id,
                            "output_index": next_output_index,
                            "item_id": item["id"],
                            "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""},
                        },
                    )
                    yield _json_line(
                        "response.reasoning_summary_text.delta",
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "response_id": response_id,
                            "output_index": next_output_index,
                            "item_id": item["id"],
                            "summary_index": 0,
                            "delta": reasoning_text,
                        },
                    )
                yield _json_line(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": next_output_index,
                        "item": item,
                    },
                )
                next_output_index += 1
                continue

            if item["type"] == "message":
                if message_item_id is None:
                    if _output_text_from_items([item]):
                        yield _json_line(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "response_id": response_id,
                                "output_index": next_output_index,
                                "item": item,
                            },
                        )
                        yield _json_line(
                            "response.content_part.added",
                            {
                                "type": "response.content_part.added",
                                "response_id": response_id,
                                "output_index": next_output_index,
                                "item_id": item["id"],
                                "content_index": 0,
                                "part": item["content"][0],
                            },
                        )
                        yield _json_line(
                            "response.output_text.done",
                            {
                                "type": "response.output_text.done",
                                "response_id": response_id,
                                "output_index": next_output_index,
                                "item_id": item["id"],
                                "content_index": 0,
                                "text": _output_text_from_items([item]),
                            },
                        )
                        yield _json_line(
                            "response.content_part.done",
                            {
                                "type": "response.content_part.done",
                                "response_id": response_id,
                                "output_index": next_output_index,
                                "item_id": item["id"],
                                "content_index": 0,
                                "part": item["content"][0],
                            },
                        )
                        next_output_index += 1
                else:
                    item["id"] = message_item_id
                    yield _json_line(
                        "response.output_text.done",
                        {
                            "type": "response.output_text.done",
                            "response_id": response_id,
                            "output_index": 0,
                            "item_id": message_item_id,
                            "content_index": 0,
                            "text": _output_text_from_items([item]),
                        },
                    )
                    yield _json_line(
                        "response.content_part.done",
                        {
                            "type": "response.content_part.done",
                            "response_id": response_id,
                            "output_index": 0,
                            "item_id": message_item_id,
                            "content_index": 0,
                            "part": item["content"][0],
                        },
                    )
                    yield _json_line(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "response_id": response_id,
                            "output_index": 0,
                            "item": item,
                        },
                    )
                    next_output_index = 1
                continue

            pending_item = copy.deepcopy(item)
            if pending_item["type"] == "function_call":
                pending_item["status"] = "in_progress"
                pending_item["arguments"] = ""
            elif pending_item["type"] == "custom_tool_call":
                pending_item["status"] = "in_progress"
                pending_item["input"] = ""
            yield _json_line(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": next_output_index,
                    "item": pending_item,
                },
            )
            if item["type"] == "function_call" and item.get("arguments"):
                yield _json_line(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "response_id": response_id,
                        "output_index": next_output_index,
                        "item_id": item["id"],
                        "delta": item["arguments"],
                    },
                )
                yield _json_line(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "response_id": response_id,
                        "output_index": next_output_index,
                        "item_id": item["id"],
                        "arguments": item["arguments"],
                    },
                )
            elif item["type"] == "custom_tool_call" and item.get("input"):
                yield _json_line(
                    "response.custom_tool_call_input.delta",
                    {
                        "type": "response.custom_tool_call_input.delta",
                        "response_id": response_id,
                        "output_index": next_output_index,
                        "item_id": item["id"],
                        "call_id": item.get("call_id"),
                        "delta": item["input"],
                    },
                )
                yield _json_line(
                    "response.custom_tool_call_input.done",
                    {
                        "type": "response.custom_tool_call_input.done",
                        "response_id": response_id,
                        "output_index": next_output_index,
                        "item_id": item["id"],
                        "call_id": item.get("call_id"),
                        "input": item["input"],
                    },
                )
            yield _json_line(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": next_output_index,
                    "item": item,
                },
            )
            next_output_index += 1

        response_status, incomplete_details = _response_completion_state(result)
        # Codex 0.122's `ResponseCompleted` deserializer accepts `status="completed"`
        # even when `incomplete_details` signals that generation hit a limit.
        # Keep the wire-level status `completed` so Codex does not error out;
        # the telemetry below still records the "real" state for our own logs.
        terminal_payload = _build_response_payload(
            response_id=response_id,
            model_name=self.model_name,
            result=result,
            output_items=output_items,
            status="completed",
            previous_response_id=previous_response_id,
            incomplete_details=incomplete_details,
        )
        _trace_event(
            f"responses.{response_status}",
            {
                "response_id": response_id,
                "previous_response_id": previous_response_id,
                "max_output_tokens": max_tokens,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "presence_penalty": sampling.presence_penalty,
                "repetition_penalty": sampling.repetition_penalty,
                "request_messages": effective_request_messages,
                "tools": tools or [],
                "raw_text": result["text"],
                "response": terminal_payload,
            },
        )
        # NOTE: Codex 0.122 treats `response.incomplete` as a terminal error
        # (ApiError::Stream). For `max_output_tokens` / `truncated_tool_call`
        # we must only emit `response.completed` with `incomplete_details`
        # inside the response payload itself. The `status` field already
        # reflects "incomplete" for client-side UI.
        yield _json_line(
            "response.completed",
            {
                "type": "response.completed",
                "response": terminal_payload,
            },
        )
        yield _done_line()

    def stream_anthropic_events(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        sampling: "SamplingParams | float | None" = None,
        tools: list[dict[str, Any]] | None = None,
        keep_alive_override: Any = None,
        *,
        temperature: float | None = None,
    ) -> Iterator[str]:
        sampling = _coerce_sampling_arg(sampling, temperature)
        from threading import Event as _Event
        message_id = f"msg_{uuid.uuid4().hex}"
        yield _json_line(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model_name,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                    },
                },
            },
        )
        event_queue: Queue = _make_stream_queue()
        stop_event = _Event()
        worker = Thread(
            target=self._generation_worker,
            args=(event_queue, messages, max_tokens, sampling, tools, keep_alive_override, None, False, stop_event),
            daemon=True,
        )
        worker.start()

        try:
            yield from self._stream_anthropic_events_body(event_queue)
        except GeneratorExit:
            stop_event.set()
            raise
        finally:
            stop_event.set()
            try:
                worker.join(timeout=30.0)
            except Exception:
                pass
        return

    def _stream_anthropic_events_body(self, event_queue: Queue) -> Iterator[str]:
        visible_stream = _IncrementalVisibleTextStream(strip_edges=False)
        text_block_open = False
        result: dict[str, Any] | None = None
        done = False

        while not done:
            try:
                kind, payload = event_queue.get(timeout=STREAM_HEARTBEAT_SECONDS)
            except Empty:
                yield _anthropic_heartbeat_line()
                continue

            if kind == "text":
                delta = visible_stream.feed(payload)
                if not delta:
                    continue
                if not text_block_open:
                    yield _json_line(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "text",
                                "text": "",
                            },
                        },
                    )
                    text_block_open = True
                yield _json_line(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "text_delta",
                            "text": delta,
                        },
                    },
                )
                continue

            if kind == "result":
                result = payload
                continue

            if kind == "error":
                yield _json_line(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": payload,
                        },
                    },
                )
                return

            if kind == "done":
                done = True

        if result is None:
            yield _json_line(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "Generation completed without a final result",
                    },
                },
            )
            return

        final_delta = visible_stream.feed("", final=True)
        if final_delta:
            if not text_block_open:
                yield _json_line(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "text",
                            "text": "",
                        },
                    },
                )
                text_block_open = True
            yield _json_line(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": final_delta,
                    },
                },
            )

        content_blocks = _build_anthropic_content_blocks(result["text"])
        stop_reason = _anthropic_stop_reason(result, content_blocks)

        next_index = 0
        for block in content_blocks:
            if block["type"] == "text":
                if text_block_open:
                    yield _json_line(
                        "content_block_stop",
                        {
                            "type": "content_block_stop",
                            "index": 0,
                        },
                    )
                    text_block_open = False
                    next_index = 1
                    continue
                if block.get("text"):
                    yield _json_line(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": next_index,
                            "content_block": block,
                        },
                    )
                    yield _json_line(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": next_index,
                            "delta": {
                                "type": "text_delta",
                                "text": block["text"],
                            },
                        },
                    )
                    yield _json_line(
                        "content_block_stop",
                        {
                            "type": "content_block_stop",
                            "index": next_index,
                        },
                    )
                    next_index += 1
                continue

            index = next_index
            if block["type"] == "tool_use":
                start_block = {
                    "type": "tool_use",
                    "id": block["id"],
                    "name": block["name"],
                    "input": {},
                }
            else:
                start_block = block

            yield _json_line(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": start_block,
                },
            )

            if block["type"] == "text" and block.get("text"):
                yield _json_line(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "text_delta",
                            "text": block["text"],
                        },
                    },
                )

            if block["type"] == "tool_use":
                partial_json = json.dumps(block.get("input", {}), ensure_ascii=False)
                if partial_json:
                    yield _json_line(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": partial_json,
                            },
                        },
                    )

            yield _json_line(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": index,
                },
            )
            next_index += 1

        yield _json_line(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                },
                "usage": {
                    "output_tokens": result["generated_tokens"],
                },
            },
        )
        yield _json_line(
            "message_stop",
            {
                "type": "message_stop",
            },
        )


def create_app(server: LocalModelServer) -> FastAPI:
    app = FastAPI(title="Local DFlash API", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        last_request_metrics = dict(server._last_request_metrics or {})
        return {
            "status": "ok",
            "model": server.model_name,
            "loaded": server._model is not None,
            "context_window": server.context_window,
            "context_reserve": server.context_reserve,
            "block_size": server.block_size,
            "generation_engine": server.generation_engine,
            "ddtree_tree_budget": server.ddtree_tree_budget,
            "ddtree_target_turboquant_bits": server.ddtree_target_turboquant_bits,
            "ddtree_fallback_to_dflash": server.ddtree_fallback_to_dflash,
            "ddtree_retry_without_turboquant": server.ddtree_retry_without_turboquant,
            "disable_thinking": server.disable_thinking,
            "sliding_window_size": server.sliding_window_size,
            "max_tokens_limit": server.max_tokens_limit,
            "keep_alive_seconds": server.keep_alive_seconds,
            "stream_heartbeat_seconds": STREAM_HEARTBEAT_SECONDS,
            "target_turboquant_bits": server.target_turboquant_bits,
            "draft_turboquant_bits": server.draft_turboquant_bits,
            "response_history_limit": server.response_history_limit,
            "response_history_entries": len(server._response_order),
            "active_generation_requests": 1 if server._active_generation_ticket is not None else 0,
            "queued_generation_requests": len(server._queued_generation_tickets),
            "prefix_cache_state_limit": server.prefix_cache_state_limit,
            "prefix_cache_state_byte_limit": server.prefix_cache_state_byte_limit,
            "prefix_cache_entries": len(server._prefix_state_order),
            "response_prefix_cache_bytes": server._response_prefix_cache_bytes,
            "global_prefix_cache_limit": server.global_prefix_cache_limit,
            "global_prefix_cache_entries": len(server._global_prefix_order),
            "global_prefix_cache_bytes": server._global_prefix_cache_bytes,
            "global_prefix_cache_hits": server._global_prefix_cache_hits,
            "global_prefix_cache_misses": server._global_prefix_cache_misses,
            "adaptive_block_size": server.adaptive_block_size_config.enabled,
            "adaptive_block_size_min": server.adaptive_block_size_config.min_block_size,
            "adaptive_block_size_max": server.adaptive_block_size_config.max_block_size,
            "adaptive_block_size_grow_threshold": server.adaptive_block_size_config.grow_threshold,
            "adaptive_block_size_shrink_threshold": server.adaptive_block_size_config.shrink_threshold,
            "adaptive_block_size_grow_step": server.adaptive_block_size_config.grow_step,
            "adaptive_block_size_shrink_step": server.adaptive_block_size_config.shrink_step,
            "adaptive_block_size_grow_streak": server.adaptive_block_size_config.grow_streak,
            "adaptive_block_size_shrink_streak": server.adaptive_block_size_config.shrink_streak,
            "request_metrics_entries": len(server._request_metrics_order),
            "last_request_metrics": last_request_metrics,
            "last_used_at": server._last_used_at,
            "active_memory_gb": mx.get_active_memory() / (1024 ** 3),
            "cache_memory_gb": mx.get_cache_memory() / (1024 ** 3),
            "peak_memory_gb": mx.get_peak_memory() / (1024 ** 3),
        }

    _server_started_at = time.time()

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        """Time-series-friendly gauges for a watchdog / observability scraper.
        All values are numeric; no strings in values.
        """
        active_pid_age = 0.0
        if server._active_generation_ticket is not None and server._last_used_at:
            active_pid_age = max(0.0, time.time() - float(server._last_used_at))
        cache_memory = 0.0
        try:
            cache_memory = float(mx.get_cache_memory())
        except Exception:
            pass
        active_memory = 0.0
        try:
            active_memory = float(mx.get_active_memory())
        except Exception:
            pass
        peak_memory = 0.0
        try:
            peak_memory = float(mx.get_peak_memory())
        except Exception:
            pass
        return {
            "dflash_uptime_seconds": time.time() - _server_started_at,
            "dflash_model_loaded": 1 if server._model is not None else 0,
            "dflash_active_generation_requests": 1 if server._active_generation_ticket is not None else 0,
            "dflash_queued_generation_requests": len(server._queued_generation_tickets),
            "dflash_active_ticket_age_seconds": active_pid_age,
            "dflash_response_history_entries": len(server._response_order),
            "dflash_prefix_cache_entries": len(server._prefix_state_order),
            "dflash_response_prefix_cache_bytes": server._response_prefix_cache_bytes,
            "dflash_global_prefix_cache_entries": len(server._global_prefix_order),
            "dflash_global_prefix_cache_bytes": server._global_prefix_cache_bytes,
            "dflash_stable_prefix_tokens_bytes": server._stable_prefix_tokens_bytes,
            "dflash_global_prefix_cache_hits": server._global_prefix_cache_hits,
            "dflash_global_prefix_cache_misses": server._global_prefix_cache_misses,
            "dflash_request_metrics_entries": len(server._request_metrics_order),
            "mlx_active_memory_bytes": active_memory,
            "mlx_cache_memory_bytes": cache_memory,
            "mlx_peak_memory_bytes": peak_memory,
            "dflash_context_window": server.context_window or 0,
            "dflash_max_tokens_limit": server.max_tokens_limit or 0,
            "dflash_block_size": server.block_size,
            "dflash_generation_engine_ddtree": 1 if server.generation_engine == "ddtree" else 0,
            "dflash_ddtree_tree_budget": server.ddtree_tree_budget,
            "dflash_ddtree_target_turboquant_bits": float(server.ddtree_target_turboquant_bits or 0),
            "dflash_adaptive_block_size_enabled": 1 if server.adaptive_block_size_config.enabled else 0,
            "dflash_adaptive_block_size_min": server.adaptive_block_size_config.min_block_size,
            "dflash_adaptive_block_size_max": server.adaptive_block_size_config.max_block_size,
            "dflash_adaptive_block_size_grow_threshold": server.adaptive_block_size_config.grow_threshold,
            "dflash_adaptive_block_size_shrink_threshold": server.adaptive_block_size_config.shrink_threshold,
            "dflash_adaptive_block_size_grow_step": server.adaptive_block_size_config.grow_step,
            "dflash_adaptive_block_size_shrink_step": server.adaptive_block_size_config.shrink_step,
            "dflash_adaptive_block_size_grow_streak": server.adaptive_block_size_config.grow_streak,
            "dflash_adaptive_block_size_shrink_streak": server.adaptive_block_size_config.shrink_streak,
            "dflash_keep_alive_seconds": float(server.keep_alive_seconds or 0),
            "dflash_last_used_at": float(server._last_used_at or 0),
            **{
                f"dflash_last_request_{key}": value
                for key, value in (server._last_request_metrics or {}).items()
                if isinstance(value, (int, float))
            },
        }

    @app.get("/requests")
    def request_metrics(limit: int = 20) -> dict[str, Any]:
        bounded_limit = max(1, min(int(limit), max(1, REQUEST_METRICS_HISTORY_LIMIT)))
        entries = list(server._request_metrics_order)[-bounded_limit:]
        return {"count": len(entries), "entries": entries}

    @app.get("/runs")
    def runs(dir: str = "") -> dict[str, Any]:
        """Return the last entries from `.agent-queue/run.jsonl` in the given
        workdir. This is what the autonomy queue appends per task start/end,
        budget events, replan triggers, etc."""
        if not dir:
            raise HTTPException(status_code=400, detail="missing 'dir' query param")
        path = os.path.join(dir, ".agent-queue", "run.jsonl")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"no run.jsonl at {path}")
        entries: list[dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        entries = entries[-500:]
        return {"dir": dir, "count": len(entries), "entries": entries}

    @app.get("/")
    @app.head("/")
    def root() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "local-dflash-api",
            "model": server.model_name,
        }

    from fastapi.responses import JSONResponse as _JSONResponse

    _MODELS_ETAG = f"W/\"local-dflash-{server.model_name}-v1\""
    _MODELS_HEADERS = {"X-Models-Etag": _MODELS_ETAG, "ETag": _MODELS_ETAG}

    @app.get("/v1/models")
    def list_models():
        payload = {
            "object": "list",
            "data": [_model_detail_payload(server)],
        }
        # Direct return for test harnesses that call the endpoint as a plain
        # function; JSONResponse wrapping attaches the ETag headers in real
        # HTTP flows.
        return _JSONResponse(content=payload, headers=_MODELS_HEADERS)

    @app.get("/v1/models/{model_id}")
    def get_model(model_id: str):
        if model_id != server.model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        return _JSONResponse(
            content=_model_detail_payload(server),
            headers=_MODELS_HEADERS,
        )

    @app.get("/api/v1/models")
    def lm_studio_models() -> dict[str, Any]:
        return {
            "models": [_lm_studio_model_payload(server)],
            "data": [_model_detail_payload(server)],
        }

    @app.get("/api/tags")
    def ollama_tags() -> dict[str, Any]:
        return {
            "models": [_ollama_model_payload(server)],
        }

    @app.get("/v1/props")
    @app.get("/props")
    def llamacpp_props() -> dict[str, Any]:
        return _llamacpp_props_payload(server)

    @app.get("/version")
    def version() -> dict[str, Any]:
        return {"version": "local-dflash/0.2.0"}

    @app.post("/v1/chat/completions")
    def chat_completions(req: OpenAIChatRequest) -> dict[str, Any]:
        tools = _filter_disabled_tools(req.tools or [])
        has_tools = bool(tools)
        sampling = SamplingParams.for_request(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            presence_penalty=req.presence_penalty,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            has_tools=has_tools,
        )
        if req.stream:
            if req.model != server.model_name:
                raise HTTPException(status_code=404, detail=f"Unknown model: {req.model}")
            max_tokens = req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_TOKENS_FALLBACK
            chat_messages = _massage_chat_tool_result_messages(
                [m.model_dump() for m in req.messages]
            )
            return StreamingResponse(
                server.stream_chat_completions(
                    chat_messages,
                    max_tokens,
                    sampling,
                    tools=tools,
                    keep_alive_override=req.keep_alive,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        if req.model != server.model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model: {req.model}")

        max_tokens = req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_TOKENS_FALLBACK
        chat_messages = _massage_chat_tool_result_messages(
            [m.model_dump() for m in req.messages]
        )
        try:
            result = server.generate(
                chat_messages,
                max_tokens,
                sampling,
                tools=tools,
                keep_alive_override=req.keep_alive,
            )
        except PromptTooLargeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _, assistant_text = _strip_reasoning_blocks(result["text"])
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": server.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_text,
                    },
                    "finish_reason": result["finish_reason"],
                }
            ],
            "usage": {
                "completion_tokens": result["generated_tokens"],
                "prompt_tokens": result["prompt_tokens"],
                "total_tokens": result["prompt_tokens"] + result["generated_tokens"],
            },
            "metrics": _response_metrics(result),
            "raw_text": result["text"],
        }
        _trace_event("chat_completions.completed", payload)
        return payload

    @app.post("/v1/messages")
    def anthropic_messages(req: AnthropicRequest) -> Any:
        if DEFAULT_TRACE_FILE:
            _trace_request("messages", req.model_dump(mode="json"))
        if req.model != server.model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model: {req.model}")

        messages, tools = _normalize_anthropic_messages(req)
        max_tokens = req.max_tokens
        has_tools = bool(tools)
        sampling = SamplingParams.for_request(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            presence_penalty=req.presence_penalty,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            has_tools=has_tools,
        )

        if req.stream:
            return StreamingResponse(
                server.stream_anthropic_events(messages, max_tokens, sampling, tools=tools, keep_alive_override=req.keep_alive),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result = server.generate(messages, max_tokens, sampling, tools=tools, keep_alive_override=req.keep_alive)
        except PromptTooLargeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        content_blocks = _build_anthropic_content_blocks(result["text"])
        payload = _build_anthropic_message_payload(
            message_id=f"msg_{uuid.uuid4().hex}",
            model_name=server.model_name,
            result=result,
            content_blocks=content_blocks,
        )
        _trace_event("messages.completed", {"raw_text": result["text"], "response": payload})
        return payload

    @app.post("/v1/messages/count_tokens")
    def anthropic_count_tokens(req: AnthropicCountTokensRequest) -> dict[str, Any]:
        messages, _ = _normalize_anthropic_messages(req)
        text_parts = [_extract_text_from_content(message.get("content")) for message in messages]
        approx_tokens = max(1, sum(len(part.split()) for part in text_parts if part))
        return {"input_tokens": approx_tokens}

    @app.post("/v1/responses")
    def responses(req: ResponsesRequest) -> Any:
        if DEFAULT_TRACE_FILE:
            _trace_request("responses", req.model_dump(mode="json"))

        if req.model != server.model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model: {req.model}")

        request_messages, request_tools = _normalize_responses_input(req)
        try:
            messages, tools = server.resolve_responses_context(
                request_messages=request_messages,
                request_tools=request_tools,
                previous_response_id=req.previous_response_id,
            )
        except UnknownPreviousResponseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        messages = _massage_responses_continuation_messages(messages)
        messages = _massage_responses_tool_result_messages(messages)
        messages = _synthesize_orphan_tool_results(messages)
        max_tokens = _responses_max_tokens(req.max_output_tokens, tools)
        has_tools = bool(tools)
        sampling = SamplingParams.for_request(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            min_p=req.min_p,
            presence_penalty=req.presence_penalty,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            has_tools=has_tools,
        )

        if req.stream:
            return StreamingResponse(
                server.stream_response_events(
                    messages,
                    max_tokens,
                    sampling,
                    tools=tools,
                    request_messages=request_messages,
                    previous_response_id=req.previous_response_id,
                    keep_alive_override=req.keep_alive,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result, output_items = server.generate_response(
                messages,
                max_tokens,
                sampling,
                tools=tools,
                keep_alive_override=req.keep_alive,
                previous_response_id=req.previous_response_id,
                capture_prompt_cache_state=True,
            )
        except PromptTooLargeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response_id = f"resp_{uuid.uuid4().hex}"
        server.remember_response(
            response_id=response_id,
            previous_response_id=req.previous_response_id,
            request_messages=request_messages,
            tools=tools,
            output_items=output_items,
            prompt_cache_state=result.get("prompt_cache_state"),
        )
        response_status, incomplete_details = _response_completion_state(result)
        payload = _build_response_payload(
            response_id=response_id,
            model_name=server.model_name,
            result=result,
            output_items=output_items,
            status=response_status,
            previous_response_id=req.previous_response_id,
            incomplete_details=incomplete_details,
        )
        _trace_event(
            f"responses.{response_status}",
            {
                "response_id": response_id,
                "previous_response_id": req.previous_response_id,
                "max_output_tokens": max_tokens,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "presence_penalty": sampling.presence_penalty,
                "repetition_penalty": sampling.repetition_penalty,
                "request_messages": request_messages,
                "tools": tools,
                "raw_text": result["text"],
                "response": payload,
            },
        )
        return payload

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local API for the MLX DFlash setup.")
    parser.add_argument("--host", default=os.environ.get("LOCAL_DFLASH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LOCAL_DFLASH_PORT", "8010")))
    parser.add_argument("--model-path", default=os.environ.get("LOCAL_DFLASH_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--draft-path", default=os.environ.get("LOCAL_DFLASH_DRAFT_PATH", DEFAULT_DRAFT_PATH))
    parser.add_argument("--model-name", default=os.environ.get("LOCAL_DFLASH_MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--block-size", type=int, default=int(os.environ.get("LOCAL_DFLASH_BLOCK_SIZE", "8")))
    parser.add_argument(
        "--adaptive-block-size",
        action="store_true",
        default=_env_bool("LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE", False),
        help="Enable adaptive speculative block size based on recent acceptance ratios.",
    )
    parser.add_argument(
        "--adaptive-block-size-min",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_MIN", "4")),
        help="Lower bound for adaptive speculative block size.",
    )
    parser.add_argument(
        "--adaptive-block-size-max",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_MAX", "12")),
        help="Upper bound for adaptive speculative block size.",
    )
    parser.add_argument(
        "--adaptive-block-size-grow-threshold",
        type=float,
        default=float(os.environ.get("LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_GROW_THRESHOLD", "0.95")),
        help="Grow block size when accepted/proposed ratio stays at or above this threshold.",
    )
    parser.add_argument(
        "--adaptive-block-size-shrink-threshold",
        type=float,
        default=float(os.environ.get("LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_SHRINK_THRESHOLD", "0.65")),
        help="Shrink block size when accepted/proposed ratio falls at or below this threshold.",
    )
    parser.add_argument(
        "--adaptive-block-size-grow-step",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_GROW_STEP", "1")),
        help="How many tokens to add when the adaptive block size grows.",
    )
    parser.add_argument(
        "--adaptive-block-size-shrink-step",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_SHRINK_STEP", "2")),
        help="How many tokens to remove when the adaptive block size shrinks.",
    )
    parser.add_argument(
        "--adaptive-block-size-grow-streak",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_GROW_STREAK", "2")),
        help="Consecutive high-acceptance steps required before growing block size.",
    )
    parser.add_argument(
        "--adaptive-block-size-shrink-streak",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_ADAPTIVE_BLOCK_SIZE_SHRINK_STREAK", "1")),
        help="Consecutive low-acceptance steps required before shrinking block size.",
    )
    parser.add_argument(
        "--generation-engine",
        choices=("dflash", "ddtree"),
        default=os.environ.get("LOCAL_DFLASH_ENGINE", "dflash").strip().lower(),
        help="Generation engine to use. ddtree uses experimental MLX DDTree verification.",
    )
    parser.add_argument(
        "--ddtree-tree-budget",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_DDTREE_TREE_BUDGET", "4")),
        help="Maximum DDTree candidate nodes to verify per speculative step.",
    )
    parser.add_argument(
        "--ddtree-target-turboquant-bits",
        type=float,
        default=(
            float(os.environ["LOCAL_DFLASH_DDTREE_TARGET_TURBOQUANT_BITS"])
            if os.environ.get("LOCAL_DFLASH_DDTREE_TARGET_TURBOQUANT_BITS")
            else (
                float(os.environ["LOCAL_DFLASH_TURBOQUANT_BITS"])
                if os.environ.get("LOCAL_DFLASH_TURBOQUANT_BITS")
                else None
            )
        ),
        help="Optional TurboQuant bit width for DDTree target KV cache. Use 0 to disable.",
    )
    parser.add_argument(
        "--ddtree-no-fallback",
        action="store_true",
        default=_env_bool("LOCAL_DFLASH_DDTREE_NO_FALLBACK", False),
        help="Raise DDTree errors instead of falling back to standard DFlash generation.",
    )
    parser.add_argument(
        "--ddtree-no-turboquant-retry",
        action="store_true",
        default=_env_bool("LOCAL_DFLASH_DDTREE_NO_TURBOQUANT_RETRY", False),
        help="Do not retry DDTree with target TurboQuant disabled after a DDTree+TurboQuant failure.",
    )
    parser.add_argument(
        "--global-prefix-cache-limit",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_GLOBAL_PREFIX_CACHE_LIMIT", str(GLOBAL_PREFIX_CACHE_LIMIT))),
        help="Maximum number of stable global prefix snapshots to keep in memory.",
    )
    parser.add_argument(
        "--prefix-cache-state-byte-limit-gb",
        type=float,
        default=(
            float(os.environ["LOCAL_DFLASH_PREFIX_CACHE_STATE_BYTE_LIMIT_GB"])
            if os.environ.get("LOCAL_DFLASH_PREFIX_CACHE_STATE_BYTE_LIMIT_GB")
            else PREFIX_CACHE_STATE_BYTE_LIMIT / (1024 ** 3)
        ),
        help="Byte ceiling (GiB) for per-response prompt-cache snapshots.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=os.environ.get("LOCAL_DFLASH_DISABLE_THINKING", "").lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--sliding-window-size",
        type=int,
        default=(int(os.environ["LOCAL_DFLASH_SLIDING_WINDOW_SIZE"]) if os.environ.get("LOCAL_DFLASH_SLIDING_WINDOW_SIZE") else 4096),
    )
    parser.add_argument(
        "--rotating-keep-tokens",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_ROTATING_KEEP_TOKENS", "1024")),
        help="Number of leading tokens (typically the system prompt) the rotating KV cache must never evict. Prevents Codex base_instructions from being dropped mid-session.",
    )
    parser.add_argument(
        "--max-tokens-limit",
        type=int,
        default=(int(os.environ["LOCAL_DFLASH_MAX_TOKENS"]) if os.environ.get("LOCAL_DFLASH_MAX_TOKENS") else 8192),
        help="Optional hard cap applied after context-window checks. Leave unset to use the full available context.",
    )
    parser.add_argument(
        "--context-reserve",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_CONTEXT_RESERVE", "512")),
        help="Token margin reserved to avoid hitting the absolute context edge.",
    )
    parser.add_argument(
        "--context-window-override",
        type=int,
        default=(int(os.environ["LOCAL_DFLASH_CONTEXT_WINDOW"]) if os.environ.get("LOCAL_DFLASH_CONTEXT_WINDOW") else 32768),
        help="Optional logical context cap. Requests above this token budget are rejected before generation.",
    )
    parser.add_argument(
        "--keep-alive-seconds",
        default=os.environ.get("LOCAL_DFLASH_KEEP_ALIVE"),
        help="Idle time before unloading the model. Use 0 to unload immediately, -1 to keep loaded forever, or values like 30, 10m, 1h.",
    )
    parser.add_argument(
        "--target-turboquant-bits",
        type=float,
        default=(float(os.environ["LOCAL_DFLASH_TURBOQUANT_BITS"]) if os.environ.get("LOCAL_DFLASH_TURBOQUANT_BITS") else 4.0),
        help="Optional TurboQuant bit width for the target model's KV cache on compatible full-attention layers. Use values like 4 or 3.5.",
    )
    parser.add_argument(
        "--draft-turboquant-bits",
        type=float,
        default=(
            float(os.environ["LOCAL_DFLASH_DRAFT_TURBOQUANT_BITS"])
            if os.environ.get("LOCAL_DFLASH_DRAFT_TURBOQUANT_BITS")
            else (
                float(os.environ["LOCAL_DFLASH_TURBOQUANT_BITS"])
                if os.environ.get("LOCAL_DFLASH_TURBOQUANT_BITS")
                else None
            )
        ),
        help="Optional TurboQuant bit width for the draft model's KV cache. Defaults to the target TurboQuant setting when unset.",
    )
    parser.add_argument(
        "--mlx-memory-limit-gb",
        type=float,
        default=(float(os.environ["LOCAL_DFLASH_MLX_MEMORY_LIMIT_GB"]) if os.environ.get("LOCAL_DFLASH_MLX_MEMORY_LIMIT_GB") else None),
        help="Optional MLX memory limit in GiB. Exceeding it raises an allocation error instead of growing indefinitely.",
    )
    parser.add_argument(
        "--mlx-cache-limit-gb",
        type=float,
        default=(float(os.environ["LOCAL_DFLASH_MLX_CACHE_LIMIT_GB"]) if os.environ.get("LOCAL_DFLASH_MLX_CACHE_LIMIT_GB") else None),
        help="Optional MLX free-cache limit in GiB. Set 0 to disable allocator cache retention.",
    )
    parser.add_argument(
        "--mlx-wired-limit-gb",
        type=float,
        default=(float(os.environ["LOCAL_DFLASH_MLX_WIRED_LIMIT_GB"]) if os.environ.get("LOCAL_DFLASH_MLX_WIRED_LIMIT_GB") else None),
        help="Optional MLX wired-memory (non-swappable) limit in GiB. Default MLX wires ~75% of RAM which on large Macs is dangerous; set this to ~60%% of total RAM for stable 24h runs.",
    )
    parser.add_argument(
        "--mlx-clear-cache-threshold",
        type=float,
        default=float(os.environ.get("LOCAL_DFLASH_MLX_CLEAR_CACHE_THRESHOLD", "0.9")),
        help="When cache memory exceeds this fraction of the cache limit, call mx.clear_cache() between requests.",
    )
    parser.add_argument(
        "--global-prefix-cache-byte-limit-gb",
        type=float,
        default=(float(os.environ["LOCAL_DFLASH_GLOBAL_PREFIX_CACHE_BYTE_LIMIT_GB"]) if os.environ.get("LOCAL_DFLASH_GLOBAL_PREFIX_CACHE_BYTE_LIMIT_GB") else 4.0),
        help="Byte ceiling (GiB) for the global prompt-cache state dict. Prevents slow leaks over 24h.",
    )
    parser.add_argument(
        "--stable-prefix-tokens-byte-limit-gb",
        type=float,
        default=(float(os.environ["LOCAL_DFLASH_STABLE_PREFIX_TOKENS_BYTE_LIMIT_GB"]) if os.environ.get("LOCAL_DFLASH_STABLE_PREFIX_TOKENS_BYTE_LIMIT_GB") else 2.0),
        help="Byte ceiling (GiB) for the stable-prefix token memoization dict.",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        default=os.environ.get("LOCAL_DFLASH_NO_PRELOAD", "").lower() in {"1", "true", "yes", "on"},
        help="Start the HTTP server before loading the model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mlx_memory_limit = _gb_to_bytes(args.mlx_memory_limit_gb)
    if mlx_memory_limit is not None:
        mx.set_memory_limit(mlx_memory_limit)
    mlx_cache_limit = _gb_to_bytes(args.mlx_cache_limit_gb)
    if mlx_cache_limit is not None:
        mx.set_cache_limit(mlx_cache_limit)
    mlx_wired_limit = _gb_to_bytes(args.mlx_wired_limit_gb)
    if mlx_wired_limit is not None:
        try:
            mx.set_wired_limit(mlx_wired_limit)
        except AttributeError:
            _logger.warning(
                "mx.set_wired_limit is unavailable in this MLX build; "
                "skipping wired-memory cap (%d GiB requested).",
                args.mlx_wired_limit_gb,
            )
        except Exception as exc:
            _logger.warning(
                "mx.set_wired_limit(%d GiB) failed: %s",
                args.mlx_wired_limit_gb,
                exc,
            )

    detected_context_window = _detect_context_window(args.model_path)
    context_window = args.context_window_override or detected_context_window
    adaptive_block_size_config = AdaptiveBlockSizeConfig(
        enabled=args.adaptive_block_size,
        min_block_size=max(1, args.adaptive_block_size_min),
        max_block_size=max(1, args.adaptive_block_size_max),
        grow_threshold=args.adaptive_block_size_grow_threshold,
        shrink_threshold=args.adaptive_block_size_shrink_threshold,
        grow_step=max(1, args.adaptive_block_size_grow_step),
        shrink_step=max(1, args.adaptive_block_size_shrink_step),
        grow_streak=max(1, args.adaptive_block_size_grow_streak),
        shrink_streak=max(1, args.adaptive_block_size_shrink_streak),
    )
    response_prefix_byte_limit = _gb_to_bytes(getattr(args, "prefix_cache_state_byte_limit_gb", None) or 0) or 0
    prefix_byte_limit = _gb_to_bytes(getattr(args, "global_prefix_cache_byte_limit_gb", None) or 0) or 0
    stable_prefix_byte_limit = _gb_to_bytes(getattr(args, "stable_prefix_tokens_byte_limit_gb", None) or 0) or 0
    server = LocalModelServer(
        model_path=args.model_path,
        draft_path=args.draft_path,
        model_name=args.model_name,
        block_size=args.block_size,
        disable_thinking=args.disable_thinking,
        sliding_window_size=args.sliding_window_size,
        rotating_keep_tokens=args.rotating_keep_tokens,
        max_tokens_limit=args.max_tokens_limit,
        context_window=context_window,
        context_reserve=args.context_reserve,
        keep_alive_seconds=_parse_keep_alive(args.keep_alive_seconds),
        target_turboquant_bits=args.target_turboquant_bits,
        draft_turboquant_bits=args.draft_turboquant_bits,
        adaptive_block_size_config=adaptive_block_size_config,
        generation_engine=args.generation_engine,
        ddtree_tree_budget=args.ddtree_tree_budget,
        ddtree_target_turboquant_bits=args.ddtree_target_turboquant_bits,
        ddtree_fallback_to_dflash=not args.ddtree_no_fallback,
        ddtree_retry_without_turboquant=not args.ddtree_no_turboquant_retry,
        global_prefix_cache_limit=max(0, args.global_prefix_cache_limit),
        prefix_cache_state_byte_limit=response_prefix_byte_limit,
        global_prefix_cache_byte_limit=prefix_byte_limit,
        stable_prefix_tokens_byte_limit=stable_prefix_byte_limit,
        mlx_clear_cache_threshold=args.mlx_clear_cache_threshold,
    )
    if not args.no_preload:
        server.ensure_loaded()
    app = create_app(server)
    uvicorn_kwargs: dict[str, Any] = {
        "host": args.host,
        "port": args.port,
        "log_level": "info",
        "access_log": False,
    }
    if importlib.util.find_spec("uvloop") is not None:
        uvicorn_kwargs["loop"] = "uvloop"
    if importlib.util.find_spec("httptools") is not None:
        uvicorn_kwargs["http"] = "httptools"
    uvicorn.run(app, **uvicorn_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
