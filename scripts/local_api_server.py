#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from queue import Empty, Queue
import re
import time
import uuid
from threading import RLock, Thread, Timer
from typing import Any, Iterator, Literal

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from dflash.model_mlx import load, load_draft, stream_generate


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


def _env_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return default if value <= 0 else value


STREAM_HEARTBEAT_SECONDS = _env_positive_float("LOCAL_DFLASH_STREAM_HEARTBEAT_SECONDS", 2.0)


class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] | str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    max_tokens: int | None = 512
    max_completion_tokens: int | None = None
    temperature: float = 0.0
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
    max_tokens: int = 512
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float = 0.0
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
    max_output_tokens: int | None = 512
    temperature: float = 0.0
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    previous_response_id: str | None = None
    include: list[str] | None = None
    reasoning: dict[str, Any] | None = None
    keep_alive: str | int | float | None = None
    model_config = ConfigDict(extra="ignore")


class PromptTooLargeError(ValueError):
    pass


def _trace_request(kind: str, payload: dict[str, Any]) -> None:
    if not DEFAULT_TRACE_FILE:
        return
    event = {
        "ts": time.time(),
        "kind": kind,
        "payload": payload,
    }
    with open(DEFAULT_TRACE_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _json_line(event: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"


def _done_line() -> str:
    return "data: [DONE]\n\n"


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


def _extract_visible_text(text: str) -> str:
    cleaned = text
    for token in SPECIAL_TOKENS:
        cleaned = cleaned.replace(token, "")

    visible_parts: list[str] = []
    cursor = 0
    while cursor < len(cleaned):
        think_idx = cleaned.find("<think>", cursor)
        next_idx = len(cleaned)
        next_end: str | None = None
        next_start_len = 0
        next_is_think = False

        if think_idx != -1 and think_idx < next_idx:
            next_idx = think_idx
            next_end = "</think>"
            next_start_len = len("<think>")
            next_is_think = True

        for start_marker, end_marker in TOOL_BLOCK_MARKERS:
            marker_idx = cleaned.find(start_marker, cursor)
            if marker_idx != -1 and marker_idx < next_idx:
                next_idx = marker_idx
                next_end = end_marker
                next_start_len = len(start_marker)
                next_is_think = False

        if next_end is None:
            visible_parts.append(cleaned[cursor:])
            break

        visible_parts.append(cleaned[cursor:next_idx])
        end_idx = cleaned.find(next_end, next_idx + next_start_len)
        if end_idx == -1:
            break
        cursor = end_idx + len(next_end)

    return "".join(visible_parts)


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
    parsed_blocks: list[tuple[int, list[dict[str, Any]]]] = []

    for match in TOOL_CALL_RE.finditer(cleaned):
        params: dict[str, Any] = {}
        for param_match in PARAM_RE.finditer(match.group("body")):
            params[param_match.group("name").strip()] = _parse_param_value(param_match.group("value"))
        parsed_blocks.append(
            (
                match.start(),
                [
                    _make_function_call_item(
                        match.group("name").strip(),
                        params,
                    )
                ],
            )
        )

    for match in TAGGED_TOOL_CALL_RE.finditer(cleaned):
        parsed = _tool_call_items_from_payload(match.group("body"))
        if parsed:
            parsed_blocks.append((match.start(), parsed))

    for match in FENCED_TOOL_CALL_RE.finditer(cleaned):
        parsed = _tool_call_items_from_payload(match.group("body"))
        if parsed:
            parsed_blocks.append((match.start(), parsed))

    tool_calls: list[dict[str, Any]] = []
    for _, items in sorted(parsed_blocks, key=lambda entry: entry[0]):
        tool_calls.extend(items)

    visible_text = _extract_visible_text(cleaned)
    return _clean_output_text(visible_text), tool_calls


def _make_internal_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    return {
        "function": {
            "name": name,
            "arguments": _coerce_tool_arguments(arguments),
        }
    }


def _response_usage(result: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": result["prompt_tokens"],
        "output_tokens": result["generated_tokens"],
        "total_tokens": result["prompt_tokens"] + result["generated_tokens"],
    }


def _response_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tps": result["prompt_tps"],
        "generation_tps": result["generation_tps"],
        "peak_memory_gb": result["peak_memory_gb"],
        "elapsed": result["elapsed"],
    }


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


def _build_output_items(full_text: str) -> list[dict[str, Any]]:
    _, visible_text = _strip_reasoning_blocks(full_text)
    assistant_text, tool_calls = _parse_tool_calls(visible_text)
    items: list[dict[str, Any]] = []
    if assistant_text:
        items.append(_make_message_item(assistant_text))
    items.extend(tool_calls)
    if not items:
        items.append(_make_message_item(""))
    return items


def _normalize_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools or []:
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
                    {
                        "role": "tool",
                        "content": _extract_text_from_content(block_data.get("content")),
                    }
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
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model_name,
        "output": output_items,
        "output_text": _output_text_from_items(output_items),
        "usage": _response_usage(result),
        "metrics": _response_metrics(result),
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


def _normalize_responses_input(req: ResponsesRequest) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    tools = req.tools or []
    system_parts: list[str] = []

    if req.instructions:
        system_parts.append(req.instructions)

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
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": tool_name,
                                "arguments": _coerce_tool_arguments(item.get("arguments")),
                            }
                        }
                    ],
                }
            )
            continue

        if item_type in {"function_call_output", "custom_tool_call_output"}:
            output = item.get("output")
            if not output:
                output = item.get("content")
            messages.append(
                {
                    "role": "tool",
                    "content": _extract_text_from_content(output),
                }
            )
            continue

        if item_type == "reasoning":
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
    ) -> None:
        self.model_path = model_path
        self.draft_path = draft_path
        self.model_name = model_name
        self.block_size = block_size
        self.disable_thinking = disable_thinking
        self.sliding_window_size = sliding_window_size
        self.max_tokens_limit = max_tokens_limit
        self.context_window = context_window
        self.context_reserve = context_reserve
        self.keep_alive_seconds = keep_alive_seconds
        self.target_turboquant_bits = (
            None if target_turboquant_bits is not None and target_turboquant_bits <= 0 else target_turboquant_bits
        )
        self._lock = RLock()
        self._model = None
        self._draft = None
        self._tokenizer = None
        self._unload_timer: Timer | None = None
        self._last_used_at: float | None = None

    def _cancel_unload_timer_locked(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def unload(self) -> None:
        with self._lock:
            self._cancel_unload_timer_locked()
            self._model = None
            self._draft = None
            self._tokenizer = None
            gc.collect()
            mx.clear_cache()

    def _schedule_unload_locked(self, keep_alive_seconds: float | None) -> None:
        self._cancel_unload_timer_locked()
        if self._model is None:
            return
        if keep_alive_seconds is None:
            return
        if keep_alive_seconds <= 0:
            self._model = None
            self._draft = None
            self._tokenizer = None
            gc.collect()
            mx.clear_cache()
            return

        timer = Timer(keep_alive_seconds, self.unload)
        timer.daemon = True
        timer.start()
        self._unload_timer = timer

    def finish_request(self, keep_alive_override: Any = None) -> None:
        keep_alive_seconds = (
            self.keep_alive_seconds if keep_alive_override is None else _parse_keep_alive(keep_alive_override)
        )
        with self._lock:
            self._last_used_at = time.time()
            self._schedule_unload_locked(keep_alive_seconds)

    def ensure_loaded(self) -> None:
        self._cancel_unload_timer_locked()
        if self._model is not None:
            return
        self._model, self._tokenizer = load(self.model_path)
        self._draft = load_draft(self.draft_path, sliding_window_size=self.sliding_window_size)

    def build_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                enable_thinking=not self.disable_thinking,
                **kwargs,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def _effective_max_tokens(self, requested_max_tokens: int, prompt: str) -> tuple[int, int]:
        prompt_tokens = len(self._tokenizer.encode(prompt))
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

        return min(candidates), prompt_tokens

    def _generate_locked(
        self,
        messages: list[dict[str, Any]],
        requested_max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        self.ensure_loaded()
        prompt = self.build_prompt(messages, tools=tools)
        max_tokens, prompt_tokens = self._effective_max_tokens(requested_max_tokens, prompt)
        text_parts: list[str] = []
        final = None
        started = time.time()
        for chunk in stream_generate(
            self._model,
            self._draft,
            self._tokenizer,
            prompt,
            block_size=self.block_size,
            max_tokens=max_tokens,
            temperature=temperature,
            target_turboquant_bits=self.target_turboquant_bits,
        ):
            if chunk.text:
                text_parts.append(chunk.text)
            final = chunk
        if final is None:
            raise RuntimeError("Model returned no output")
        result = {
            "text": "".join(text_parts),
            "finish_reason": final.finish_reason or "stop",
            "prompt_tokens": prompt_tokens,
            "prompt_tps": final.prompt_tps,
            "generation_tps": final.generation_tps,
            "generated_tokens": final.generation_tokens,
            "peak_memory_gb": final.peak_memory,
            "elapsed": time.time() - started,
        }
        return text_parts, result

    def _stream_generate_locked(
        self,
        messages: list[dict[str, Any]],
        requested_max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[Iterator[Any], int, float]:
        self.ensure_loaded()
        prompt = self.build_prompt(messages, tools=tools)
        max_tokens, prompt_tokens = self._effective_max_tokens(requested_max_tokens, prompt)
        started = time.time()
        iterator = stream_generate(
            self._model,
            self._draft,
            self._tokenizer,
            prompt,
            block_size=self.block_size,
            max_tokens=max_tokens,
            temperature=temperature,
            target_turboquant_bits=self.target_turboquant_bits,
        )
        return iterator, prompt_tokens, started

    def _generation_worker(
        self,
        queue: Queue,
        messages: list[dict[str, Any]],
        requested_max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        keep_alive_override: Any = None,
    ) -> None:
        text_parts: list[str] = []
        final = None
        prompt_tokens = 0
        started = time.time()
        try:
            with self._lock:
                iterator, prompt_tokens, started = self._stream_generate_locked(
                    messages,
                    requested_max_tokens,
                    temperature,
                    tools=tools,
                )
                for chunk in iterator:
                    if chunk.text:
                        text_parts.append(chunk.text)
                        queue.put(("text", chunk.text))
                    final = chunk

            if final is None:
                raise RuntimeError("Model returned no output")

            result = {
                "text": "".join(text_parts),
                "finish_reason": final.finish_reason or "stop",
                "prompt_tokens": prompt_tokens,
                "prompt_tps": final.prompt_tps,
                "generation_tps": final.generation_tps,
                "generated_tokens": final.generation_tokens,
                "peak_memory_gb": final.peak_memory,
                "elapsed": time.time() - started,
            }
            queue.put(("result", result))
        except Exception as exc:
            queue.put(("error", str(exc)))
        finally:
            self.finish_request(keep_alive_override)
            queue.put(("done", None))

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        keep_alive_override: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            _, result = self._generate_locked(messages, max_tokens, temperature, tools=tools)
        self.finish_request(keep_alive_override)
        return result

    def stream_response_events(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        keep_alive_override: Any = None,
    ) -> Iterator[str]:
        response_id = f"resp_{uuid.uuid4().hex}"
        created = {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": int(time.time()),
                "status": "in_progress",
                "model": self.model_name,
            },
        }
        yield _json_line("response.created", created)
        yield _json_line(
            "response.server_model",
            {
                "type": "response.server_model",
                "response_id": response_id,
                "model": self.model_name,
            },
        )
        yield _json_line(
            "response.server_reasoning_included",
            {
                "type": "response.server_reasoning_included",
                "response_id": response_id,
                "included": False,
            },
        )
        event_queue: Queue = Queue()
        worker = Thread(
            target=self._generation_worker,
            args=(event_queue, messages, max_tokens, temperature, tools, keep_alive_override),
            daemon=True,
        )
        worker.start()

        full_text = ""
        streamed_visible = ""
        message_item_id: str | None = None
        result: dict[str, Any] | None = None
        done = False

        while not done:
            try:
                kind, payload = event_queue.get(timeout=STREAM_HEARTBEAT_SECONDS)
            except Empty:
                yield _comment_line()
                continue

            if kind == "text":
                full_text += payload
                current_visible = _extract_visible_text(full_text)
                if current_visible.startswith(streamed_visible):
                    delta = current_visible[len(streamed_visible):]
                else:
                    delta = current_visible
                if not delta:
                    continue
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
                streamed_visible = current_visible
                yield _json_line(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "output_index": 0,
                        "item_id": message_item_id,
                        "content_index": 0,
                        "delta": delta,
                    },
                )
                continue

            if kind == "result":
                result = payload
                full_text = result["text"]
                continue

            if kind == "error":
                yield _json_line(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "status": "failed",
                            "model": self.model_name,
                        },
                        "error": {
                            "message": payload,
                        },
                    },
                )
                yield _done_line()
                return

            if kind == "done":
                done = True

        if result is None:
            yield _json_line(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "failed",
                        "model": self.model_name,
                    },
                    "error": {
                        "message": "Generation completed without a final result",
                    },
                },
            )
            yield _done_line()
            return

        output_items = _build_output_items(full_text)

        next_output_index = 0
        for item in output_items:
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
                        next_output_index += 1
                else:
                    item["id"] = message_item_id
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

            yield _json_line(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": next_output_index,
                    "item": item,
                },
            )
            if item["type"] == "function_call" and item.get("arguments"):
                yield _json_line(
                    "response.tool_call_input.delta",
                    {
                        "type": "response.tool_call_input.delta",
                        "response_id": response_id,
                        "output_index": next_output_index,
                        "item_id": item["id"],
                        "delta": item["arguments"],
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

        completed_payload = _build_response_payload(
            response_id=response_id,
            model_name=self.model_name,
            result=result,
            output_items=output_items,
            status="completed",
        )
        yield _json_line(
            "response.completed",
            {
                "type": "response.completed",
                "response": completed_payload,
            },
        )
        yield _done_line()

    def stream_anthropic_events(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        keep_alive_override: Any = None,
    ) -> Iterator[str]:
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
        event_queue: Queue = Queue()
        worker = Thread(
            target=self._generation_worker,
            args=(event_queue, messages, max_tokens, temperature, tools, keep_alive_override),
            daemon=True,
        )
        worker.start()

        full_text = ""
        streamed_visible = ""
        text_block_open = False
        result: dict[str, Any] | None = None
        done = False

        while not done:
            try:
                kind, payload = event_queue.get(timeout=STREAM_HEARTBEAT_SECONDS)
            except Empty:
                yield _comment_line()
                continue

            if kind == "text":
                full_text += payload
                current_visible = _extract_visible_text(full_text)
                if current_visible.startswith(streamed_visible):
                    delta = current_visible[len(streamed_visible):]
                else:
                    delta = current_visible
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
                streamed_visible = current_visible
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
                full_text = result["text"]
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

        content_blocks = _build_anthropic_content_blocks(full_text)
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
        return {
            "status": "ok",
            "model": server.model_name,
            "loaded": server._model is not None,
            "context_window": server.context_window,
            "context_reserve": server.context_reserve,
            "block_size": server.block_size,
            "disable_thinking": server.disable_thinking,
            "sliding_window_size": server.sliding_window_size,
            "max_tokens_limit": server.max_tokens_limit,
            "keep_alive_seconds": server.keep_alive_seconds,
            "stream_heartbeat_seconds": STREAM_HEARTBEAT_SECONDS,
            "target_turboquant_bits": server.target_turboquant_bits,
            "last_used_at": server._last_used_at,
            "active_memory_gb": mx.get_active_memory() / (1024 ** 3),
            "cache_memory_gb": mx.get_cache_memory() / (1024 ** 3),
            "peak_memory_gb": mx.get_peak_memory() / (1024 ** 3),
        }

    @app.get("/")
    @app.head("/")
    def root() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "local-dflash-api",
            "model": server.model_name,
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": server.model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: OpenAIChatRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="Chat Completions streaming is not implemented")
        if req.model != server.model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model: {req.model}")

        max_tokens = req.max_completion_tokens or req.max_tokens or 512
        try:
            result = server.generate(
                [m.model_dump() for m in req.messages],
                max_tokens,
                req.temperature,
                keep_alive_override=req.keep_alive,
            )
        except PromptTooLargeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _, assistant_text = _strip_reasoning_blocks(result["text"])
        return {
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
        }

    @app.post("/v1/messages")
    def anthropic_messages(req: AnthropicRequest) -> Any:
        _trace_request("messages", req.model_dump(mode="json"))
        if req.model != server.model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model: {req.model}")

        messages, tools = _normalize_anthropic_messages(req)
        max_tokens = req.max_tokens

        if req.stream:
            return StreamingResponse(
                server.stream_anthropic_events(messages, max_tokens, req.temperature, tools=tools, keep_alive_override=req.keep_alive),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result = server.generate(messages, max_tokens, req.temperature, tools=tools, keep_alive_override=req.keep_alive)
        except PromptTooLargeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        content_blocks = _build_anthropic_content_blocks(result["text"])
        return _build_anthropic_message_payload(
            message_id=f"msg_{uuid.uuid4().hex}",
            model_name=server.model_name,
            result=result,
            content_blocks=content_blocks,
        )

    @app.post("/v1/messages/count_tokens")
    def anthropic_count_tokens(req: AnthropicCountTokensRequest) -> dict[str, Any]:
        messages, _ = _normalize_anthropic_messages(req)
        text_parts = [_extract_text_from_content(message.get("content")) for message in messages]
        approx_tokens = max(1, sum(len(part.split()) for part in text_parts if part))
        return {"input_tokens": approx_tokens}

    @app.post("/v1/responses")
    def responses(req: ResponsesRequest) -> Any:
        _trace_request("responses", req.model_dump(mode="json"))

        if req.model != server.model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model: {req.model}")

        messages, tools = _normalize_responses_input(req)
        max_tokens = req.max_output_tokens or 512

        if req.stream:
            return StreamingResponse(
                server.stream_response_events(messages, max_tokens, req.temperature, tools=tools, keep_alive_override=req.keep_alive),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result = server.generate(messages, max_tokens, req.temperature, tools=tools, keep_alive_override=req.keep_alive)
        except PromptTooLargeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        output_items = _build_output_items(result["text"])
        return _build_response_payload(
            response_id=f"resp_{uuid.uuid4().hex}",
            model_name=server.model_name,
            result=result,
            output_items=output_items,
            status="completed",
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local API for the MLX DFlash setup.")
    parser.add_argument("--host", default=os.environ.get("LOCAL_DFLASH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LOCAL_DFLASH_PORT", "8010")))
    parser.add_argument("--model-path", default=os.environ.get("LOCAL_DFLASH_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--draft-path", default=os.environ.get("LOCAL_DFLASH_DRAFT_PATH", DEFAULT_DRAFT_PATH))
    parser.add_argument("--model-name", default=os.environ.get("LOCAL_DFLASH_MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--block-size", type=int, default=int(os.environ.get("LOCAL_DFLASH_BLOCK_SIZE", "15")))
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
        "--max-tokens-limit",
        type=int,
        default=(int(os.environ["LOCAL_DFLASH_MAX_TOKENS"]) if os.environ.get("LOCAL_DFLASH_MAX_TOKENS") else 8192),
        help="Optional hard cap applied after context-window checks. Leave unset to use the full available context.",
    )
    parser.add_argument(
        "--context-reserve",
        type=int,
        default=int(os.environ.get("LOCAL_DFLASH_CONTEXT_RESERVE", "256")),
        help="Token margin reserved to avoid hitting the absolute context edge.",
    )
    parser.add_argument(
        "--context-window-override",
        type=int,
        default=(int(os.environ["LOCAL_DFLASH_CONTEXT_WINDOW"]) if os.environ.get("LOCAL_DFLASH_CONTEXT_WINDOW") else 65536),
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

    detected_context_window = _detect_context_window(args.model_path)
    context_window = args.context_window_override or detected_context_window
    server = LocalModelServer(
        model_path=args.model_path,
        draft_path=args.draft_path,
        model_name=args.model_name,
        block_size=args.block_size,
        disable_thinking=args.disable_thinking,
        sliding_window_size=args.sliding_window_size,
        max_tokens_limit=args.max_tokens_limit,
        context_window=context_window,
        context_reserve=args.context_reserve,
        keep_alive_seconds=_parse_keep_alive(args.keep_alive_seconds),
        target_turboquant_bits=args.target_turboquant_bits,
    )
    if not args.no_preload:
        server.ensure_loaded()
    app = create_app(server)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
