from __future__ import annotations

import importlib.util
import json
import threading
from types import SimpleNamespace
import unittest
from pathlib import Path
from queue import Queue
from unittest import mock

import mlx.core as mx


import sys as _sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "local_api_server.py"
SPEC = importlib.util.spec_from_file_location("local_api_server", MODULE_PATH)
local_api_server = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
# Register in sys.modules BEFORE exec_module so @dataclass decorators can
# resolve class.__module__ during class creation.
_sys.modules["local_api_server"] = local_api_server
SPEC.loader.exec_module(local_api_server)


def _sse_event_names(events: str) -> list[str]:
    names: list[str] = []
    for line in events.splitlines():
        if line.startswith("event: "):
            names.append(line.removeprefix("event: ").strip())
    return names


def _sse_payloads(events: str, event_name: str) -> list[dict]:
    payloads: list[dict] = []
    current_event: str | None = None
    for line in events.splitlines():
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ").strip()
            continue
        if current_event == event_name and line.startswith("data: "):
            payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


class FakeStreamingServer(local_api_server.LocalModelServer):
    @staticmethod
    def _result():
        return {
            "text": '<function_call>{"name":"write_file","arguments":{"path":"index.html","content":"ok"}}</function_call>',
            "finish_reason": "stop",
            "prompt_tokens": 12,
            "prefill_seconds": 0.1,
            "prompt_tps": 10.0,
            "reused_prefix_tokens": 4,
            "decode_seconds": 0.2,
            "generation_tps": 9.0,
            "generated_tokens": 8,
            "speculative_steps": 3,
            "proposed_tokens": 9,
            "accepted_tokens": 7,
            "avg_acceptance_length": 2.33,
            "avg_acceptance_ratio": 0.77,
            "acceptance_lengths": [2, 2, 3],
            "acceptance_ratios": [0.66, 0.66, 1.0],
            "block_size_history": [3, 3, 3],
            "adaptive_block_size": True,
            "prefix_cache_source": "global",
            "peak_memory_gb": 1.0,
            "elapsed": 0.5,
        }

    def _generation_worker(self, queue: Queue, *args, **kwargs) -> None:
        queue.put(("result", self._result()))
        queue.put(("done", None))

    def generate_response(self, messages, max_tokens, *args, **kwargs):
        return self._result(), local_api_server._build_output_items(self._result()["text"])


class FakeHangingResponsesServer(local_api_server.LocalModelServer):
    def _responses_generation_worker(self, queue: Queue, *args, **kwargs) -> None:
        stop_event = kwargs.get("stop_event")
        if stop_event is None and args:
            stop_event = args[-1]
        if stop_event is not None:
            stop_event.wait(1.0)


class FakeCustomToolStreamingServer(local_api_server.LocalModelServer):
    @staticmethod
    def _result():
        return {
            "text": (
                '<function_call>{"name":"apply_patch","arguments":'
                '{"input":"*** Begin Patch\\n*** Add File: hello.txt\\n+hello\\n*** End Patch\\n"}}'
                "</function_call>"
            ),
            "finish_reason": "stop",
            "prompt_tokens": 12,
            "prefill_seconds": 0.1,
            "prompt_tps": 10.0,
            "reused_prefix_tokens": 4,
            "decode_seconds": 0.2,
            "generation_tps": 9.0,
            "generated_tokens": 8,
            "speculative_steps": 3,
            "proposed_tokens": 9,
            "accepted_tokens": 7,
            "avg_acceptance_length": 2.33,
            "avg_acceptance_ratio": 0.77,
            "acceptance_lengths": [2, 2, 3],
            "acceptance_ratios": [0.66, 0.66, 1.0],
            "block_size_history": [3, 3, 3],
            "adaptive_block_size": True,
            "prefix_cache_source": "global",
            "peak_memory_gb": 1.0,
            "elapsed": 0.5,
        }

    def generate_response(self, messages, max_tokens, *args, tools=None, **kwargs):
        result = self._result()
        output_items = local_api_server._build_output_items(result["text"])
        output_items = local_api_server._convert_items_for_custom_tools(output_items, tools)
        return result, output_items


class FakeTruncatedToolCallServer(local_api_server.LocalModelServer):
    @staticmethod
    def _result():
        return {
            "text": (
                "I'll create the file now.\n\n"
                "<tool_call>\n"
                "<function=write_file>\n"
                "<parameter=path>\n"
                "\"index.html\"\n"
                "</parameter>\n"
                "<parameter=content>\n"
                "<!DOCTYPE html>\n"
                "<html>"
            ),
            "finish_reason": "length",
            "prompt_tokens": 12,
            "prefill_seconds": 0.1,
            "prompt_tps": 10.0,
            "reused_prefix_tokens": 4,
            "decode_seconds": 0.2,
            "generation_tps": 9.0,
            "generated_tokens": 8,
            "speculative_steps": 3,
            "proposed_tokens": 9,
            "accepted_tokens": 7,
            "avg_acceptance_length": 2.33,
            "avg_acceptance_ratio": 0.77,
            "acceptance_lengths": [2, 2, 3],
            "acceptance_ratios": [0.66, 0.66, 1.0],
            "block_size_history": [3, 3, 3],
            "adaptive_block_size": True,
            "prefix_cache_source": "global",
            "peak_memory_gb": 1.0,
            "elapsed": 0.5,
            "prompt_cache_state": None,
        }

    def _generation_worker(self, queue: Queue, *args, **kwargs) -> None:
        queue.put(("result", self._result()))
        queue.put(("done", None))

    def generate(self, messages, max_tokens, *args, **kwargs):
        return self._result()

    def generate_response(self, messages, max_tokens, *args, **kwargs):
        result = self._result()
        return result, local_api_server._build_output_items(result["text"])


class FakeChatStreamingServer(local_api_server.LocalModelServer):
    def _generation_worker(self, queue: Queue, *args, **kwargs) -> None:
        queue.put(("text", "Hello"))
        queue.put(("text", " world"))
        result = {
            "text": "Hello world",
            "finish_reason": "stop",
            "prompt_tokens": 8,
            "prefill_seconds": 0.05,
            "prompt_tps": 10.0,
            "reused_prefix_tokens": 0,
            "decode_seconds": 0.15,
            "generation_tps": 9.0,
            "generated_tokens": 2,
            "speculative_steps": 1,
            "proposed_tokens": 2,
            "accepted_tokens": 2,
            "avg_acceptance_length": 2.0,
            "avg_acceptance_ratio": 1.0,
            "acceptance_lengths": [2],
            "acceptance_ratios": [1.0],
            "block_size_history": [2],
            "adaptive_block_size": False,
            "prefix_cache_source": "none",
            "peak_memory_gb": 1.0,
            "elapsed": 0.2,
        }
        queue.put(("result", result))
        queue.put(("done", None))


class FakeStreamErrorServer(local_api_server.LocalModelServer):
    def _generation_worker(self, queue: Queue, *args, **kwargs) -> None:
        queue.put(("error", "boom"))
        queue.put(("done", None))

    def _responses_generation_worker(self, queue: Queue, *args, **kwargs) -> None:
        queue.put(("error", "boom"))
        queue.put(("done", None))


class FakeStreamNoResultServer(local_api_server.LocalModelServer):
    def _generation_worker(self, queue: Queue, *args, **kwargs) -> None:
        queue.put(("done", None))

    def _responses_generation_worker(self, queue: Queue, *args, **kwargs) -> None:
        queue.put(("done", None))


class FakeReasoningResponsesServer(local_api_server.LocalModelServer):
    @staticmethod
    def _result():
        return {
            "text": "The user wants me to inspect the repo. I'll plan the work carefully.</think>Visible answer.",
            "finish_reason": "stop",
            "prompt_tokens": 8,
            "prefill_seconds": 0.05,
            "prompt_tps": 10.0,
            "reused_prefix_tokens": 0,
            "decode_seconds": 0.15,
            "generation_tps": 9.0,
            "generated_tokens": 2,
            "speculative_steps": 1,
            "proposed_tokens": 2,
            "accepted_tokens": 2,
            "avg_acceptance_length": 2.0,
            "avg_acceptance_ratio": 1.0,
            "acceptance_lengths": [2],
            "acceptance_ratios": [1.0],
            "block_size_history": [2],
            "adaptive_block_size": False,
            "prefix_cache_source": "none",
            "peak_memory_gb": 1.0,
            "elapsed": 0.2,
            "prompt_cache_state": None,
        }

    def _generation_worker(self, queue: Queue, *args, **kwargs) -> None:
        queue.put(("text", "The user wants me to inspect the repo. "))
        queue.put(("text", "I'll plan the work carefully.</think>Visible answer."))
        queue.put(("result", self._result()))
        queue.put(("done", None))

    def generate_response(self, messages, max_tokens, *args, **kwargs):
        result = self._result()
        return result, local_api_server._build_output_items(result["text"])


class FakeTimer:
    def __init__(self, interval, func) -> None:
        self.interval = interval
        self.func = func
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.func()


class TrackingServer(local_api_server.LocalModelServer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clear_request_state_calls = 0
        self.reset_loaded_state_calls = 0

    def _clear_request_state_locked(self) -> None:
        self.clear_request_state_calls += 1

    def _reset_loaded_state_locked(self) -> None:
        self.reset_loaded_state_calls += 1
        self._model = None
        self._draft = None
        self._tokenizer = None


class LocalApiServerTests(unittest.TestCase):
    def _make_server(self, cls=local_api_server.LocalModelServer, keep_alive_seconds=0):
        return cls(
            model_path="model",
            draft_path="draft",
            model_name="local-test-model",
            block_size=8,
            disable_thinking=True,
            sliding_window_size=256,
            max_tokens_limit=1024,
            context_window=4096,
            context_reserve=128,
            keep_alive_seconds=keep_alive_seconds,
            target_turboquant_bits=None,
            draft_turboquant_bits=None,
        )

    def _get_health_endpoint(self, server):
        app = local_api_server.create_app(server)
        for route in app.routes:
            if getattr(route, "path", None) == "/health":
                return route.endpoint
        self.fail("Health endpoint not found")

    def _get_endpoint(self, server, path):
        app = local_api_server.create_app(server)
        for route in app.routes:
            if getattr(route, "path", None) == path:
                return route.endpoint
        self.fail(f"Endpoint not found: {path}")

    def _make_generation_chunk(self, *, prefill_state=None):
        return SimpleNamespace(
            text="ok",
            finish_reason="stop",
            prefill_seconds=0.1,
            prompt_tps=10.0,
            reused_prefix_tokens=0,
            decode_seconds=0.2,
            generation_tps=9.0,
            generation_tokens=2,
            speculative_steps=1,
            proposed_tokens=2,
            accepted_tokens=2,
            avg_acceptance_length=2.0,
            avg_acceptance_ratio=1.0,
            acceptance_lengths=(),
            acceptance_ratios=(),
            block_size_history=(),
            adaptive_block_size=False,
            prefill_hidden_bytes=11,
            prefill_target_cache_bytes=22,
            prefill_logits_bytes=33,
            prefill_working_set_bytes=66,
            prompt_cache_state_bytes=77,
            peak_memory=1.5,
            prefill_state=prefill_state,
        )

    def test_previous_response_id_restores_context_and_tools(self):
        server = self._make_server()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "parameters": {"type": "object"},
                },
            }
        ]

        first_request_messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Inspect the repository."},
        ]
        first_output_items = [
            local_api_server._make_message_item("I'll inspect the repository."),
            local_api_server._make_function_call_item("search_files", {"path": "."}, call_id="call_1"),
        ]
        server.remember_response(
            response_id="resp_1",
            previous_response_id=None,
            request_messages=first_request_messages,
            tools=tools,
            output_items=first_output_items,
        )

        second_request_messages = [
            {"role": "tool", "content": '[{"path":"README.md"}]'},
        ]
        merged_messages, merged_tools = server.resolve_responses_context(
            request_messages=second_request_messages,
            request_tools=[],
            previous_response_id="resp_1",
        )

        self.assertEqual(merged_messages[0]["role"], "system")
        self.assertEqual(merged_messages[0]["content"], "You are a coding agent.")
        self.assertEqual(merged_messages[1], {"role": "user", "content": "Inspect the repository."})
        self.assertEqual(merged_messages[2], {"role": "assistant", "content": "I'll inspect the repository."})
        self.assertEqual(merged_messages[3]["role"], "assistant")
        self.assertEqual(
            merged_messages[3]["tool_calls"][0]["function"]["name"],
            "search_files",
        )
        self.assertEqual(merged_messages[3]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(merged_messages[3]["tool_calls"][0]["type"], "function")
        self.assertEqual(merged_messages[4], {"role": "tool", "content": '[{"path":"README.md"}]'})
        self.assertEqual(merged_tools, tools)

    def test_messages_from_output_items_restores_custom_tool_call(self):
        output_items = [
            local_api_server._make_custom_tool_call_item(
                "apply_patch",
                "*** Begin Patch\n*** End Patch\n",
                call_id="call_patch",
            )
        ]

        messages = local_api_server._messages_from_output_items(output_items)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_patch")
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], "apply_patch")
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            {"input": "*** Begin Patch\n*** End Patch\n"},
        )

    def test_normalize_responses_input_preserves_tool_call_ids(self):
        req = SimpleNamespace(
            tools=None,
            instructions=None,
            input=[
                {
                    "type": "function_call",
                    "name": "search_files",
                    "call_id": "call_123",
                    "arguments": {"path": "."},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": '[{"path":"README.md"}]',
                },
            ],
        )

        messages, tools = local_api_server._normalize_responses_input(req)

        self.assertEqual(tools, [])
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["id"], "call_123")
        self.assertEqual(messages[0]["tool_calls"][0]["type"], "function")
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], "search_files")
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["tool_call_id"], "call_123")
        self.assertEqual(messages[1]["content"], '[{"path":"README.md"}]')

    def test_normalize_responses_input_drops_reasoning_when_tools_are_present(self):
        req = SimpleNamespace(
            tools=[{"type": "function", "function": {"name": "shell_command"}}],
            tool_choice=None,
            instructions=None,
            input=[
                {"type": "message", "role": "user", "content": "continue"},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "reasoning summary"}],
                },
            ],
        )

        messages, tools = local_api_server._normalize_responses_input(req)

        self.assertTrue(tools)
        self.assertFalse(any("<think>" in message.get("content", "") for message in messages))

    def test_build_output_items_deduplicates_identical_consecutive_tool_calls(self):
        output_items = local_api_server._build_output_items(
            (
                '<function_call>{"name":"search_files","arguments":{"path":"."}}</function_call>\n'
                '<function_call>{"name":"search_files","arguments":{"path":"."}}</function_call>'
            )
        )

        function_calls = [item for item in output_items if item["type"] == "function_call"]
        self.assertEqual(len(function_calls), 1)
        self.assertEqual(function_calls[0]["name"], "search_files")
        self.assertEqual(function_calls[0]["call_id"][:5], "call_")

    def test_build_output_items_drops_text_when_tool_call_exists(self):
        output_items = local_api_server._build_output_items(
            (
                "I will inspect the repo now.\n"
                '<function_call>{"name":"search_files","arguments":{"path":"."}}</function_call>'
            )
        )

        self.assertEqual([item["type"] for item in output_items], ["function_call"])
        self.assertEqual(output_items[0]["name"], "search_files")

    def test_generate_locked_skips_prefill_capture_when_caches_are_disabled(self):
        server = self._make_server()
        server.prefix_cache_state_limit = 0
        server.global_prefix_cache_limit = 0
        server.ensure_loaded = mock.Mock()
        server.build_prompt = mock.Mock(return_value="prompt")
        server.tokenize_prompt = mock.Mock(return_value=[1, 2, 3])
        chunk = self._make_generation_chunk(prefill_state=object())
        observed = {}

        def fake_stream_generate(*args, **kwargs):
            observed["capture_prefill_state"] = kwargs["capture_prefill_state"]
            yield chunk

        with mock.patch.object(local_api_server, "stream_generate", side_effect=fake_stream_generate):
            _, result = server._generate_locked(
                messages=[{"role": "user", "content": "hello"}],
                requested_max_tokens=16,
                temperature=0.0,
                capture_prompt_cache_state=True,
            )

        self.assertFalse(observed["capture_prefill_state"])
        self.assertIsNone(result["prompt_cache_state"])
        self.assertEqual(result["prefill_hidden_bytes"], 11)
        self.assertEqual(result["prefill_target_cache_bytes"], 22)
        self.assertEqual(result["prefill_logits_bytes"], 33)
        self.assertEqual(result["prefill_working_set_bytes"], 66)
        self.assertEqual(result["prompt_cache_state_bytes"], 77)

    def test_generate_locked_skips_full_prompt_capture_for_global_cache_only(self):
        server = self._make_server()
        server.prefix_cache_state_limit = 0
        server.global_prefix_cache_limit = 1
        server.ensure_loaded = mock.Mock()
        server.build_prompt = mock.Mock(return_value="prompt")
        server.tokenize_prompt = mock.Mock(return_value=[1, 2, 3])
        server._stable_prefix_tokens_locked = mock.Mock(return_value=(1,))
        server._remember_global_prefix_state_locked = mock.Mock()
        chunk = self._make_generation_chunk(prefill_state=object())
        observed = {}

        def fake_stream_generate(*args, **kwargs):
            observed["capture_prefill_state"] = kwargs["capture_prefill_state"]
            yield chunk

        with mock.patch.object(local_api_server, "stream_generate", side_effect=fake_stream_generate):
            _, result = server._generate_locked(
                messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
                requested_max_tokens=16,
                temperature=0.0,
                capture_prompt_cache_state=True,
            )

        self.assertFalse(observed["capture_prefill_state"])
        self.assertIsNone(result["prompt_cache_state"])
        server._remember_global_prefix_state_locked.assert_called_once()

    def test_generate_locked_captures_prefill_state_when_prefix_cache_is_enabled(self):
        server = self._make_server()
        server.prefix_cache_state_limit = 1
        server.ensure_loaded = mock.Mock()
        server.build_prompt = mock.Mock(return_value="prompt")
        server.tokenize_prompt = mock.Mock(return_value=[1, 2, 3])
        chunk = self._make_generation_chunk(prefill_state=object())
        observed = {}

        def fake_stream_generate(*args, **kwargs):
            observed["capture_prefill_state"] = kwargs["capture_prefill_state"]
            yield chunk

        with mock.patch.object(local_api_server, "stream_generate", side_effect=fake_stream_generate):
            _, result = server._generate_locked(
                messages=[{"role": "user", "content": "hello"}],
                requested_max_tokens=16,
                temperature=0.0,
                capture_prompt_cache_state=True,
            )

        self.assertTrue(observed["capture_prefill_state"])
        self.assertIs(result["prompt_cache_state"], chunk.prefill_state)

    def test_tool_calling_rules_prompt_omits_apply_patch_when_tool_absent(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "shell_command",
                    "parameters": {"type": "object"},
                },
            }
        ]

        prompt = local_api_server._tool_calling_rules_prompt(tools)

        self.assertNotIn("apply_patch", prompt)
        self.assertIn("Tool-calling rules (strict):", prompt)
        self.assertIn("Function calls MUST", prompt)
        self.assertIn("<function=tool_name>", prompt)
        self.assertIn("shell_command", prompt)
        self.assertIn("required: command, workdir, timeout_ms", prompt)
        self.assertIn("parameters: command, workdir, timeout_ms", prompt)
        self.assertNotIn("pathlib", prompt)
        self.assertNotIn("dev server", prompt)
        self.assertNotIn("test/build", prompt)

    def test_build_prompt_injects_tool_calling_rules_for_any_tool_surface(self):
        server = self._make_server()

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(
                    _message.get("content", "")
                    for _message in messages
                    if _message.get("role") == "system"
                )

        server._tokenizer = FakeTokenizer()
        prompt = server.build_prompt(
            [{"role": "user", "content": "inspect repo"}],
            tools=[{"type": "function", "function": {"name": "shell_command"}}],
        )

        self.assertIn("Tool-calling rules (strict):", prompt)
        self.assertIn("<function=tool_name>", prompt)
        self.assertIn("Do NOT add prose before or after a tool call.", prompt)
        self.assertIn("shell_command", prompt)
        self.assertNotIn("Do not use long-running dev servers", prompt)
        self.assertNotIn("Do not print raw patches", prompt)

    def test_build_prompt_disables_thinking_for_tool_turns(self):
        server = self._make_server()
        server.disable_thinking = False
        observed: dict[str, Any] = {}

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                observed.update(kwargs)
                return "prompt"

        server._tokenizer = FakeTokenizer()
        server.build_prompt(
            [{"role": "user", "content": "inspect repo"}],
            tools=[{"type": "function", "function": {"name": "shell_command"}}],
        )

        self.assertFalse(observed["enable_thinking"])
        self.assertFalse(observed["preserve_thinking"])

    def test_tool_calling_rules_injection_deduplicates_existing_system_rules(self):
        messages = [
            {
                "role": "system",
                "content": "abc\n\nTool-calling rules (strict):\nexisting",
            },
            {"role": "user", "content": "inspect repo"},
        ]

        result = local_api_server._ensure_tool_calling_rules_message(
            messages,
            [{"type": "function", "function": {"name": "shell_command"}}],
        )

        self.assertIs(result, messages)

    def test_tool_choice_none_disables_tools(self):
        tools = [{"type": "function", "function": {"name": "shell_command"}}]

        selected, directive = local_api_server._apply_tool_choice_to_tools(tools, "none")

        self.assertEqual(selected, [])
        self.assertIsNone(directive)

    def test_tool_choice_required_adds_protocol_directive(self):
        tools = [{"type": "function", "function": {"name": "shell_command"}}]

        selected, directive = local_api_server._apply_tool_choice_to_tools(tools, "required")

        self.assertEqual(selected, tools)
        self.assertIn("requires a tool call", directive)

    def test_tool_choice_specific_filters_to_named_tool(self):
        tools = [
            {"type": "function", "function": {"name": "shell_command"}},
            {"type": "function", "function": {"name": "read_file"}},
        ]

        selected, directive = local_api_server._apply_tool_choice_to_tools(
            tools,
            {"type": "function", "function": {"name": "read_file"}},
        )

        self.assertEqual([local_api_server._tool_name(tool) for tool in selected], ["read_file"])
        self.assertIn("`read_file`", directive)

    def test_normalize_tool_schemas_accepts_openai_anthropic_responses_and_custom(self):
        normalized = local_api_server._normalize_tool_schemas(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "openai_tool",
                        "description": "OpenAI nested",
                        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                    },
                },
                {
                    "name": "anthropic_tool",
                    "description": "Anthropic flat",
                    "input_schema": {"type": "object", "properties": {"y": {"type": "string"}}},
                },
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "freeform",
                },
            ]
        )

        self.assertEqual(normalized[0]["function"]["name"], "openai_tool")
        self.assertEqual(normalized[1]["function"]["name"], "anthropic_tool")
        self.assertEqual(normalized[1]["function"]["parameters"]["properties"]["y"]["type"], "string")
        self.assertEqual(normalized[2]["type"], "custom")
        self.assertEqual(normalized[2]["name"], "apply_patch")

    def test_normalize_tool_schemas_requires_timeout_for_shell_tools(self):
        normalized = local_api_server._normalize_tool_schemas(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "shell_command",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ]
        )

        parameters = normalized[0]["function"]["parameters"]
        self.assertIn("workdir", parameters["properties"])
        self.assertIn("workdir", parameters["required"])
        self.assertIn("timeout_ms", parameters["properties"])
        self.assertIn("timeout_ms", parameters["required"])

    def test_tool_argument_sanitizer_coerces_literals_and_removes_impossible_escalation(self):
        item = local_api_server._make_function_call_item(
            "shell_command",
            {
                "command": "npm install",
                "timeout_ms": "120000",
                "disableTimeout": "True",
                "sandbox_permissions": "require_escalated",
                "justification": "Need approval",
            },
        )
        args = json.loads(item["arguments"])

        self.assertTrue(args["disableTimeout"])
        self.assertNotIn("sandbox_permissions", args)
        self.assertNotIn("justification", args)

    def test_normalize_openai_messages_preserves_agentic_tool_turns(self):
        messages = [
            local_api_server.OpenAIMessage(role="developer", content="dev rules"),
            local_api_server.OpenAIMessage(role="user", content=[{"type": "text", "text": "inspect"}]),
            local_api_server.OpenAIMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "shell_command",
                            "arguments": {"cmd": "pwd"},
                        },
                    }
                ],
            ),
            local_api_server.OpenAIMessage(
                role="tool",
                tool_call_id="call_1",
                name="shell_command",
                content={"output": "/tmp/project"},
            ),
        ]

        normalized = local_api_server._normalize_openai_messages(messages)

        self.assertEqual(normalized[0], {"role": "system", "content": "dev rules"})
        self.assertEqual(normalized[1], {"role": "user", "content": "inspect"})
        self.assertEqual(normalized[2]["role"], "assistant")
        self.assertEqual(normalized[2]["content"], "")
        self.assertEqual(normalized[2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(normalized[3]["role"], "tool")
        self.assertEqual(normalized[3]["tool_call_id"], "call_1")
        self.assertIn("/tmp/project", normalized[3]["content"])

    def test_parse_tool_calls_detailed_accepts_xml_json_fenced_and_marks_incomplete(self):
        xml = local_api_server._parse_tool_calls_detailed(
            "<tool_call>\n<function=shell_command>\n<parameter=cmd>\npwd\n</parameter>\n</function>\n</tool_call>"
        )
        tagged_json = local_api_server._parse_tool_calls_detailed(
            '<tool_call>{"name":"shell_command","arguments":{"cmd":"pwd"}}</tool_call>'
        )
        fenced = local_api_server._parse_tool_calls_detailed(
            '```tool_call\n{"name":"shell_command","arguments":{"cmd":"pwd"}}\n```'
        )
        incomplete = local_api_server._parse_tool_calls_detailed(
            "<tool_call>\n<function=shell_command>\n<parameter=cmd>\npwd"
        )

        self.assertEqual(xml.tool_calls[0]["name"], "shell_command")
        self.assertEqual(json.loads(tagged_json.tool_calls[0]["arguments"])["cmd"], "pwd")
        self.assertEqual(fenced.tool_calls[0]["name"], "shell_command")
        self.assertEqual(xml.formats, ["qwen_xml"])
        self.assertEqual(tagged_json.formats, ["tagged_json"])
        self.assertEqual(fenced.formats, ["fenced_json"])
        self.assertFalse(xml.incomplete)
        self.assertTrue(incomplete.incomplete)
        self.assertEqual(incomplete.tool_calls, [])

    def test_sampling_clamps_tool_turn_temperature_to_agentic_range(self):
        sampling = local_api_server.SamplingParams.for_request(
            temperature=0.8,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            repetition_penalty=None,
            frequency_penalty=None,
            has_tools=True,
        )

        self.assertLessEqual(sampling.temperature, local_api_server.MAX_TEMPERATURE_WITH_TOOLS)

    def test_agentic_metrics_record_tool_retry_and_parse_format(self):
        result = dict(FakeStreamingServer._result())
        local_api_server._annotate_agentic_metrics(
            result,
            tools=[{"type": "function", "function": {"name": "write_file"}}],
            protocol_no_tool_retries=2,
            protocol_malformed_tool_retries=1,
        )

        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(result["tool_call_parse_format"], "tagged_json")
        self.assertEqual(result["tool_call_parse_format_code"], 2)
        self.assertEqual(result["protocol_no_tool_retries"], 2)
        self.assertEqual(result["protocol_malformed_tool_retries"], 1)
        self.assertEqual(result["protocol_final_with_tools"], 0)

    def test_responses_tool_choice_required_is_added_to_system_message(self):
        req = SimpleNamespace(
            model="local-test-model",
            instructions=None,
            input="inspect repo",
            tools=[{"type": "function", "function": {"name": "shell_command"}}],
            tool_choice="required",
        )

        messages, tools = local_api_server._normalize_responses_input(req)

        self.assertEqual(local_api_server._available_tool_names(tools), {"shell_command"})
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Tool choice requires a tool call", messages[0]["content"])

    def test_filter_disabled_tools_removes_apply_patch_by_default(self):
        tools = [
            {"type": "function", "function": {"name": "apply_patch"}},
            {"type": "function", "function": {"name": "shell_command"}},
        ]

        filtered = local_api_server._filter_disabled_tools(tools)

        self.assertEqual(local_api_server._available_tool_names(filtered), {"shell_command"})

    def test_build_output_items_drops_disabled_apply_patch_calls(self):
        output_items = local_api_server._build_output_items(
            '<function_call>{"name":"apply_patch","arguments":{"input":"bad"}}</function_call>'
        )

        self.assertEqual(len(output_items), 1)
        self.assertEqual(output_items[0]["type"], "message")
        self.assertEqual(output_items[0]["content"][0]["text"], "")

    def test_tool_calling_rules_prompt_mentions_apply_patch_when_available(self):
        with mock.patch.object(local_api_server, "ALLOW_APPLY_PATCH_TOOL", True):
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "parameters": {"type": "object"},
                    },
                }
            ]

            prompt = local_api_server._tool_calling_rules_prompt(tools)

        self.assertIn("apply_patch", prompt)
        self.assertIn("Tool-calling rules (strict):", prompt)
        self.assertNotIn("pathlib", prompt)

    def test_filter_disabled_tools_keeps_apply_patch_when_enabled(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "parameters": {"type": "object"},
                },
            }
        ]

        with mock.patch.object(local_api_server, "ALLOW_APPLY_PATCH_TOOL", True):
            filtered = local_api_server._filter_disabled_tools(tools)

        self.assertEqual(local_api_server._available_tool_names(filtered), {"apply_patch"})

    def test_responses_max_tokens_raises_floor_when_tools_are_available(self):
        self.assertEqual(
            local_api_server._responses_max_tokens(512, [{"type": "function"}]),
            local_api_server.MIN_TOOL_RESPONSE_MAX_TOKENS,
        )
        floor = local_api_server.MIN_TOOL_RESPONSE_MAX_TOKENS
        self.assertEqual(
            local_api_server._responses_max_tokens(floor + 4096, [{"type": "function"}]),
            floor + 4096,
        )
        self.assertEqual(
            local_api_server._responses_max_tokens(512, None),
            512,
        )

    def test_invalid_requested_max_tokens_raise_http_400(self):
        with self.assertRaises(local_api_server.HTTPException) as responses_exc:
            local_api_server._responses_max_tokens(0, None)
        self.assertEqual(responses_exc.exception.status_code, 400)

        with self.assertRaises(local_api_server.HTTPException) as chat_exc:
            local_api_server._validate_requested_max_tokens(-1, "max_tokens")
        self.assertEqual(chat_exc.exception.status_code, 400)

    def test_exec_tool_arguments_use_codex_command_field(self):
        sanitized = local_api_server._sanitize_function_call_arguments(
            "exec",
            {"cmd": "npm run build", "workdir": "/tmp/app", "timeout_ms": 120000},
        )

        self.assertEqual(sanitized["command"], "npm run build")
        self.assertNotIn("cmd", sanitized)
        self.assertEqual(sanitized["workdir"], "/tmp/app")
        self.assertEqual(sanitized["timeout_ms"], 120000)

    def test_non_shell_tool_arguments_do_not_get_command_alias(self):
        sanitized = local_api_server._sanitize_function_call_arguments(
            "custom_tool",
            {"cmd": "literal user field", "path": "file.txt"},
        )

        self.assertEqual(sanitized, {"cmd": "literal user field", "path": "file.txt"})
        self.assertNotIn("command", sanitized)

    def test_shell_tool_argument_aliases_normalize_to_command(self):
        sanitized = local_api_server._sanitize_function_call_arguments(
            "exec",
            {"bash_command": "pwd", "workdir": "/tmp/app", "timeout_ms": 120000},
        )

        self.assertEqual(sanitized["command"], "pwd")
        self.assertNotIn("bash_command", sanitized)

    def test_exec_tool_schema_requires_command_workdir_and_timeout(self):
        normalized = local_api_server._normalize_tool_schemas(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )
        parameters = normalized[0]["function"]["parameters"]

        self.assertIn("command", parameters["properties"])
        self.assertIn("workdir", parameters["properties"])
        self.assertIn("timeout_ms", parameters["properties"])
        self.assertIn("command", parameters["required"])
        self.assertIn("workdir", parameters["required"])
        self.assertIn("timeout_ms", parameters["required"])

    def test_protocol_error_tool_call_uses_exec_command_field(self):
        text = local_api_server._protocol_error_tool_call_text(
            [{"type": "function", "function": {"name": "exec", "parameters": {"type": "object"}}}],
            "timeout",
        )
        _, tool_calls = local_api_server._parse_tool_calls(text)
        args = json.loads(tool_calls[0]["arguments"])

        self.assertEqual(tool_calls[0]["name"], "exec")
        self.assertIn("command", args)
        self.assertNotIn("cmd", args)
        self.assertIn("LOCAL_DFLASH_PROTOCOL_ERROR", args["command"])

    def test_shell_command_sanitizer_preserves_model_command_semantics(self):
        background = local_api_server._sanitize_function_call_arguments(
            "shell_command",
            {"command": "npm run test 2>&1 &", "workdir": "/tmp/app", "timeout_ms": 120000},
        )
        alias = local_api_server._sanitize_function_call_arguments(
            "shell_command",
            {"cmd": "npm run dev", "workdir": "/tmp/app", "timeout_ms": 120000},
        )

        self.assertEqual(background["command"], "npm run test 2>&1 &")
        self.assertEqual(background["timeout_ms"], 120000)
        self.assertEqual(alias["command"], "npm run dev")
        self.assertNotIn("LOCAL_DFLASH_PROTOCOL_ERROR", background["command"])
        self.assertNotIn("LOCAL_DFLASH_PROTOCOL_ERROR", alias["command"])

    def test_shell_tool_call_signature_preserves_workdir_and_canonicalizes_aliases(self):
        from_command = local_api_server._tool_call_signature(
            "shell_command",
            {"command": "pwd", "workdir": "/repo/a"},
        )
        from_alias = local_api_server._tool_call_signature(
            "shell_command",
            {"cmd": "pwd", "workdir": "/repo/a"},
        )
        different_workdir = local_api_server._tool_call_signature(
            "shell_command",
            {"command": "pwd", "workdir": "/repo/b"},
        )

        self.assertEqual(from_command, from_alias)
        self.assertNotEqual(from_command, different_workdir)

    def test_ddtree_generation_receives_cooperative_stop_callback(self):
        server = self._make_server()
        server.generation_engine = "ddtree"
        should_stop = lambda: True

        def fake_generate_ddtree(**kwargs):
            self.assertIs(kwargs["should_stop"], should_stop)
            return {
                "text": "",
                "finish_reason": "cancelled",
                "prompt_tokens": 2,
                "generated_tokens": 0,
                "prompt_cache_state": None,
            }

        with (
            mock.patch.object(server, "ensure_loaded"),
            mock.patch.object(server, "build_prompt", return_value="prompt"),
            mock.patch.object(server, "tokenize_prompt", return_value=[1, 2]),
            mock.patch.object(server, "_stable_prefix_key", return_value="key"),
            mock.patch.object(server, "_stable_prefix_tokens_locked", return_value=(1, 2)),
            mock.patch.object(server, "_select_prefix_state_locked", return_value=(None, "none")),
            mock.patch.object(local_api_server, "generate_ddtree", side_effect=fake_generate_ddtree) as generate_mock,
        ):
            _, result = server._generate_locked(
                [{"role": "user", "content": "hi"}],
                8,
                tools=[{"type": "function", "function": {"name": "run"}}],
                should_stop=should_stop,
            )

        self.assertEqual(result["finish_reason"], "cancelled")
        generate_mock.assert_called_once()

    def test_stream_response_events_emit_standard_function_call_events(self):
        server = self._make_server(FakeStreamingServer)
        events = "".join(
            server.stream_response_events(
                messages=[{"role": "user", "content": "Create index.html"}],
                max_tokens=128,
                temperature=0.0,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                request_messages=[{"role": "user", "content": "Create index.html"}],
            )
        )

        self.assertIn("event: response.created", events)
        self.assertIn("event: response.in_progress", events)
        self.assertIn("event: response.output_item.added", events)
        self.assertIn("event: response.function_call_arguments.delta", events)
        self.assertIn("event: response.function_call_arguments.done", events)
        self.assertIn("event: response.completed", events)

    def test_stream_response_events_function_call_event_order_is_codex_compatible(self):
        server = self._make_server(FakeStreamingServer)
        events = "".join(
            server.stream_response_events(
                messages=[{"role": "user", "content": "Create index.html"}],
                max_tokens=128,
                temperature=0.0,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                request_messages=[{"role": "user", "content": "Create index.html"}],
            )
        )
        names = _sse_event_names(events)

        expected_order = [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
        positions = [names.index(name) for name in expected_order]
        self.assertEqual(positions, sorted(positions))

    def test_stream_response_events_timeout_returns_protocol_error_tool_call(self):
        server = self._make_server(FakeHangingResponsesServer)

        with (
            mock.patch.object(local_api_server, "STREAM_HEARTBEAT_SECONDS", 0.01),
            mock.patch.object(local_api_server, "STREAM_RESULT_TIMEOUT_SECONDS", 0.02),
        ):
            events = "".join(
                server.stream_response_events(
                    messages=[{"role": "user", "content": "Create index.html"}],
                    max_tokens=128,
                    temperature=0.0,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "exec",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    request_messages=[{"role": "user", "content": "Create index.html"}],
                )
            )

        self.assertNotIn("event: response.failed", events)
        self.assertIn("event: response.function_call_arguments.delta", events)
        self.assertIn("event: response.completed", events)
        self.assertIn("LOCAL_DFLASH_PROTOCOL_ERROR", events)
        self.assertIn("timed out before producing a protocol result", events)

    def test_responses_custom_apply_patch_outputs_non_empty_custom_tool_call(self):
        tools = [
            {
                "type": "custom",
                "name": "apply_patch",
            }
        ]
        result = FakeCustomToolStreamingServer._result()
        with mock.patch.object(local_api_server, "ALLOW_APPLY_PATCH_TOOL", True):
            output_items = local_api_server._build_output_items(result["text"])
            output_items = local_api_server._convert_items_for_custom_tools(output_items, tools)

        self.assertEqual(output_items[0]["type"], "custom_tool_call")
        self.assertEqual(output_items[0]["name"], "apply_patch")
        self.assertIn("*** Begin Patch", output_items[0]["input"])
        self.assertNotEqual(output_items[0]["input"].strip(), "")

    def test_stream_response_events_custom_tool_call_emits_input_delta_done(self):
        server = self._make_server(FakeCustomToolStreamingServer)
        with mock.patch.object(local_api_server, "ALLOW_APPLY_PATCH_TOOL", True):
            events = "".join(
                server.stream_response_events(
                    messages=[{"role": "user", "content": "Patch file"}],
                    max_tokens=128,
                    temperature=0.0,
                    tools=[
                        {
                            "type": "custom",
                            "name": "apply_patch",
                        }
                    ],
                    request_messages=[{"role": "user", "content": "Patch file"}],
                )
            )
        names = _sse_event_names(events)
        expected_order = [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.output_item.done",
            "response.completed",
        ]
        positions = [names.index(name) for name in expected_order]
        self.assertEqual(positions, sorted(positions))
        delta = _sse_payloads(events, "response.custom_tool_call_input.delta")[0]
        done = _sse_payloads(events, "response.custom_tool_call_input.done")[0]
        self.assertIn("*** Begin Patch", delta["delta"])
        self.assertEqual(done["input"], delta["delta"])

    def test_generate_response_returns_no_tool_stop_as_final(self):
        server = self._make_server()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "edit",
                    "parameters": {"type": "object"},
                },
            }
        ]
        planning_result = {
            "text": "The test file has a module resolution issue. Let me fix the import path and run the tests again.",
            "finish_reason": "stop",
            "prompt_tokens": 12,
            "prefill_seconds": 0.1,
            "prompt_tps": 10.0,
            "reused_prefix_tokens": 0,
            "decode_seconds": 0.2,
            "generation_tps": 9.0,
            "generated_tokens": 8,
            "speculative_steps": 1,
            "proposed_tokens": 8,
            "accepted_tokens": 8,
            "avg_acceptance_length": 8.0,
            "avg_acceptance_ratio": 1.0,
            "acceptance_lengths": [8],
            "acceptance_ratios": [1.0],
            "block_size_history": [8],
            "adaptive_block_size": False,
            "prefix_cache_source": "none",
            "peak_memory_gb": 1.0,
            "elapsed": 0.2,
            "prompt_cache_state": None,
        }
        with (
            mock.patch.object(
                server,
                "_generate_locked",
                return_value=(["The test file has a module resolution issue."], planning_result),
            ) as generate_locked_mock,
        ):
            result, output_items = server.generate_response(
                messages=[{"role": "user", "content": "Fix until all tests pass."}],
                max_tokens=128,
                temperature=0.0,
                tools=tools,
                keep_alive_override=None,
                previous_response_id="resp_1",
                capture_prompt_cache_state=True,
            )

        self.assertIs(result, planning_result)
        self.assertEqual(output_items[0]["type"], "message")
        self.assertEqual(generate_locked_mock.call_count, 1)
        first_call = generate_locked_mock.call_args_list[0]
        self.assertEqual(first_call.kwargs["previous_response_id"], "resp_1")
        self.assertEqual(first_call.args[1], 128)
        self.assertEqual(result["protocol_no_tool_retries"], 0)

    def test_generate_response_returns_no_tool_text_as_final(self):
        server = self._make_server()
        tools = [{"type": "function", "function": {"name": "shell_command"}}]
        text_result = {
            "text": "The command timed out. Let me inspect the project and continue.",
            "finish_reason": "stop",
            "generated_tokens": 9,
        }
        with (
            mock.patch.object(
                server,
                "_generate_locked",
                return_value=(["The command timed out."], text_result),
            ) as generate_locked_mock,
        ):
            result, output_items = server._generate_response_locked(
                [{"role": "user", "content": "Create Snake."}],
                64,
                tools=tools,
            )

        self.assertIs(result, text_result)
        self.assertEqual(generate_locked_mock.call_count, 1)
        self.assertEqual(output_items[0]["type"], "message")
        self.assertEqual(result["protocol_no_tool_retries"], 0)

    def test_tool_turn_generation_is_capped_but_total_budget_remains(self):
        server = self._make_server()
        tools = [{"type": "function", "function": {"name": "shell_command"}}]
        text_result = {
            "text": "I will keep working.",
            "finish_reason": "length",
            "generated_tokens": 256,
        }
        tool_result = {
            "text": '<tool_call>{"name":"shell_command","arguments":{"cmd":"pwd"}}</tool_call>',
            "finish_reason": "stop",
            "generated_tokens": 8,
        }

        with (
            mock.patch.object(local_api_server, "MAX_TOOL_TURN_TOKENS", 256),
            mock.patch.object(
                server,
                "_generate_locked",
                side_effect=[
                    (["I will keep working."], text_result),
                    (['<tool_call>{"name":"shell_command"}</tool_call>'], tool_result),
                ],
            ) as generate_locked_mock,
        ):
            result, output_items = server._generate_response_locked(
                [{"role": "user", "content": "Create app."}],
                1000,
                tools=tools,
            )

        self.assertIs(result, tool_result)
        self.assertEqual(output_items[0]["type"], "function_call")
        self.assertEqual(generate_locked_mock.call_args_list[0].args[1], 256)
        self.assertEqual(generate_locked_mock.call_args_list[1].args[1], 256)

    def test_generate_response_returns_repeated_tool_call_without_loop_guard(self):
        server = self._make_server()
        tools = [{"type": "function", "function": {"name": "shell_command"}}]
        messages = [
            {"role": "user", "content": "Verify."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    local_api_server._make_internal_tool_call(
                        "shell_command",
                        {"command": "npm list vite", "workdir": "/repo"},
                    )
                ],
            },
            {"role": "tool", "content": "vite@8.0.10", "tool_call_id": "call_1"},
        ]
        repeated = {
            "text": '<tool_call>{"name":"shell_command","arguments":{"command":"npm list vite","workdir":"/repo"}}</tool_call>',
            "finish_reason": "stop",
            "generated_tokens": 12,
        }

        with mock.patch.object(
            server,
            "_generate_locked",
            return_value=(["repeat"], repeated),
        ) as generate_locked_mock:
            result, output_items = server._generate_response_locked(
                messages,
                256,
                tools=tools,
            )

        self.assertIs(result, repeated)
        self.assertEqual(generate_locked_mock.call_count, 1)
        self.assertEqual(output_items[0]["type"], "function_call")
        self.assertNotIn("protocol_final_message", result)

    def test_generate_response_returns_no_tool_stop_without_forced_tool(self):
        server = self._make_server()
        tools = [{"type": "function", "function": {"name": "shell_command"}}]
        no_tool = {
            "text": "```ts\nconst x = 1\n```",
            "finish_reason": "stop",
            "generated_tokens": 1,
        }

        with (
            mock.patch.object(local_api_server, "RESPONSES_ACTION_FOLLOWUP_LIMIT", 0),
            mock.patch.object(server, "_generate_locked", return_value=(["code"], no_tool)),
        ):
            result, output_items = server._generate_response_locked(
                [{"role": "user", "content": "Create app."}],
                64,
                tools=tools,
            )

        self.assertIs(result, no_tool)
        self.assertEqual(output_items[0]["type"], "message")
        self.assertEqual(output_items[0]["content"][0]["text"], no_tool["text"])

    def test_generate_returns_chat_no_tool_stop_as_final(self):
        server = self._make_server()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "shell_command",
                    "parameters": {"type": "object"},
                },
            }
        ]
        planning_result = {
            "text": "Now let me explore the project structure and start building the Snake game components.",
            "finish_reason": "stop",
            "generated_tokens": 8,
        }
        with (
            mock.patch.object(
                server,
                "_generate_locked",
                return_value=(["Now let me explore."], planning_result),
            ) as generate_locked_mock,
            mock.patch.object(server, "_record_generation_metrics"),
        ):
            result = server.generate(
                messages=[{"role": "user", "content": "Create Snake."}],
                max_tokens=128,
                sampling=local_api_server.SamplingParams(temperature=0.0),
                tools=tools,
            )

        self.assertIs(result, planning_result)
        self.assertEqual(generate_locked_mock.call_count, 1)

    def test_generate_response_auto_continues_after_empty_stop(self):
        server = self._make_server()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "edit",
                    "parameters": {"type": "object"},
                },
            }
        ]
        empty_result = {
            "text": "",
            "finish_reason": "stop",
            "prompt_tokens": 12000,
            "prefill_seconds": 1.0,
            "prompt_tps": 2000.0,
            "reused_prefix_tokens": 0,
            "decode_seconds": 0.01,
            "generation_tps": 100.0,
            "generated_tokens": 1,
            "speculative_steps": 0,
            "proposed_tokens": 0,
            "accepted_tokens": 0,
            "avg_acceptance_length": 0.0,
            "avg_acceptance_ratio": 0.0,
            "acceptance_lengths": [],
            "acceptance_ratios": [],
            "block_size_history": [],
            "adaptive_block_size": True,
            "prefix_cache_source": "global",
            "peak_memory_gb": 1.0,
            "elapsed": 1.01,
            "prompt_cache_state": None,
        }
        tool_result = {
            "text": '<function_call>{"name":"edit","arguments":{"filePath":"src/App.jsx","oldString":"todo","newString":"done"}}</function_call>',
            "finish_reason": "stop",
            "prompt_tokens": 12100,
            "prefill_seconds": 1.0,
            "prompt_tps": 2000.0,
            "reused_prefix_tokens": 0,
            "decode_seconds": 0.2,
            "generation_tps": 50.0,
            "generated_tokens": 10,
            "speculative_steps": 1,
            "proposed_tokens": 10,
            "accepted_tokens": 8,
            "avg_acceptance_length": 8.0,
            "avg_acceptance_ratio": 0.8,
            "acceptance_lengths": [8],
            "acceptance_ratios": [0.8],
            "block_size_history": [8],
            "adaptive_block_size": True,
            "prefix_cache_source": "global",
            "peak_memory_gb": 1.0,
            "elapsed": 1.2,
            "prompt_cache_state": None,
        }
        with (
            mock.patch.object(
                server,
                "_generate_locked",
                side_effect=[(None, empty_result), (None, tool_result)],
            ) as generate_locked,
        ):
            result, output_items = server.generate_response(
                [{"role": "user", "content": "create app"}],
                128,
                tools=tools,
            )

        self.assertIs(result, tool_result)
        self.assertEqual(output_items[0]["type"], "function_call")
        self.assertEqual(generate_locked.call_count, 2)
        second_messages = generate_locked.call_args_list[1].args[0]
        self.assertEqual(len(second_messages), 2)
        self.assertEqual(second_messages[-1]["content"], local_api_server.PROTOCOL_TOOL_RETRY_PROMPT)
        self.assertEqual(second_messages[-1]["role"], "user")
        self.assertEqual(result["text"], tool_result["text"])
        self.assertEqual(output_items[0]["type"], "function_call")
        self.assertEqual(output_items[0]["name"], "edit")

    def test_stream_response_events_do_not_leak_reasoning_text(self):
        server = self._make_server(FakeReasoningResponsesServer)
        events = "".join(
            server.stream_response_events(
                messages=[{"role": "user", "content": "Inspect the repository."}],
                max_tokens=64,
                temperature=0.0,
                request_messages=[{"role": "user", "content": "Inspect the repository."}],
            )
        )

        # Visible answer still shows up in output_text.delta
        self.assertIn('"delta": "Visible answer."', events)
        # Raw <think>…</think> markers must not leak to the wire
        self.assertNotIn("</think>", events)
        # Reasoning text MUST NOT appear in output_text.* events (only in
        # the dedicated reasoning_summary_text.delta events, which Codex
        # handles separately from the user-visible message content).
        reasoning_phrase = "The user wants me to inspect the repo."
        for line in events.splitlines():
            if not line.startswith("data:"):
                continue
            if reasoning_phrase not in line:
                continue
            # Any line containing the reasoning text must be a reasoning event.
            self.assertTrue(
                "reasoning" in line,
                f"reasoning text leaked into a non-reasoning event: {line[:160]}",
            )

    def test_stream_response_events_mark_truncated_tool_call_as_incomplete(self):
        server = self._make_server(FakeTruncatedToolCallServer)
        events = "".join(
            server.stream_response_events(
                messages=[{"role": "user", "content": "Create index.html"}],
                max_tokens=128,
                temperature=0.0,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                request_messages=[{"role": "user", "content": "Create index.html"}],
            )
        )

        # Codex 0.122 treats `response.incomplete` as a terminal error, so we
        # only emit `response.completed` with `incomplete_details` inline.
        self.assertNotIn("event: response.incomplete", events)
        self.assertIn("event: response.completed", events)
        self.assertIn('"reason": "truncated_tool_call"', events)
        # Note: `status` in the final payload is intentionally "completed" on
        # the wire; the real state is surfaced via incomplete_details.

    def test_responses_endpoint_marks_truncated_tool_call_as_incomplete(self):
        server = self._make_server(FakeTruncatedToolCallServer)
        endpoint = self._get_endpoint(server, "/v1/responses")

        class FakeResponsesRequest(SimpleNamespace):
            def model_dump(self, mode="json"):
                return {
                    "model": self.model,
                    "input": self.input,
                    "instructions": self.instructions,
                    "max_output_tokens": self.max_output_tokens,
                    "temperature": self.temperature,
                    "stream": self.stream,
                    "tools": self.tools,
                    "tool_choice": self.tool_choice,
                    "parallel_tool_calls": self.parallel_tool_calls,
                    "prompt_cache_key": self.prompt_cache_key,
                    "store": self.store,
                    "service_tier": self.service_tier,
                    "previous_response_id": self.previous_response_id,
                    "include": self.include,
                    "reasoning": self.reasoning,
                    "keep_alive": self.keep_alive,
                }

        payload = endpoint(
            FakeResponsesRequest(
                model="local-test-model",
                input="Create index.html",
                instructions=None,
                max_output_tokens=128,
                temperature=0.0,
                top_p=None,
                top_k=None,
                min_p=None,
                presence_penalty=None,
                repetition_penalty=None,
                frequency_penalty=None,
                stream=False,
                tools=None,
                tool_choice=None,
                parallel_tool_calls=None,
                prompt_cache_key=None,
                store=None,
                service_tier=None,
                previous_response_id=None,
                include=None,
                reasoning=None,
                text=None,
                client_metadata=None,
                metadata=None,
                truncation=None,
                keep_alive=None,
            )
        )

        self.assertEqual(payload["status"], "incomplete")
        self.assertEqual(payload["incomplete_details"], {"reason": "truncated_tool_call"})
        self.assertEqual(payload["output_text"], "I'll create the file now.")

    def test_stream_chat_completions_emits_openai_chunks(self):
        server = self._make_server(FakeChatStreamingServer)
        events = "".join(
            server.stream_chat_completions(
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=32,
                temperature=0.0,
            )
        )

        self.assertIn('"object": "chat.completion.chunk"', events)
        self.assertIn('"role": "assistant"', events)
        self.assertIn('"content": "Hello"', events)
        self.assertIn('"finish_reason": "stop"', events)
        self.assertTrue(events.rstrip().endswith("data: [DONE]"))

    def test_chat_completions_non_stream_emits_tool_calls(self):
        server = self._make_server()
        endpoint = self._get_endpoint(server, "/v1/chat/completions")
        result = FakeStreamingServer._result()
        with mock.patch.object(server, "generate", return_value=result):
            payload = endpoint(
                local_api_server.OpenAIChatRequest(
                    model=server.model_name,
                    messages=[local_api_server.OpenAIMessage(role="user", content="write file")],
                    tools=[{"type": "function", "function": {"name": "write_file"}}],
                )
            )

        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "write_file")
        self.assertIn("index.html", choice["message"]["tool_calls"][0]["function"]["arguments"])

    def test_stream_chat_completions_emits_tool_call_deltas(self):
        server = self._make_server(FakeStreamingServer)
        events = "".join(
            server.stream_chat_completions(
                messages=[{"role": "user", "content": "write file"}],
                max_tokens=64,
                temperature=0.0,
                tools=[{"type": "function", "function": {"name": "write_file"}}],
            )
        )

        self.assertIn('"tool_calls"', events)
        self.assertIn('"name": "write_file"', events)
        self.assertIn('"finish_reason": "tool_calls"', events)
        self.assertTrue(events.rstrip().endswith("data: [DONE]"))

    def test_anthropic_messages_non_stream_emits_tool_use(self):
        server = self._make_server()
        endpoint = self._get_endpoint(server, "/v1/messages")
        result = FakeStreamingServer._result()
        with mock.patch.object(server, "generate", return_value=result):
            payload = endpoint(
                local_api_server.AnthropicRequest(
                    model=server.model_name,
                    max_tokens=64,
                    messages=[
                        local_api_server.AnthropicMessage(role="user", content="write file")
                    ],
                    tools=[{"name": "write_file", "input_schema": {"type": "object"}}],
                )
            )

        self.assertEqual(payload["stop_reason"], "tool_use")
        self.assertEqual(payload["content"][0]["type"], "tool_use")
        self.assertEqual(payload["content"][0]["name"], "write_file")
        self.assertEqual(payload["content"][0]["input"]["path"], "index.html")

    def test_stream_anthropic_events_emits_tool_use_blocks(self):
        server = self._make_server(FakeStreamingServer)
        events = "".join(
            server.stream_anthropic_events(
                messages=[{"role": "user", "content": "write file"}],
                max_tokens=64,
                temperature=0.0,
                tools=[{"name": "write_file", "input_schema": {"type": "object"}}],
            )
        )

        self.assertIn('"type": "tool_use"', events)
        self.assertIn('"name": "write_file"', events)
        self.assertIn('"type": "input_json_delta"', events)
        self.assertIn('"stop_reason": "tool_use"', events)

    def test_stream_chat_error_closes_with_done(self):
        server = self._make_server(FakeStreamErrorServer)
        events = "".join(
            server.stream_chat_completions(
                messages=[{"role": "user", "content": "fail"}],
                max_tokens=16,
                temperature=0.0,
            )
        )

        self.assertIn('"message": "boom"', events)
        self.assertTrue(events.rstrip().endswith("data: [DONE]"))

    def test_stream_chat_no_result_closes_with_done(self):
        server = self._make_server(FakeStreamNoResultServer)
        events = "".join(
            server.stream_chat_completions(
                messages=[{"role": "user", "content": "empty"}],
                max_tokens=16,
                temperature=0.0,
            )
        )

        self.assertIn("Generation completed without a final result", events)
        self.assertTrue(events.rstrip().endswith("data: [DONE]"))

    def test_stream_responses_error_closes_with_failed_and_done(self):
        server = self._make_server(FakeStreamErrorServer)
        events = "".join(
            server.stream_response_events(
                messages=[{"role": "user", "content": "fail"}],
                max_tokens=16,
                temperature=0.0,
            )
        )

        self.assertIn("event: response.failed", events)
        self.assertIn('"message": "boom"', events)
        self.assertTrue(events.rstrip().endswith("data: [DONE]"))

    def test_stream_responses_no_result_closes_with_failed_and_done(self):
        server = self._make_server(FakeStreamNoResultServer)
        events = "".join(
            server.stream_response_events(
                messages=[{"role": "user", "content": "empty"}],
                max_tokens=16,
                temperature=0.0,
            )
        )

        self.assertIn("event: response.failed", events)
        self.assertIn("Generation completed without a final result", events)
        self.assertTrue(events.rstrip().endswith("data: [DONE]"))

    def test_stream_anthropic_error_emits_error_event(self):
        server = self._make_server(FakeStreamErrorServer)
        events = "".join(
            server.stream_anthropic_events(
                messages=[{"role": "user", "content": "fail"}],
                max_tokens=16,
                temperature=0.0,
            )
        )

        self.assertIn("event: error", events)
        self.assertIn('"message": "boom"', events)

    def test_stream_anthropic_no_result_emits_error_event(self):
        server = self._make_server(FakeStreamNoResultServer)
        events = "".join(
            server.stream_anthropic_events(
                messages=[{"role": "user", "content": "empty"}],
                max_tokens=16,
                temperature=0.0,
            )
        )

        self.assertIn("event: error", events)
        self.assertIn("Generation completed without a final result", events)

    def test_streaming_generation_returns_no_tool_text_as_final(self):
        server = self._make_server()

        def chunk(text: str, tokens: int) -> SimpleNamespace:
            return SimpleNamespace(
                text=text,
                finish_reason="stop",
                prefill_seconds=0.01,
                prompt_tps=10.0,
                reused_prefix_tokens=0,
                decode_seconds=0.02,
                generation_tps=20.0,
                generation_tokens=tokens,
                speculative_steps=1,
                proposed_tokens=tokens,
                accepted_tokens=tokens,
                avg_acceptance_length=float(tokens),
                avg_acceptance_ratio=1.0,
                acceptance_lengths=[tokens],
                acceptance_ratios=[1.0],
                block_size_history=[tokens],
                adaptive_block_size=False,
                prefill_hidden_bytes=0,
                prefill_target_cache_bytes=0,
                prefill_logits_bytes=0,
                prefill_working_set_bytes=0,
                prompt_cache_state_bytes=0,
                peak_memory=1.0,
                prefill_state=None,
            )

        queue = Queue()
        tools = [{"type": "function", "function": {"name": "shell_command"}}]
        with (
            mock.patch.object(server, "_acquire_generation_turn", return_value=1),
            mock.patch.object(server, "_release_generation_turn"),
            mock.patch.object(server, "finish_request"),
            mock.patch.object(server, "_record_generation_metrics"),
            mock.patch.object(
                server,
                "_stream_generate_locked",
                return_value=(
                    iter(
                        [
                            chunk(
                                "Let me rewrite the Snake game and add tests.",
                                9,
                            )
                        ]
                    ),
                    10,
                    1.0,
                    None,
                    (),
                    "none",
                ),
            ) as stream_mock,
        ):
            server._generation_worker(
                queue,
                messages=[{"role": "user", "content": "create react vite game"}],
                requested_max_tokens=64,
                sampling=local_api_server.SamplingParams(temperature=0.0),
                tools=tools,
            )

        events = []
        while not queue.empty():
            events.append(queue.get())

        self.assertEqual(stream_mock.call_count, 1)
        self.assertIn(
            ("text", "Let me rewrite the Snake game and add tests."),
            events,
        )
        result = next(payload for kind, payload in events if kind == "result")
        self.assertEqual(result["text"], "Let me rewrite the Snake game and add tests.")

    def test_incremental_visible_text_stream_matches_full_visible_output(self):
        stream = local_api_server._IncrementalVisibleTextStream(strip_edges=False)
        chunks = [
            "  <thi",
            "nk>hidden</think>He",
            "llo<tool_",
            'call>{"name":"write_file"',
            '}</tool_call> world<|im_',
            "end|>",
        ]

        emitted = "".join(stream.feed(chunk) for chunk in chunks)
        emitted += stream.feed("", final=True)

        self.assertEqual(
            emitted,
            local_api_server._extract_visible_text("".join(chunks)),
        )

    def test_incremental_visible_text_stream_matches_chat_strip_behavior(self):
        stream = local_api_server._IncrementalVisibleTextStream(strip_edges=True)
        chunks = [
            "  Hello",
            " ",
            "<think>hidden</think>",
            "world  ",
        ]

        emitted = "".join(stream.feed(chunk) for chunk in chunks)
        emitted += stream.feed("", final=True)

        self.assertEqual(
            emitted,
            local_api_server._clean_output_text(local_api_server._extract_visible_text("".join(chunks))),
        )

    def test_parse_keep_alive_accepts_five_minutes(self):
        self.assertEqual(local_api_server._parse_keep_alive("5m"), 300.0)

    def test_finish_request_with_positive_keep_alive_schedules_unload(self):
        server = self._make_server(TrackingServer, keep_alive_seconds=300)
        sentinel_model = object()
        sentinel_draft = object()
        sentinel_tokenizer = object()
        server._model = sentinel_model
        server._draft = sentinel_draft
        server._tokenizer = sentinel_tokenizer

        with (
            mock.patch.object(local_api_server, "Timer", FakeTimer),
            mock.patch.object(local_api_server.time, "time", return_value=123.0),
        ):
            server.finish_request()

        self.assertEqual(server.clear_request_state_calls, 1)
        self.assertEqual(server.reset_loaded_state_calls, 0)
        self.assertEqual(server._last_used_at, 123.0)
        self.assertIsNotNone(server._unload_timer)
        self.assertTrue(server._unload_timer.started)
        self.assertEqual(server._unload_timer.interval, 300)
        self.assertIs(server._model, sentinel_model)

        timer = server._unload_timer
        timer.fire()

        self.assertIsNone(server._model)
        self.assertIsNone(server._draft)
        self.assertIsNone(server._tokenizer)
        self.assertEqual(server.reset_loaded_state_calls, 1)
        self.assertTrue(timer.cancelled)
        self.assertIsNone(server._unload_timer)

    def test_generate_finishes_request_when_generation_raises(self):
        server = self._make_server(TrackingServer, keep_alive_seconds=300)
        server._generate_locked = mock.Mock(side_effect=RuntimeError("boom"))

        with self.assertRaises(RuntimeError):
            server.generate(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=16,
                temperature=0.0,
            )

        self.assertEqual(server.clear_request_state_calls, 1)
        self.assertIsNone(server._active_generation_ticket)

    def test_ensure_loaded_reloads_model_after_idle_unload(self):
        server = self._make_server(TrackingServer, keep_alive_seconds=300)

        with (
            mock.patch.object(local_api_server, "load", return_value=("model-1", "tokenizer-1")) as load_mock,
            mock.patch.object(local_api_server, "load_draft", return_value="draft-1") as load_draft_mock,
        ):
            server.ensure_loaded()

        self.assertEqual(server._model, "model-1")
        self.assertEqual(server._tokenizer, "tokenizer-1")
        self.assertEqual(server._draft, "draft-1")
        load_mock.assert_called_once_with("model")
        load_draft_mock.assert_called_once_with(
            "draft",
            sliding_window_size=256,
            turboquant_bits=None,
            rotating_keep_tokens=0,
        )

        server.unload()
        self.assertEqual(server.reset_loaded_state_calls, 1)

        with (
            mock.patch.object(local_api_server, "load", return_value=("model-2", "tokenizer-2")) as load_mock,
            mock.patch.object(local_api_server, "load_draft", return_value="draft-2") as load_draft_mock,
        ):
            server.ensure_loaded()

        self.assertEqual(server._model, "model-2")
        self.assertEqual(server._tokenizer, "tokenizer-2")
        self.assertEqual(server._draft, "draft-2")
        load_mock.assert_called_once_with("model")
        load_draft_mock.assert_called_once_with(
            "draft",
            sliding_window_size=256,
            turboquant_bits=None,
            rotating_keep_tokens=0,
        )

    def test_health_reports_loaded_state_without_preload(self):
        server = self._make_server(TrackingServer, keep_alive_seconds=300)
        health = self._get_health_endpoint(server)

        with (
            mock.patch.object(local_api_server.mx, "get_active_memory", return_value=0),
            mock.patch.object(local_api_server.mx, "get_cache_memory", return_value=0),
            mock.patch.object(local_api_server.mx, "get_peak_memory", return_value=0),
        ):
            before = health()
            server._model = object()
            server._draft = object()
            server._tokenizer = object()
            loaded = health()
            server.unload()
            after = health()

        self.assertFalse(before["loaded"])
        self.assertTrue(loaded["loaded"])
        self.assertFalse(after["loaded"])
        self.assertEqual(before["keep_alive_seconds"], 300)
        self.assertEqual(
            before["stream_result_timeout_seconds"],
            local_api_server.STREAM_RESULT_TIMEOUT_SECONDS,
        )
        self.assertEqual(before["response_history_limit"], server.response_history_limit)
        self.assertEqual(before["response_history_entries"], 0)
        self.assertEqual(before["active_generation_requests"], 0)
        self.assertEqual(before["queued_generation_requests"], 0)
        self.assertIsNone(before["draft_turboquant_bits"])
        self.assertEqual(before["response_prefix_cache_bytes"], 0)
        self.assertEqual(before["global_prefix_cache_bytes"], 0)

    def test_health_reports_prefix_cache_bytes(self):
        server = self._make_server()
        health = self._get_health_endpoint(server)
        state = local_api_server.PromptPrefillState(
            prompt_tokens=(1, 2),
            target_cache=[
                SimpleNamespace(
                    keys=mx.zeros((1, 1, 2, 2), dtype=mx.float16),
                    values=mx.zeros((1, 1, 2, 2), dtype=mx.float16),
                )
            ],
            hidden=mx.zeros((1, 2, 3), dtype=mx.float16),
            last_logits=mx.zeros((1, 1, 4), dtype=mx.float16),
        )
        expected_bytes = local_api_server.estimate_memory_bytes(state)
        server.remember_response(
            response_id="resp_1",
            previous_response_id=None,
            request_messages=[{"role": "user", "content": "hello"}],
            tools=[],
            output_items=[],
            prompt_cache_state=state,
        )
        server._global_prefix_states["stable"] = state
        server._global_prefix_order.append("stable")
        server._global_prefix_state_bytes["stable"] = expected_bytes
        server._global_prefix_cache_bytes = expected_bytes

        with (
            mock.patch.object(local_api_server.mx, "get_active_memory", return_value=0),
            mock.patch.object(local_api_server.mx, "get_cache_memory", return_value=0),
            mock.patch.object(local_api_server.mx, "get_peak_memory", return_value=0),
        ):
            payload = health()

        self.assertEqual(payload["response_prefix_cache_bytes"], expected_bytes)
        self.assertEqual(payload["global_prefix_cache_bytes"], expected_bytes)

    def test_compatibility_endpoints_expose_model_metadata(self):
        server = self._make_server()
        list_models = self._get_endpoint(server, "/v1/models")
        get_model = self._get_endpoint(server, "/v1/models/{model_id}")
        lm_studio_models = self._get_endpoint(server, "/api/v1/models")
        ollama_tags = self._get_endpoint(server, "/api/tags")
        llamacpp_props = self._get_endpoint(server, "/v1/props")
        version = self._get_endpoint(server, "/version")

        import json as _json
        list_response = list_models()
        detail_response = get_model(server.model_name)
        list_payload = _json.loads(list_response.body.decode("utf-8")) if hasattr(list_response, "body") else list_response
        detail_payload = _json.loads(detail_response.body.decode("utf-8")) if hasattr(detail_response, "body") else detail_response
        # ETag headers are what Codex uses to skip repeat model fetches
        if hasattr(list_response, "headers"):
            self.assertIn("X-Models-Etag", list_response.headers)
        lm_studio_payload = lm_studio_models()
        ollama_payload = ollama_tags()
        props_payload = llamacpp_props()
        version_payload = version()

        self.assertEqual(list_payload["data"][0]["id"], server.model_name)
        self.assertEqual(detail_payload["max_model_len"], server.context_window)
        self.assertEqual(detail_payload["context_length"], server.context_window)
        self.assertEqual(lm_studio_payload["models"][0]["id"], server.model_name)
        self.assertEqual(
            lm_studio_payload["models"][0]["loaded_instances"][0]["config"]["context_length"],
            server.context_window,
        )
        self.assertEqual(ollama_payload["models"][0]["name"], server.model_name)
        self.assertEqual(
            ollama_payload["models"][0]["details"]["context_length"],
            server.context_window,
        )
        self.assertEqual(
            props_payload["default_generation_settings"]["n_ctx"],
            server.context_window,
        )
        self.assertEqual(version_payload["version"], "local-dflash/0.2.0")

    def test_remember_response_keeps_only_recent_prefix_cache_states(self):
        server = self._make_server()
        server.prefix_cache_state_limit = 1
        first_state = local_api_server.PromptPrefillState(
            prompt_tokens=(1, 2),
            target_cache=["cache-1"],
            hidden="hidden-1",
            last_logits="logits-1",
        )
        second_state = local_api_server.PromptPrefillState(
            prompt_tokens=(1, 2, 3),
            target_cache=["cache-2"],
            hidden="hidden-2",
            last_logits="logits-2",
        )

        server.remember_response(
            response_id="resp_1",
            previous_response_id=None,
            request_messages=[{"role": "user", "content": "one"}],
            tools=[],
            output_items=[],
            prompt_cache_state=first_state,
        )
        server.remember_response(
            response_id="resp_2",
            previous_response_id="resp_1",
            request_messages=[{"role": "user", "content": "two"}],
            tools=[],
            output_items=[],
            prompt_cache_state=second_state,
        )

        self.assertIsNone(server._response_states["resp_1"]["prompt_cache_state"])
        self.assertEqual(server._response_states["resp_2"]["prompt_cache_state"], second_state)
        self.assertEqual(list(server._prefix_state_order), ["resp_2"])

    def test_remember_response_drops_prefix_state_over_byte_limit(self):
        server = self._make_server()
        server.prefix_cache_state_limit = 1
        server.prefix_cache_state_byte_limit = 1
        state = local_api_server.PromptPrefillState(
            prompt_tokens=(1, 2),
            target_cache=[],
            hidden=mx.zeros((1, 2, 2), dtype=mx.float16),
            last_logits=None,
        )

        server.remember_response(
            response_id="resp_1",
            previous_response_id=None,
            request_messages=[{"role": "user", "content": "one"}],
            tools=[],
            output_items=[],
            prompt_cache_state=state,
        )

        self.assertIsNone(server._response_states["resp_1"]["prompt_cache_state"])
        self.assertEqual(list(server._prefix_state_order), [])
        self.assertEqual(server._response_prefix_cache_bytes, 0)

    def test_remember_response_discards_history_when_limit_is_zero(self):
        server = self._make_server()
        server.response_history_limit = 0

        server.remember_response(
            response_id="resp_1",
            previous_response_id=None,
            request_messages=[{"role": "user", "content": "one"}],
            tools=[],
            output_items=[],
        )

        self.assertEqual(server._response_states, {})
        self.assertEqual(list(server._response_order), [])

    def test_remember_response_prunes_parent_but_keeps_latest_tools_and_continuation(self):
        server = self._make_server()
        server.response_history_limit = 1
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "parameters": {"type": "object"},
                },
            }
        ]

        server.remember_response(
            response_id="resp_1",
            previous_response_id=None,
            request_messages=[{"role": "user", "content": "one"}],
            tools=tools,
            output_items=[local_api_server._make_message_item("first")],
        )
        server.remember_response(
            response_id="resp_2",
            previous_response_id="resp_1",
            request_messages=[{"role": "user", "content": "two"}],
            tools=tools,
            output_items=[local_api_server._make_message_item("second")],
        )

        self.assertNotIn("resp_1", server._response_states)
        self.assertIsNone(server._response_states["resp_2"]["previous_response_id"])
        self.assertEqual(server._response_states["resp_2"]["tools"], tools)

        merged_messages, merged_tools = server.resolve_responses_context(
            request_messages=[{"role": "user", "content": "three"}],
            request_tools=[],
            previous_response_id="resp_2",
        )

        self.assertEqual(
            merged_messages,
            [
                {"role": "user", "content": "two"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "three"},
            ],
        )
        self.assertEqual(merged_tools, tools)

    def test_generate_response_with_tool_call_returns_tool_call(self):
        server = self._make_server()
        tools = [{"type": "function", "function": {"name": "edit"}}]
        tool_result = {
            "text": '<function_call>{"name":"edit","arguments":{"filePath":"src/App.tsx","oldString":"a","newString":"b"}}</function_call>',
            "finish_reason": "stop",
            "generated_tokens": 8,
        }

        with (
            mock.patch.object(server, "_generate_locked", return_value=(None, tool_result)) as generate_locked,
            mock.patch.object(server, "_record_generation_metrics"),
        ):
            result, output_items = server.generate_response(
                [{"role": "user", "content": "edit file"}],
                64,
                tools=tools,
            )

        self.assertIs(result, tool_result)
        self.assertEqual(generate_locked.call_count, 1)
        self.assertEqual(output_items[0]["type"], "function_call")
        self.assertEqual(output_items[0]["name"], "edit")

    def test_stable_prefix_tokens_are_disabled_when_global_cache_limit_is_zero(self):
        server = self._make_server()
        server.global_prefix_cache_limit = 0

        with mock.patch.object(server, "build_prompt") as build_prompt_mock:
            tokens = server._stable_prefix_tokens_locked(
                messages=[{"role": "system", "content": "You are a coding agent."}],
                tools=[],
            )

        self.assertEqual(tokens, ())
        self.assertEqual(server._stable_prefix_tokens_by_key, {})
        build_prompt_mock.assert_not_called()

    def test_generation_requests_wait_in_fifo_order(self):
        server = self._make_server()
        first_ticket = server._acquire_generation_turn()
        self.assertEqual(server._active_generation_ticket, first_ticket)
        self.assertEqual(len(server._queued_generation_tickets), 0)

        second_ticket_holder = {}
        second_acquired = threading.Event()

        def acquire_second() -> None:
            second_ticket_holder["ticket"] = server._acquire_generation_turn()
            second_acquired.set()

        worker = threading.Thread(target=acquire_second, daemon=True)
        worker.start()

        self.assertFalse(second_acquired.wait(0.05))
        self.assertEqual(len(server._queued_generation_tickets), 1)
        self.assertEqual(server._queued_generation_tickets[0], first_ticket + 1)

        server._release_generation_turn(first_ticket)

        self.assertTrue(second_acquired.wait(1.0))
        self.assertEqual(server._active_generation_ticket, second_ticket_holder["ticket"])
        self.assertEqual(len(server._queued_generation_tickets), 0)

        server._release_generation_turn(second_ticket_holder["ticket"])
        worker.join(timeout=1.0)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(server._active_generation_ticket)

    def test_generate_reuses_previous_response_prefix_cache(self):
        server = self._make_server(TrackingServer)
        server.global_prefix_cache_limit = 0
        prior_state = local_api_server.PromptPrefillState(
            prompt_tokens=(10, 20, 30),
            target_cache=["cached-prefix"],
            hidden="hidden-prefix",
            last_logits="logits-prefix",
        )
        next_state = local_api_server.PromptPrefillState(
            prompt_tokens=(10, 20, 30, 40),
            target_cache=["cached-next"],
            hidden="hidden-next",
            last_logits="logits-next",
        )
        server.remember_response(
            response_id="resp_1",
            previous_response_id=None,
            request_messages=[{"role": "user", "content": "Inspect"}],
            tools=[],
            output_items=[],
            prompt_cache_state=prior_state,
        )

        class FakeTokenizer:
            def apply_chat_template(self, messages, enable_thinking=True, **kwargs):
                return "prompt"

        observed: dict[str, object] = {}

        def fake_stream_generate(model, draft, tokenizer, prompt, **kwargs):
            observed["prompt"] = prompt
            observed["prefix_state"] = kwargs.get("prefix_state")
            observed["capture_prefill_state"] = kwargs.get("capture_prefill_state")
            yield SimpleNamespace(
                text="ok",
                finish_reason="stop",
                prefill_seconds=0.1,
                prompt_tps=10.0,
                reused_prefix_tokens=3,
                decode_seconds=0.2,
                generation_tps=9.0,
                generation_tokens=2,
                speculative_steps=1,
                proposed_tokens=2,
                accepted_tokens=2,
                avg_acceptance_length=2.0,
                avg_acceptance_ratio=1.0,
                acceptance_lengths=(2,),
                acceptance_ratios=(1.0,),
                block_size_history=(2,),
                adaptive_block_size=False,
                prefill_hidden_bytes=11,
                prefill_target_cache_bytes=22,
                prefill_logits_bytes=33,
                prefill_working_set_bytes=66,
                prompt_cache_state_bytes=77,
                peak_memory=1.0,
                prefill_state=next_state,
            )

        with (
            mock.patch.object(local_api_server, "load", return_value=("model-1", FakeTokenizer())),
            mock.patch.object(local_api_server, "load_draft", return_value="draft-1"),
            mock.patch.object(local_api_server, "tokenize_prompt", return_value=local_api_server.mx.array([10, 20, 30])),
            mock.patch.object(local_api_server, "stream_generate", side_effect=fake_stream_generate),
        ):
            result = server.generate(
                messages=[{"role": "user", "content": "Continue"}],
                max_tokens=32,
                temperature=0.0,
                previous_response_id="resp_1",
                capture_prompt_cache_state=True,
            )

        self.assertEqual(observed["prompt"], [10, 20, 30])
        self.assertEqual(observed["prefix_state"], prior_state)
        self.assertTrue(observed["capture_prefill_state"])
        self.assertEqual(result["prompt_cache_state"], next_state)

    def test_select_prefix_state_prefers_longer_global_match(self):
        server = self._make_server()
        response_state = local_api_server.PromptPrefillState(
            prompt_tokens=(1, 2),
            target_cache=["resp-cache"],
            hidden="resp-hidden",
            last_logits="resp-logits",
        )
        global_state = local_api_server.PromptPrefillState(
            prompt_tokens=(1, 2, 3),
            target_cache=["global-cache"],
            hidden="global-hidden",
            last_logits="global-logits",
        )
        server._response_states["resp_1"] = {"prompt_cache_state": response_state}
        server._global_prefix_states["stable"] = global_state

        selected, source = server._select_prefix_state_locked([1, 2, 3, 4], "resp_1", "stable")

        self.assertEqual(source, "global")
        self.assertEqual(selected, global_state)

    def test_prune_global_prefix_states_also_prunes_stable_prefix_tokens(self):
        server = self._make_server()
        server.global_prefix_cache_limit = 1
        server._global_prefix_states = {
            "stale": local_api_server.PromptPrefillState(
                prompt_tokens=(1, 2),
                target_cache=["stale-cache"],
                hidden="stale-hidden",
                last_logits="stale-logits",
            ),
            "fresh": local_api_server.PromptPrefillState(
                prompt_tokens=(1, 2, 3),
                target_cache=["fresh-cache"],
                hidden="fresh-hidden",
                last_logits="fresh-logits",
            ),
        }
        server._global_prefix_order.extend(["stale", "fresh"])
        server._stable_prefix_tokens_by_key = {
            "stale": (1, 2),
            "fresh": (1, 2, 3),
        }

        server._prune_global_prefix_states_locked()

        self.assertNotIn("stale", server._global_prefix_states)
        self.assertNotIn("stale", server._stable_prefix_tokens_by_key)
        self.assertIn("fresh", server._global_prefix_states)
        self.assertIn("fresh", server._stable_prefix_tokens_by_key)


if __name__ == "__main__":
    unittest.main()
