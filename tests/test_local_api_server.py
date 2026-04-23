from __future__ import annotations

import importlib.util
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

    def test_massage_responses_continuation_messages_adds_explicit_continue_prompt(self):
        messages = [
            {"role": "user", "content": "Build the app."},
            {"role": "assistant", "content": "I'll build it now."},
            {"role": "assistant", "content": "I'll create the file next."},
        ]

        massaged = local_api_server._massage_responses_continuation_messages(messages)

        self.assertEqual(len(massaged), 3)
        self.assertEqual(massaged[0], messages[0])
        self.assertEqual(massaged[1], messages[-1])
        self.assertEqual(massaged[2]["role"], "user")
        self.assertEqual(massaged[2]["content"], local_api_server.RESPONSES_CONTINUE_PROMPT)

    def test_massage_responses_continuation_messages_leaves_user_turn_unchanged(self):
        messages = [
            {"role": "user", "content": "Build the app."},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Now add tests."},
        ]

        self.assertEqual(
            local_api_server._massage_responses_continuation_messages(messages),
            messages,
        )

    def test_massage_responses_tool_result_messages_adds_anti_repeat_prompt(self):
        messages = [
            {"role": "user", "content": "Inspect the repository."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    local_api_server._make_internal_tool_call(
                        "search_files",
                        {"path": "."},
                        call_id="call_1",
                    )
                ],
            },
            {"role": "tool", "content": '[{"path":"README.md"}]', "tool_call_id": "call_1"},
        ]

        massaged = local_api_server._massage_responses_tool_result_messages(messages)

        self.assertEqual(massaged[:-1], messages)
        self.assertEqual(massaged[-1]["role"], "user")
        self.assertEqual(massaged[-1]["content"], local_api_server.RESPONSES_TOOL_RESULT_PROMPT)

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

    def test_generate_response_auto_continues_after_action_only_stop(self):
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
        tool_result = {
            "text": '<function_call>{"name":"edit","arguments":{"filePath":"tests.js","oldString":"../index.js","newString":"./index.js"}}</function_call>',
            "finish_reason": "stop",
            "prompt_tokens": 24,
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
                side_effect=[
                    (["The test file has a module resolution issue."], planning_result),
                    (['<function_call>{"name":"edit"}</function_call>'], tool_result),
                ],
            ) as generate_locked_mock,
            mock.patch.object(server, "_judge_response_needs_followup", return_value=True),
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

        self.assertEqual(generate_locked_mock.call_count, 2)
        first_call = generate_locked_mock.call_args_list[0]
        second_call = generate_locked_mock.call_args_list[1]
        self.assertEqual(first_call.kwargs["previous_response_id"], "resp_1")
        self.assertIsNone(second_call.kwargs["previous_response_id"])
        self.assertEqual(first_call.args[1], 128)
        self.assertEqual(second_call.args[1], 120)
        second_messages = second_call.args[0]
        self.assertEqual(second_messages[-2]["role"], "assistant")
        self.assertEqual(
            second_messages[-2]["content"],
            "The test file has a module resolution issue. Let me fix the import path and run the tests again.",
        )
        self.assertEqual(second_messages[-1]["role"], "user")
        self.assertEqual(second_messages[-1]["content"], local_api_server.RESPONSES_ACTION_PROMPT)
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
        server._response_states["resp_1"] = {"prompt_cache_state": state}
        server._global_prefix_states["stable"] = state

        with (
            mock.patch.object(local_api_server.mx, "get_active_memory", return_value=0),
            mock.patch.object(local_api_server.mx, "get_cache_memory", return_value=0),
            mock.patch.object(local_api_server.mx, "get_peak_memory", return_value=0),
        ):
            payload = health()

        expected_bytes = local_api_server.estimate_memory_bytes([state])
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

    def test_response_followup_candidate_treats_custom_tool_call_as_action(self):
        result = {
            "text": "",
            "finish_reason": "stop",
        }
        tools = [{"type": "custom", "name": "apply_patch"}]
        output_items = [
            local_api_server._make_custom_tool_call_item(
                "apply_patch",
                "*** Begin Patch\n*** End Patch\n",
            )
        ]

        self.assertFalse(
            local_api_server._response_is_followup_candidate(result, output_items, tools)
        )

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
