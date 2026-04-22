from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest
from pathlib import Path
from queue import Queue
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "local_api_server.py"
SPEC = importlib.util.spec_from_file_location("local_api_server", MODULE_PATH)
local_api_server = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(local_api_server)


class FakeStreamingServer(local_api_server.LocalModelServer):
    def _generation_worker(
        self,
        queue: Queue,
        messages,
        requested_max_tokens,
        temperature,
        tools=None,
        keep_alive_override=None,
        previous_response_id=None,
        capture_prompt_cache_state=False,
    ) -> None:
        result = {
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
        queue.put(("result", result))
        queue.put(("done", None))


class FakeChatStreamingServer(local_api_server.LocalModelServer):
    def _generation_worker(
        self,
        queue: Queue,
        messages,
        requested_max_tokens,
        temperature,
        tools=None,
        keep_alive_override=None,
        previous_response_id=None,
        capture_prompt_cache_state=False,
    ) -> None:
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
        )

    def _get_health_endpoint(self, server):
        app = local_api_server.create_app(server)
        for route in app.routes:
            if getattr(route, "path", None) == "/health":
                return route.endpoint
        self.fail("Health endpoint not found")

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
        self.assertEqual(merged_messages[4], {"role": "tool", "content": '[{"path":"README.md"}]'})
        self.assertEqual(merged_tools, tools)

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
        load_draft_mock.assert_called_once_with("draft", sliding_window_size=256)

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
        load_draft_mock.assert_called_once_with("draft", sliding_window_size=256)

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
