from __future__ import annotations

import importlib.util
import json
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "opencode_watchdog.py"
SPEC = importlib.util.spec_from_file_location("opencode_watchdog", MODULE_PATH)
opencode_watchdog = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(opencode_watchdog)


class OpenCodeWatchdogTests(unittest.TestCase):
    def test_ensure_json_run_args_adds_required_flags(self):
        args = opencode_watchdog._ensure_json_run_args(["--dir", "/tmp/work", "Build the app"])

        self.assertIn("--format", args)
        self.assertIn("json", args)
        self.assertIn("--print-logs", args)

    def test_ensure_json_run_args_rejects_non_json_format(self):
        with self.assertRaises(SystemExit):
            opencode_watchdog._ensure_json_run_args(["--format", "text", "Build the app"])

    def test_parse_workdir_extracts_dir_flag(self):
        self.assertEqual(
            opencode_watchdog._parse_workdir(["--dir", "/tmp/work", "Build the app"]),
            "/tmp/work",
        )
        self.assertEqual(
            opencode_watchdog._parse_workdir(["--dir=/tmp/work", "Build the app"]),
            "/tmp/work",
        )

    def test_extract_task_returns_last_positional_prompt(self):
        self.assertEqual(
            opencode_watchdog._extract_task(["--dir", "/tmp/work", "Build the app"]),
            "Build the app",
        )

    def test_build_resume_prompt_includes_failure_and_recent_history(self):
        prompt = opencode_watchdog._build_resume_prompt(
            {
                "original_task": "Fix everything until all tests pass.",
                "attempt": 3,
                "last_failure_reason": "stall_timeout",
                "recent_texts": [
                    "I found a failing test.",
                    "Let me fix the import path and rerun the tests.",
                ],
                "recent_tool_signatures": [
                    'bash:{"command":"node tests.js"}',
                    'edit:{"filePath":"tests.js"}',
                ],
            }
        )

        self.assertIn("Fix everything until all tests pass.", prompt)
        self.assertIn("stall_timeout", prompt)
        self.assertIn("Let me fix the import path", prompt)
        self.assertIn('edit:{"filePath":"tests.js"}', prompt)

    def test_should_retry_after_exit_detects_action_only_stop(self):
        state = {
            "last_text": "The wall collision test is failing. Let me fix it.",
            "last_text_at": time.time(),
            "last_tool_at": 0.0,
            "last_step_reason": "stop",
        }

        self.assertEqual(
            opencode_watchdog._should_retry_after_exit(0, state),
            "action_only_stop",
        )

    def test_handle_json_event_detects_repeated_text_loop(self):
        state = {
            "last_progress_at": time.time(),
            "last_progress_event": None,
            "recent_events": [],
            "recent_event_limit": 16,
            "recent_text_limit": 8,
            "recent_tool_limit": 8,
            "recent_texts_deque": deque(maxlen=8),
            "recent_tool_signatures_deque": deque(maxlen=8),
            "recent_texts": [],
            "recent_tool_signatures": [],
            "loop_repeat_threshold": 3,
        }
        event = {
            "type": "text",
            "part": {
                "text": "Let me fix the failing test now.",
            },
        }

        self.assertIsNone(opencode_watchdog._handle_json_event(event, state))
        self.assertIsNone(opencode_watchdog._handle_json_event(event, state))
        self.assertEqual(
            opencode_watchdog._handle_json_event(event, state),
            "repeated_text_loop",
        )

    def test_write_checkpoint_omits_internal_deques(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.json"
            state = {
                "attempt": 1,
                "recent_texts": ["hello"],
                "recent_texts_deque": deque(["hello"], maxlen=8),
                "recent_tool_signatures": [],
                "recent_tool_signatures_deque": deque(maxlen=8),
            }

            opencode_watchdog._write_checkpoint(checkpoint, state)
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertEqual(payload["attempt"], 1)
        self.assertEqual(payload["recent_texts"], ["hello"])
        self.assertNotIn("recent_texts_deque", payload)
        self.assertNotIn("recent_tool_signatures_deque", payload)


if __name__ == "__main__":
    unittest.main()
