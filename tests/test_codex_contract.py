from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(?P<body>.*?)\n\}}", source, re.M | re.S)
    assert match, f"{name} not found"
    return match.group("body")


def test_codex_wrapper_uses_responses_contract_and_agentic_defaults():
    source = _read("scripts/run_codex_local.sh")

    assert 'wire_api = "responses"' in source
    assert '"base_instructions": "You are an autonomous coding agent running inside Codex CLI.' in source
    assert "For Codex exec shell calls, the shell command argument is command" in source
    assert "use a Python pathlib write_text helper" not in source
    assert "If a test/build script is missing, edit project config" not in source
    assert "Never run dev servers or background commands as verification" not in source
    assert '"instructions_template": "Complete the task end-to-end through tool calls.' in source
    assert 'model_reasoning_effort = "medium"' in source
    assert 'show_raw_agent_reasoning = false' in source
    assert 'hide_agent_reasoning = true' in source
    assert 'LOCAL_INCLUDE_APPLY_PATCH_TOOL="${LOCAL_DFLASH_CODEX_INCLUDE_APPLY_PATCH_TOOL:-false}"' in source
    assert 'LOCAL_CONTEXT_WINDOW="${LOCAL_DFLASH_CODEX_CONTEXT_WINDOW:-${LOCAL_DFLASH_CONTEXT_WINDOW:-32768}}"' in source
    assert 'LOCAL_AUTO_COMPACT_LIMIT="${LOCAL_DFLASH_CODEX_AUTO_COMPACT_LIMIT:-24576}"' in source
    assert 'LOCAL_TOOL_OUTPUT_LIMIT="${LOCAL_DFLASH_CODEX_TOOL_OUTPUT_LIMIT:-12000}"' in source


def test_dflash_codex_does_not_start_server_and_start_uses_agentic_profile():
    source = _read("scripts/dflash.sh")
    cmd_codex = _function_body(source, "cmd_codex")
    cmd_start = _function_body(source, "cmd_start")
    profile = _function_body(source, "apply_profile_114")

    assert "cmd_start" not in cmd_codex
    assert "run_codex_local.sh" in cmd_codex
    assert "prepare_profile_114_args" in cmd_start
    assert "LOCAL_DFLASH_PROFILE=codex-agentic" in profile
    assert "LOCAL_DFLASH_MAX_TOKENS=8192" in profile
    assert "LOCAL_DFLASH_DEFAULT_TEMPERATURE_WITH_TOOLS=0.0" in profile
    assert "LOCAL_DFLASH_KEEP_ALIVE=600" in profile


def test_start_wrapper_honors_background_no_caffeinate_and_timeout_env():
    source = _read("scripts/start_local_wrapper.sh")

    assert 'LOCAL_DFLASH_STREAM_RESULT_TIMEOUT_SECONDS="${LOCAL_DFLASH_STREAM_RESULT_TIMEOUT_SECONDS:-600}"' in source
    assert 'LOCAL_DFLASH_MAX_TOOL_TURN_TOKENS="${LOCAL_DFLASH_MAX_TOOL_TURN_TOKENS:-4096}"' in source
    assert 'DFLASH_NO_CAFFEINATE:-0' in source
    assert 'exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/local_api_server.py" "$@"' in source


def test_snake_smoke_script_contract():
    source = _read("scripts/smoke_codex_snake.sh")

    assert "watchdog" not in source.lower()
    assert "codex exec" in source
    assert "create the famous snake game. Use react, vite and typescript. Create tests to make sure it work." in source
    assert "LOCAL_DFLASH_TRACE_FILE" in source
    assert "timeout_seconds = int(sys.argv[1])" in source
    assert "npm install" in source
    assert "npm run build" in source
    assert "npm test" in source
