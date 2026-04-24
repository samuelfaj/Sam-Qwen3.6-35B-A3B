#!/usr/bin/env bash
set -euo pipefail

CODEX_HOME_DIR="${CODEX_HOME:-${CODEX_HOME_DIR:-/tmp/codex-local-dflash}}"
LOCAL_MODEL_NAME="${LOCAL_DFLASH_MODEL_NAME:-qwen3.6-35b-a3b-dflash-local}"
LOCAL_HOST="${LOCAL_DFLASH_HOST:-127.0.0.1}"
LOCAL_PORT="${LOCAL_DFLASH_PORT:-8010}"
LOCAL_CONTEXT_WINDOW="${LOCAL_DFLASH_CODEX_CONTEXT_WINDOW:-65536}"
LOCAL_AUTO_COMPACT_LIMIT="${LOCAL_DFLASH_CODEX_AUTO_COMPACT_LIMIT:-49152}"
LOCAL_MAX_OUTPUT_TOKENS="${LOCAL_DFLASH_CODEX_MAX_OUTPUT_TOKENS:-16384}"
LOCAL_STREAM_IDLE_MS="${LOCAL_DFLASH_CODEX_STREAM_IDLE_MS:-1800000}"
LOCAL_BG_TERM_MAX_MS="${LOCAL_DFLASH_CODEX_BG_TERM_MAX_MS:-21600000}"
LOCAL_TOOL_OUTPUT_LIMIT="${LOCAL_DFLASH_CODEX_TOOL_OUTPUT_LIMIT:-32000}"
LOCAL_REQUEST_RETRIES="${LOCAL_DFLASH_CODEX_REQUEST_RETRIES:-2}"
LOCAL_STREAM_RETRIES="${LOCAL_DFLASH_CODEX_STREAM_RETRIES:-3}"
LOCAL_STARTUP_TIMEOUT_SEC="${LOCAL_DFLASH_CODEX_STARTUP_TIMEOUT_SEC:-60}"
LOCAL_TOOL_TIMEOUT_SEC="${LOCAL_DFLASH_CODEX_TOOL_TIMEOUT_SEC:-900}"
LOCAL_INCLUDE_APPLY_PATCH_TOOL="${LOCAL_DFLASH_CODEX_INCLUDE_APPLY_PATCH_TOOL:-false}"

mkdir -p "${CODEX_HOME_DIR}"

echo "${CODEX_HOME_DIR}"

CATALOG_PATH="${CODEX_HOME_DIR}/catalog.json"

cat > "${CATALOG_PATH}" <<'EOF'
{
  "models": [
    {
      "slug": "__LOCAL_MODEL_NAME__",
      "display_name": "Qwen3.6 35B A3B (dflash local)",
      "description": "Local Qwen3.6 35B A3B served by dflash.",
      "default_reasoning_level": "low",
      "supported_reasoning_levels": [
        { "effort": "low",    "description": "Fast, minimal deliberation" },
        { "effort": "medium", "description": "Balanced for everyday tasks" }
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "additional_speed_tiers": [],
      "availability_nux": null,
      "upgrade": null,
      "base_instructions": "",
      "model_messages": {
        "instructions_template": "",
        "instructions_variables": {
          "personality_default": "",
          "personality_friendly": "",
          "personality_pragmatic": ""
        }
      },
      "supports_reasoning_summaries": false,
      "default_reasoning_summary": "none",
      "support_verbosity": true,
      "default_verbosity": "low",
      "apply_patch_tool_type": "function",
      "web_search_tool_type": "text",
      "truncation_policy": {
        "mode": "tokens",
        "limit": 10000
      },
      "supports_parallel_tool_calls": false,
      "supports_image_detail_original": false,
      "context_window": __LOCAL_CONTEXT_WINDOW__,
      "max_context_window": __LOCAL_CONTEXT_WINDOW__,
      "effective_context_window_percent": 95,
      "experimental_supported_tools": [],
      "input_modalities": ["text"],
      "supports_search_tool": false
    }
  ]
}
EOF

# Substitute the templated values without confusing the heredoc.
sed -i '' \
  -e "s/__LOCAL_MODEL_NAME__/${LOCAL_MODEL_NAME}/g" \
  -e "s/__LOCAL_CONTEXT_WINDOW__/${LOCAL_CONTEXT_WINDOW}/g" \
  "${CATALOG_PATH}"

cat > "${CODEX_HOME_DIR}/config.toml" <<EOF
# ----- core selection -----
model = "${LOCAL_MODEL_NAME}"
model_provider = "localdflash"
model_catalog_json = "${CATALOG_PATH}"

# ----- model knobs (non-reasoning local) -----
model_context_window = ${LOCAL_CONTEXT_WINDOW}
model_auto_compact_token_limit = ${LOCAL_AUTO_COMPACT_LIMIT}
model_reasoning_effort = "medium"
plan_mode_reasoning_effort = "low"
model_reasoning_summary = "none"
model_supports_reasoning_summaries = false
model_verbosity = "low"
personality = "pragmatic"

# ----- tools -----
include_apply_patch_tool = ${LOCAL_INCLUDE_APPLY_PATCH_TOOL}
tool_output_token_limit = ${LOCAL_TOOL_OUTPUT_LIMIT}
background_terminal_max_timeout = ${LOCAL_BG_TERM_MAX_MS}
include_permissions_instructions = true
include_apps_instructions = false
include_environment_context = true
# commit_attribution expects a string/struct in Codex 0.122+, not a bool.
# Leave unset so the default applies; users can add
#   commit_attribution = "none"
# (or "always"/"on-commit") manually if they want to override.

# ----- autonomy -----
approval_policy = "never"
sandbox_mode = "danger-full-access"
approvals_reviewer = "user"

# ----- quality-of-life for headless 24h run -----
check_for_update_on_startup = false
suppress_unstable_features_warning = true
disable_paste_burst = true
show_raw_agent_reasoning = true
hide_agent_reasoning = false
file_opener = "none"

[history]
persistence = "save-all"
max_bytes = 1073741824

[shell_environment_policy]
inherit = "core"
ignore_default_excludes = false
exclude = ["ANTHROPIC_*", "OPENAI_*", "GITHUB_TOKEN", "*_API_KEY"]

[analytics]
enabled = false

[feedback]
enabled = false

[memories]
generate_memories = false
use_memories = false
disable_on_external_context = true

[agents]
max_threads = 2
max_depth = 1
job_max_runtime_seconds = 7200

[projects."/Users/samuelfajreldines/dev/dflash"]
trust_level = "trusted"

[features]
apply_patch_freeform = false
unified_exec = false

[model_providers.localdflash]
name = "Local DFlash"
base_url = "http://${LOCAL_HOST}:${LOCAL_PORT}/v1"
wire_api = "responses"
request_max_retries = ${LOCAL_REQUEST_RETRIES}
stream_max_retries = ${LOCAL_STREAM_RETRIES}
stream_idle_timeout_ms = ${LOCAL_STREAM_IDLE_MS}
startup_timeout_sec = ${LOCAL_STARTUP_TIMEOUT_SEC}
tool_timeout_sec = ${LOCAL_TOOL_TIMEOUT_SEC}
supports_websockets = false
supports_parallel_tool_calls = false

[model_providers.localdflash.http_headers]
X-Dflash-Client = "codex-0.122"

[profiles.dflash]
model = "${LOCAL_MODEL_NAME}"
model_provider = "localdflash"
approval_policy = "never"
sandbox_mode = "danger-full-access"
model_reasoning_effort = "low"
model_verbosity = "low"
EOF

CODEX_ARGS=("$@")
if [[ "${CODEX_ARGS[0]:-}" == "exec" ]]; then
  has_skip_git_repo_check=false
  for arg in "${CODEX_ARGS[@]}"; do
    if [[ "${arg}" == "--skip-git-repo-check" ]]; then
      has_skip_git_repo_check=true
      break
    fi
  done
  if [[ "${has_skip_git_repo_check}" == "false" ]]; then
    CODEX_ARGS=("exec" "--skip-git-repo-check" "${CODEX_ARGS[@]:1}")
  fi
fi

if ((${#CODEX_ARGS[@]})); then
  exec env CODEX_HOME="${CODEX_HOME_DIR}" codex "${CODEX_ARGS[@]}"
fi
exec env CODEX_HOME="${CODEX_HOME_DIR}" codex
