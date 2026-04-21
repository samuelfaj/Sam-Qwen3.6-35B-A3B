#!/usr/bin/env bash
set -euo pipefail

CODEX_HOME_DIR="${CODEX_HOME:-${CODEX_HOME_DIR:-/tmp/codex-local-dflash}}"
LOCAL_MODEL_NAME="${LOCAL_DFLASH_MODEL_NAME:-qwen3.6-35b-a3b-dflash-local}"
LOCAL_HOST="${LOCAL_DFLASH_HOST:-127.0.0.1}"
LOCAL_PORT="${LOCAL_DFLASH_PORT:-8010}"

mkdir -p "${CODEX_HOME_DIR}"

cat > "${CODEX_HOME_DIR}/config.toml" <<EOF
model = "${LOCAL_MODEL_NAME}"
model_provider = "localdflash"
approval_policy = "never"
sandbox_mode = "danger-full-access"
model_reasoning_summary = "none"
model_supports_reasoning_summaries = false
model_verbosity = "low"

[model_providers.localdflash]
name = "Local DFlash"
base_url = "http://${LOCAL_HOST}:${LOCAL_PORT}/v1"
wire_api = "responses"
request_max_retries = 1
stream_max_retries = 1
stream_idle_timeout_ms = 600000
supports_websockets = false
EOF

exec env CODEX_HOME="${CODEX_HOME_DIR}" codex "$@"
