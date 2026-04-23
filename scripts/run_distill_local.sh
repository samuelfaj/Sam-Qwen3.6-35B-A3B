#!/usr/bin/env bash
set -euo pipefail

DISTILL_SERVER_CTL="${DISTILL_SERVER_CTL:-/Users/samuelfajreldines/dev/distill-server/scripts/distill-server.sh}"
DISTILL_SERVER_HOST="${DISTILL_SERVER_HOST:-127.0.0.1}"
DISTILL_SERVER_PORT="${DISTILL_SERVER_PORT:-8022}"
DISTILL_MODEL="${DISTILL_MODEL:-qwen3.5-9b-ddtree-distill-local}"
DISTILL_HOST="${DISTILL_HOST:-http://${DISTILL_SERVER_HOST}:${DISTILL_SERVER_PORT}/v1}"
DISTILL_API_KEY="${DISTILL_API_KEY:-local-distill-server}"

export DISTILL_MODEL DISTILL_HOST DISTILL_API_KEY

if [[ "${DISTILL_SERVER_AUTOSTART:-1}" == "1" ]]; then
  "${DISTILL_SERVER_CTL}" start >/dev/null
fi

exec distill "$@"
