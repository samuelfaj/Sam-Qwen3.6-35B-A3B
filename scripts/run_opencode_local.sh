#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${LOCAL_DFLASH_MODEL_NAME:-qwen3.6-35b-a3b-dflash-local}"
OPEN_CODE_MODEL="localdflash/${MODEL_NAME}"

if [[ "${1:-}" == "run" ]]; then
  shift
  exec opencode run -m "${OPEN_CODE_MODEL}" --dangerously-skip-permissions "$@"
fi

exec opencode --model "${OPEN_CODE_MODEL}" "$@"
