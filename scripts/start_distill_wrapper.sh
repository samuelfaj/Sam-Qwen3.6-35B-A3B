#!/usr/bin/env bash
set -euo pipefail

DISTILL_SERVER_CTL="${DISTILL_SERVER_CTL:-/Users/samuelfajreldines/dev/distill-server/scripts/distill-server.sh}"

echo "start_distill_wrapper.sh moved to ${DISTILL_SERVER_CTL}" >&2
exec "${DISTILL_SERVER_CTL}" start "$@"

