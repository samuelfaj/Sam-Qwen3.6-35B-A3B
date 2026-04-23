#!/usr/bin/env bash
set -euo pipefail

DISTILL_SERVER_CTL="${DISTILL_SERVER_CTL:-/Users/samuelfajreldines/dev/distill-server/scripts/distill-server.sh}"

if [[ ! -x "${DISTILL_SERVER_CTL}" ]]; then
  echo "dflash-distill: distill server control script not found at ${DISTILL_SERVER_CTL}" >&2
  exit 1
fi

case "${1:-help}" in
  distill)
    shift || true
    "${DISTILL_SERVER_CTL}" start >/dev/null
    exec distill "$@"
    ;;
  start|stop|restart|status|health|logs)
    exec "${DISTILL_SERVER_CTL}" "$@"
    ;;
  ""|-h|--help|help)
    cat <<EOF
dflash-distill moved out of dflash.

Use:
  ${DISTILL_SERVER_CTL} start|stop|restart|status|health|logs
  distill <prompt>

This shim exists only to avoid starting the old 35B DFlash distill server.
EOF
    ;;
  *)
    exec "${DISTILL_SERVER_CTL}" "$@"
    ;;
esac

