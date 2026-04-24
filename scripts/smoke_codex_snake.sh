#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${1:-/Users/samuelfajreldines/dev/test}"
PROMPT="create the famous snake game. Use react, vite and typescript. Create tests to make sure it work."
ARTIFACT_ROOT="${LOCAL_DFLASH_SMOKE_ARTIFACTS:-${REPO_ROOT}/artifacts/codex-smoke}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
ARTIFACT_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}"
TIMEOUT_SECONDS="${LOCAL_DFLASH_CODEX_SMOKE_TIMEOUT_SECONDS:-2700}"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "smoke: python3 not found" >&2
  exit 2
fi

mkdir -p "${TARGET_DIR}" "${ARTIFACT_DIR}"

echo "smoke: artifacts ${ARTIFACT_DIR}"
echo "smoke: cleaning ${TARGET_DIR}"
find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

export LOCAL_DFLASH_TRACE_FILE="${LOCAL_DFLASH_TRACE_FILE:-${ARTIFACT_DIR}/dflash-trace.jsonl}"

echo "smoke: starting dflash"
bash "${SCRIPT_DIR}/dflash.sh" start >"${ARTIFACT_DIR}/dflash-start.log" 2>&1

curl -fsS --max-time 5 "http://${LOCAL_DFLASH_HOST:-127.0.0.1}:${LOCAL_DFLASH_PORT:-8010}/health" \
  >"${ARTIFACT_DIR}/health-before.json" 2>"${ARTIFACT_DIR}/health-before.err" || true
curl -fsS --max-time 5 "http://${LOCAL_DFLASH_HOST:-127.0.0.1}:${LOCAL_DFLASH_PORT:-8010}/metrics" \
  >"${ARTIFACT_DIR}/metrics-before.json" 2>"${ARTIFACT_DIR}/metrics-before.err" || true

echo "smoke: running codex exec"
"${PYTHON_BIN}" - "${TIMEOUT_SECONDS}" "${ARTIFACT_DIR}/codex-exec.log" \
  bash "${SCRIPT_DIR}/dflash.sh" codex exec --cd "${TARGET_DIR}" "${PROMPT}" <<'PY'
import subprocess
import sys
from pathlib import Path

timeout_seconds = int(sys.argv[1])
log_path = Path(sys.argv[2])
cmd = sys.argv[3:]

with log_path.open("wb") as log:
    log.write(("COMMAND: " + " ".join(cmd) + "\n").encode())
    log.flush()
    try:
        completed = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.write(f"\nTIMEOUT after {timeout_seconds}s\n".encode())
        sys.exit(124)
sys.exit(completed.returncode)
PY

APP_DIR="${TARGET_DIR}/snake-game"
echo "smoke: validating ${APP_DIR}"
find "${TARGET_DIR}" -maxdepth 3 -print >"${ARTIFACT_DIR}/project-tree.txt"

if [[ ! -d "${APP_DIR}" ]]; then
  echo "smoke: missing app dir ${APP_DIR}" >&2
  exit 1
fi
if [[ ! -f "${APP_DIR}/package.json" ]]; then
  echo "smoke: missing package.json" >&2
  exit 1
fi
if [[ ! -f "${APP_DIR}/src/App.tsx" ]]; then
  echo "smoke: missing src/App.tsx" >&2
  exit 1
fi

cp "${APP_DIR}/package.json" "${ARTIFACT_DIR}/package.json"
cp "${APP_DIR}/src/App.tsx" "${ARTIFACT_DIR}/App.tsx"

"${PYTHON_BIN}" - "${APP_DIR}" <<'PY'
import json
import sys
from pathlib import Path

app_dir = Path(sys.argv[1])
package = json.loads((app_dir / "package.json").read_text())
deps = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
scripts = package.get("scripts", {})
missing = []
for name in ("react", "vite", "typescript"):
    if name not in deps:
        missing.append(name)
for script in ("build", "test"):
    if script not in scripts:
        missing.append(f"script:{script}")
app = (app_dir / "src" / "App.tsx").read_text().lower()
source_parts = []
for source_path in sorted((app_dir / "src").glob("**/*")):
    if source_path.suffix.lower() not in {".ts", ".tsx"}:
        continue
    if "node_modules" in source_path.parts:
        continue
    source_parts.append(source_path.read_text(errors="ignore"))
source = "\n".join(source_parts).lower()
checks = {
    "snake": "snake" in source or "cobra" in source,
    "food": "food" in source or "comida" in source,
    "state": "usestate" in source or "usereducer" in source,
    "collision": "collision" in source or "collid" in source or "gameover" in source,
    "score": "score" in source,
    "keyboard": "keydown" in source or "keyup" in source,
    "restart": "restart" in source or "reset" in source,
    "not_template": "vite + react" not in app and "learn react" not in app,
}
missing.extend([f"app:{name}" for name, ok in checks.items() if not ok])
if missing:
    print("smoke: validation failed: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY

(
  cd "${APP_DIR}"
  npm install
) >"${ARTIFACT_DIR}/npm-install.log" 2>&1

(
  cd "${APP_DIR}"
  npm test -- --run
) >"${ARTIFACT_DIR}/npm-test.log" 2>&1

(
  cd "${APP_DIR}"
  npm run build
) >"${ARTIFACT_DIR}/npm-build.log" 2>&1

curl -fsS --max-time 5 "http://${LOCAL_DFLASH_HOST:-127.0.0.1}:${LOCAL_DFLASH_PORT:-8010}/health" \
  >"${ARTIFACT_DIR}/health-after.json" 2>"${ARTIFACT_DIR}/health-after.err" || true
curl -fsS --max-time 5 "http://${LOCAL_DFLASH_HOST:-127.0.0.1}:${LOCAL_DFLASH_PORT:-8010}/metrics" \
  >"${ARTIFACT_DIR}/metrics-after.json" 2>"${ARTIFACT_DIR}/metrics-after.err" || true

echo "smoke: PASS ${ARTIFACT_DIR}"
