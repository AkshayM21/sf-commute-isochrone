#!/usr/bin/env bash
# Runnable entry point for the SF Commute Explorer end-to-end browser suite.
#
# Prereqs (one-time, already done in this environment):
#   uv pip install -p .venv/bin/python playwright pytest-playwright
#   .venv/bin/python -m playwright install chromium
#
# The featured graph-native Flask server must be running at http://127.0.0.1:8000:
#   .venv/bin/python scripts/server.py
#
# Usage:
#   tests/e2e/run.sh                       # full suite (desktop + mobile), headless
#   tests/e2e/run.sh test_desktop.py       # just desktop specs
#   tests/e2e/run.sh -k autocomplete       # filter by name
#   HEADED=1 tests/e2e/run.sh              # watch it run in a real browser window
#   E2E_BASE_URL=http://host:port tests/e2e/run.sh   # point at a different server
set -euo pipefail

# Resolve repo root from this script's location (works from any cwd).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "error: $PY not found. Create the repo-local .venv first (see CLAUDE.md)." >&2
  exit 1
fi

BASE="${E2E_BASE_URL:-http://127.0.0.1:8000}"
if ! curl -sf -o /dev/null "$BASE/"; then
  echo "error: server not reachable at $BASE — start it with: $PY scripts/server.py" >&2
  exit 1
fi

# Run from tests/e2e so pytest.ini + conftest.py apply and `import conftest` works.
cd "$HERE"
if [[ "${HEADED:-0}" == "1" ]]; then
  exec "$PY" -m pytest --headed "$@"
fi
exec "$PY" -m pytest "$@"
