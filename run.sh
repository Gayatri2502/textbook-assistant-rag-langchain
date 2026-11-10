#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; sys.exit("Python 3.10+ required; set PYTHON_BIN to a newer Python executable.") if sys.version_info < (3,10) else None'
if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -q -r requirements.txt
exec .venv/bin/python -m uvicorn travel_assistant.app:app --host 127.0.0.1 --port "${PORT:-8001}"
