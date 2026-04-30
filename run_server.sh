#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

RELOAD_ARGS=()
if [[ "${RELOAD:-1}" == "1" ]]; then
  RELOAD_ARGS=(
    --reload
    --reload-dir .
    --reload-include "maira_api.py"
    --reload-include "static/*.html"
  )
fi

exec uvicorn maira_api:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8100}" "${RELOAD_ARGS[@]}"
