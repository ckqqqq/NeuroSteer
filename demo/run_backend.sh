#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs/demo
# Bind locally by default. Use an explicit host override when deploying remotely.
python -m uvicorn demo.backend.main_fastapp_backend:app \
    --host "${NEUROSTEER_HOST:-127.0.0.1}" --port "${NEUROSTEER_PORT:-8000}" \
    2>&1 | tee -a logs/demo/backend.log
