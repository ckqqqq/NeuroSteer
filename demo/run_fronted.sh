#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs/demo
python -m streamlit run demo/frontend/main_streamlit_frontend.py \
    --server.address "${NEUROSTEER_HOST:-127.0.0.1}" --server.port "${NEUROSTEER_FRONTEND_PORT:-8501}" \
    2>&1 | tee -a logs/demo/frontend.log
