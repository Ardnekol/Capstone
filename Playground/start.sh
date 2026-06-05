#!/usr/bin/env bash
# Florence-2 Playground — production startup script
# Builds the frontend and launches the FastAPI server on port 8000.
# The server serves both the API (/api/*) and the React UI (/).
#
# Usage:
#   ./start.sh               # default model (Florence-2-large)
#   PORT=9000 ./start.sh     # custom port
#   MODEL_CACHE_DIR=/nvme/hf ./start.sh   # custom HF cache dir
#   TORCH_DTYPE=bfloat16 ./start.sh       # use bfloat16 on A100/H100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo "=== Florence-2 Playground ==="

# ── 1. Build frontend ─────────────────────────────────────────────────────────
echo "[1/2] Building frontend..."
cd "$FRONTEND_DIR"
npm install --silent
npm run build
echo "      Frontend built → frontend/dist/"

# ── 2. Kill any lingering server processes ────────────────────────────────────
_PORT="${PORT:-7860}"
pkill -f "python.*main\.py" 2>/dev/null || true
pkill -f "uvicorn.*main"    2>/dev/null || true
_PIDS="$(lsof -ti :"$_PORT" 2>/dev/null || true)"
if [ -n "$_PIDS" ]; then
  echo "      Freeing port $_PORT (PIDs: $_PIDS)..."
  echo "$_PIDS" | xargs kill -9 2>/dev/null || true
fi
sleep 1

# ── 3. Start backend (serves static + API) ───────────────────────────────────
echo "[2/2] Starting backend on port $_PORT..."
cd "$BACKEND_DIR"

# Ensure production env is set
export SERVE_STATIC=true
export STATIC_DIR="${STATIC_DIR:-../frontend/dist}"

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
exec "$PYTHON" main.py
