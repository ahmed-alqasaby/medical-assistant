#!/usr/bin/env bash
# One-shot development environment setup for medical-assistant (M1).
# Idempotent: safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH — bootstrapping it…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "> creating .venv (${PYTHON_BIN})"
uv venv .venv --python "$PYTHON_BIN"

echo "> installing pinned requirements"
uv pip install --python .venv/bin/python -r requirements.txt

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "> created .env from .env.example (edit values as needed)"
fi

if [ ! -s "$HOME/.kaggle/access_token" ] && [ ! -s "$HOME/.kaggle/kaggle.json" ]; then
  echo "!! Kaggle API token not found at ~/.kaggle"
  echo "   Create one at kaggle.com/settings/account -> API -> Create New Token,"
  echo "   then save it to ~/.kaggle/access_token (chmod 600) or ~/.kaggle/kaggle.json."
fi

echo "> fetching + validating the corpus"
uv run --python .venv/bin/python scripts/fetch_corpus.py --all
uv run --python .venv/bin/python scripts/check_corpus.py

echo "Setup complete. Activate with:  source .venv/bin/activate"