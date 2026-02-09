#!/usr/bin/env bash
set -euo pipefail

# Simple runner for ai_foundry_agent_sk Streamlit app + optional tests.
# Usage:
#   ./run_ai_foundry_agent_sk.sh          # run app
#   ./run_ai_foundry_agent_sk.sh test     # run pytest only
#   ./run_ai_foundry_agent_sk.sh both     # run tests then app
#   PORT=8502 ./run_ai_foundry_agent_sk.sh

MODE=${1:-app}
PORT=${PORT:-8501}
APP_MODULE="agent_rag_resume/ai_foundry_agent_sk.py"

# If inside a repo root relative path adjust to script dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$APP_MODULE" ]]; then
  echo "Cannot find $APP_MODULE" >&2
  exit 1
fi

# Provide default (dummy) environment variables if not already set.
export AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-"https://dummy-endpoint.openai.azure.com"}
export AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY:-"test-key"}
export AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT:-"gpt-4o-mini"}
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT=${AZURE_OPENAI_EMBEDDING_DEPLOYMENT:-"text-embedding-3-large"}
export AZURE_SEARCH_ENDPOINT=${AZURE_SEARCH_ENDPOINT:-"https://dummy.search.windows.net"}
export AZURE_SEARCH_API_KEY=${AZURE_SEARCH_API_KEY:-"search-key"}
export AZURE_SEARCH_INDEX_NAME=${AZURE_SEARCH_INDEX_NAME:-"test-index"}
# Optional
export TOP_K=${TOP_K:-5}
export MAX_TOKENS=${MAX_TOKENS:-600}
export TEMPERATURE=${TEMPERATURE:-0.1}

# Ensure dependencies installed (lightweight check)
if ! python -c "import streamlit" >/dev/null 2>&1; then
  echo "Installing dependencies (pip)..." >&2
  pip install -q -e .[dev] || pip install -q -r requirements.txt || true
fi

run_tests() {
  if [[ -d tests && -n $(ls -A tests 2>/dev/null) ]]; then
    echo "Running pytest..." >&2
    pytest -q || { echo "Tests failed" >&2; return 1; }
  else
    echo "No tests directory or tests are empty; skipping." >&2
  fi
}

run_app() {
  echo "Launching Streamlit app on port $PORT" >&2
  exec streamlit run "$APP_MODULE" --server.port="$PORT" --server.headless=true
}

case "$MODE" in
  test)
    run_tests
    ;;
  app)
    run_app
    ;;
  both)
    run_tests && run_app
    ;;
  *)
    echo "Unknown mode: $MODE (expected app|test|both)" >&2
    exit 2
    ;;
esac
