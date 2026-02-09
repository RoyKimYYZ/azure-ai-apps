#!/usr/bin/env bash
set -euo pipefail # Ensures the script exits on errors, undefined variables, and failed pipes


az account set --subscription "f1a72634-3a90-4cdb-a6d6-069cf5115068"

# UV Runbook for Python Projects
# This script is a runbook-style reference with safe, non-destructive commands.
# Replace "my_project" with your project folder as needed.

# 1) Install uv and Verify Version
# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# 2) Initialize a New Python Project
uv init my_project
cd my_project
uv python pin 3.13

# 3) Create and Activate a Virtual Environment
uv venv
source .venv/bin/activate

# 4) Add and Lock Dependencies
uv add requests pydantic
uv lock

# 5) Run Scripts and Modules
uv run python -m my_project
uv run pytest

# 6) Manage Dev Dependencies
uv add --dev ruff mypy pytest
uv remove --dev mypy

# 7) Sync Environments Across Machines
uv sync

# 8) Upgrade and Audit Dependencies
uv lock --upgrade
git diff -- uv.lock

# 9) Remove Packages and Clean Cache
uv remove requests
uv cache clean 

# End of runbook

# Testing uv run with a sample command

uv run main.py "show emotions of anger and revenge, then sing a rap song"

uv run cli.py fitness food-images/beef-egg-wrap.jpg

uv run cli.py fitness food-images/eggwrap-chickensalad.jpg


# run ai_chat_client.py with uv
uv run ai_chat_client.py --system "You are a helpful assistant." --user "Tell me a joke about computers."

# run chatbot.py with uv
streamlit run chatbot/chatbot.py