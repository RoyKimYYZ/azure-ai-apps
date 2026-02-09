#!/usr/bin/env bash
set -euo pipefail

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
