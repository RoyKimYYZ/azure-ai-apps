---
name: agentframework-repo
description: "Repo-specific guidance for implementing and modifying the agentframework/ project in azure-ai-apps. Covers existing ChatAgent patterns, config-driven agent selection, Kaito/Azure Foundry backends, diagnostics integration, SQLite vs Azure SQL handling, and targeted validation commands. WHEN: change agentframework, add agent to chatbot, add BaseChatClient backend, modify chatbot/config.yaml, update prompts, add diagnostics, change thread or memory handling, work on Kaito integration, work on Azure SQL fitness memory, update Streamlit chatbot behavior."
---

# Agent Framework Repo Specialization

Use this skill for work inside `agentframework/` when the task is not just about the generic Microsoft Agent Framework SDK, but about how this repository applies it.

## Use This Skill For

- Adding or modifying agents in `agentframework/`
- Extending `BaseChatClient` implementations such as KAITO-backed clients
- Updating `chatbot/config.yaml`, prompt templates, or provider selection behavior
- Working on diagnostics, thread persistence, or durable fitness memory
- Adjusting Streamlit chatbot behavior without redesigning the UI

## Repo Biases

- Default to `agentframework/` when the user asks about agent orchestration in this repo.
- Preserve the current config-driven architecture instead of hardcoding providers, models, or prompts.
- Reuse existing Agent Framework SDK patterns already present in the repo before introducing new abstractions.
- Prefer targeted changes over framework-wide refactors.

## Key Files

- `agentframework/chatbot/chatbot.py`: Streamlit UI orchestration and agent interaction flow
- `agentframework/chatbot/config.yaml`: provider and agent selection configuration
- `agentframework/ai_chat_client.py`: custom chat client implementations, including KAITO/OpenAI-compatible backends
- `agentframework/main.py`: agent factory patterns and assembly
- `agentframework/fitness_memory.py`: context providers and durable memory integration
- `agentframework/db.py`: database backend access and repository selection
- `agentframework/run_utils.py`: retry and streaming helpers
- `agentframework/prompts/`: prompt templates and structured prompt assets

## Implementation Rules

- Keep the boundary between `ChatAgent` orchestration and backend-specific client logic intact.
- Prefer changes in config, prompt files, or agent factory setup before changing the Streamlit app flow.
- When adding a new backend, follow the existing `BaseChatClient` extension pattern instead of introducing a separate orchestration path.
- When adding memory or profile features, verify whether the active backend is SQLite or Azure SQL and avoid mixing assumptions.
- Reuse the existing diagnostics path and store patterns rather than inventing parallel logging or tracing systems.

## Validation Defaults

- From `agentframework/`, prefer `uv sync` and `uv run ...` commands.
- Use the smallest relevant validation step for the edit:
  - `uv run ruff check .`
  - `uv run mypy .`
  - `uv run pytest`
  - `uv run streamlit run chatbot/chatbot.py`
- If the change is local to one module, prefer targeted validation over full-project runs.

## Azure And AKS Safety

- Read-only Azure investigation is acceptable when needed for diagnostics.
- Do not modify live Azure resources, deploy to Azure, or change AKS state unless the user explicitly asks.
- For local code changes that affect AKS deployment assets, update manifests or scripts only in the repo unless the user explicitly requests rollout actions.

## Relationship To The Generic Skill

The user-level `microsoft-agent-framework` skill should remain the source for broad SDK guidance.
This repo skill narrows that guidance to the conventions and file structure used by `azure-ai-apps/agentframework`.