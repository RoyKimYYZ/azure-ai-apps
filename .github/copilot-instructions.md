# Copilot Instructions for azure-ai-apps

This repository is a collection of standalone Python demo applications focused on Azure AI, RAG patterns, Streamlit chat UIs, and agent orchestration. Unless the user explicitly targets another subproject, prefer to reason from the top-level README and treat `agentframework/` as the primary project for detailed implementation guidance.

## Repo Priorities

- Prioritize `agentframework/` first when the user asks a repo-level question without naming a subproject.
- Treat each subproject as independently runnable. Avoid making cross-project changes unless the user explicitly wants shared refactoring.
- Keep changes minimal and local. Do not broad-refactor demo apps that are already working unless the task requires it.
- Preserve the existing Streamlit UI style and interaction patterns unless the user asks for a redesign.
- Prefer config-driven changes when practical: update YAML, `.env`, prompts, or other configuration before making larger code changes.
- If behavior changes, update the nearest relevant README when the change materially affects setup, execution, or usage.

## Working Conventions

- Start from the project README and local project files before making architectural assumptions.
- For `agentframework/`, prefer existing patterns built around Microsoft Agent Framework SDK concepts such as `ChatAgent`, `BaseChatClient`, `ContextProvider`, thread/state handling, structured output, and config-driven agent selection.
- Reuse existing diagnostics and observability patterns instead of inventing parallel mechanisms.
- Be careful with durable memory and data access code. Distinguish clearly between SQLite local development flows and Azure SQL flows.
- Treat prompt templates, config files, and environment variables as first-class parts of the application, not incidental files.

## Validation Defaults

- In `uv`-based projects, prefer `uv sync` and `uv run ...` over raw `python` or `pip` commands.
- In `agentframework/`, use the smallest relevant validation command for the task before proposing larger validation runs.
- When Python code changes, prefer to run relevant lint, type-check, or test commands when they are available and proportional to the edit.
- Preferred validation tools in this repo are `ruff`, `mypy`, `pytest`, and targeted local runs such as `uv run streamlit run ...` when UI validation is relevant.
- Some subprojects still use `requirements.txt`; use the project's existing package-management pattern instead of forcing `uv` everywhere.

## Azure and Hosting Guardrails

- Follow Azure-specific guidance and best practices whenever the task involves Azure services, Azure SDKs, Azure deployment assets, or Azure operations.
- Read-only Azure investigation is acceptable by default when needed for diagnostics or understanding the environment.
- Treat **Azure App Service and AKS (Kubernetes)** as equal first-class hosting targets. Do not assume AKS-only when deployment or hosting concerns arise.
- Do not modify live Azure resources, run deployments, apply Kubernetes manifests, restart workloads, or change AKS or App Service state unless the user explicitly asks for that action.
- Avoid touching live hosted resources by default, even if local manifest or script changes are being made.
- Prefer safe inspection and explanation before any operational command that changes cloud or cluster state.

## `agentframework/` Biases

- Prefer `uv run streamlit run chatbot/chatbot.py` for local UI execution.
- Keep agent behavior aligned with the existing config-driven architecture in `chatbot/config.yaml`, prompt files, and provider selection flows.
- When working on agent backends, preserve the abstraction boundary between orchestration and backend client implementation.
- Favor targeted changes around agent selection, prompt handling, diagnostics, database access, and Streamlit integration rather than introducing new frameworks.
- If a task touches testing, add or update focused unit tests where practical rather than introducing large new test scaffolding.
- **Persistence default:** When adding any new feature that writes data (metrics, meals, profile fields, goals, etc.), default to immediate in-request persistence so the UI reflects changes in the same rerender cycle. Only use background/async writes if the user explicitly asks or if the operation is too expensive to block on. Controlled via the `FITNESS_PERSISTENCE_MODE` env var (`inline` default; `async` to opt out).
- **Save/persist UI indicator:** When adding new Streamlit UI that triggers a database write, use an ephemeral `st.empty()` placeholder to show a brief `"⏳ Saving to memory…"` message that is cleared (`.empty()`) immediately after the write completes. Do not leave persistent status text in the chat thread.

## Repo Skill

- A repo-specific companion skill lives at `.github/skills/agentframework-repo/SKILL.md`.
- Use it for `agentframework/` implementation details that are specific to this codebase rather than generic Microsoft Agent Framework SDK usage.
- Keep critical always-on rules in this file; use the skill for richer project-specific workflow guidance.

## Response Style for This Repo

- Be pragmatic and concrete.
- Prefer actionable steps over broad theory.
- Surface assumptions when the selected subproject or runtime target is ambiguous.
- If the user asks for repo-wide changes, call out which subprojects are affected before editing.