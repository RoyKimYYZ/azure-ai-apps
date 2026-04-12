from __future__ import annotations

import json
import logging
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64decode
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from agent_framework_compat import Agent, ChatOptions
from agent_framework.openai import OpenAIChatClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from jinja2 import Template
from pydantic import BaseModel

from db import DEFAULT_DB_PATH, StructuredOutputStore
from run_utils import run_with_retry, run_with_stream

logger = logging.getLogger(__name__)


class WorkflowPlan(BaseModel):
    steps: list[str]
    rationale: str


class StructuredOutput(BaseModel):
    steps: list[str]
    rationale: str
    type: str
    grounding_path: str | None = None
    confidence: str = "partial"


@dataclass(frozen=True)
class Agent1Config:
    prompt_template_path: Path
    stream_output: bool
    system_prompt_override: str | None = None
    target_repo_owner: str = "RoyKimYYZ"
    target_repo_name: str = "aks-demos"
    target_repo_ref: str = "main"
    step1_prompt: str = "Tell me a joke about a pirate."


class Agent1DemoRunner:
    def __init__(self, config: Agent1Config) -> None:
        """Initialize runner state and DB store; called by module-level `agent1()` entrypoint."""
        self._config = config
        self._store = StructuredOutputStore(DEFAULT_DB_PATH)

    @staticmethod
    def _load_prompt_template(path: Path) -> dict[str, Any]:
        """Load YAML prompt template and resolve `=Env.*` model IDs; called by `_create_agent()`."""
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        model_id = data.get("model", {}).get("id")
        if isinstance(model_id, str) and model_id.startswith("=Env."):
            env_key = model_id.removeprefix("=Env.")
            data["model"]["id"] = os.getenv(env_key, model_id)

        return data

    @staticmethod
    def _render_instructions(template: str, context: dict[str, str]) -> str:
        """Render prompt instructions via Jinja or `str.format`; called by `_create_agent()`."""
        if "{{" in template and "}}" in template:
            return Template(template).render(**context)
        return template.format(**context)

    @staticmethod
    def _to_serializable_payload(result: Any) -> dict[str, Any]:
        """Normalize model/SDK outputs into a serializable dict; called by `run()`."""
        if hasattr(result, "model_dump"):
            payload = result.model_dump()
            return payload if isinstance(payload, dict) else {"value": payload}

        parsed = getattr(result, "parsed", None)
        if isinstance(parsed, dict):
            return parsed

        text = getattr(result, "text", None)
        if isinstance(text, str):
            try:
                loaded = json.loads(text)
                if isinstance(loaded, dict):
                    return loaded
            except json.JSONDecodeError:
                pass
            return {
                "steps": [text],
                "rationale": "",
                "type": "Chat",
            }

        if isinstance(result, dict):
            return result

        return {
            "steps": [str(result)],
            "rationale": "",
            "type": "Chat",
        }

    @staticmethod
    def _extract_keywords(text: str, max_terms: int = 8) -> list[str]:
        """Extract simple keyword set for repo search; called by `_fetch_repo_context_from_prompt()`."""
        stop_words = {
            "the", "and", "for", "with", "from", "that", "this", "have", "your", "about",
            "what", "when", "where", "would", "could", "should", "into", "just", "like", "show",
            "tell", "please", "help", "step", "agent", "repo", "file", "code", "workflow",
        }
        words = re.findall(r"[a-zA-Z0-9_\-]{3,}", text.lower())
        ordered_unique: list[str] = []
        for word in words:
            if word in stop_words:
                continue
            if word not in ordered_unique:
                ordered_unique.append(word)
            if len(ordered_unique) >= max_terms:
                break
        return ordered_unique

    @staticmethod
    def _print_json(payload: Any) -> None:
        """Pretty-print JSON payloads to stdout for step logs; called by step print helpers and `run()`."""
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))

    def _resolve_step1_prompt(self, user_message: str | None) -> str:
        """Resolve Step 1 prompt from UI input, then CLI arg, then config default; called by `run()`."""
        cli_message = sys.argv[1] if len(sys.argv) > 1 else ""
        return (user_message or "").strip() or cli_message.strip() or self._config.step1_prompt

    @staticmethod
    def _github_api_get_json(url: str, token: str | None) -> Any:
        """Issue authenticated GitHub GET and parse JSON response; called by `_fetch_repo_context_from_prompt()`."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "agent1-demo",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        request = urllib.request.Request(url=url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)

    # The following helper functions are adapted from github_api_mcp_server.py for reuse in this agent demo, but are not identical to the versions in that file.
    def _fetch_repo_context_from_prompt(self, previous_response_text: str) -> dict[str, Any]:
        """Search repo tree and read a candidate file excerpt for grounding; called by `run()` after Step 1."""
        owner = self._config.target_repo_owner
        repo = self._config.target_repo_name
        ref = self._config.target_repo_ref
        token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")


        keywords = self._extract_keywords(previous_response_text)
        query = " ".join(keywords) if keywords else "readme"

        repo_context: dict[str, Any] = {
            "repo": f"{owner}/{repo}@{ref}",
            "query": query,
            "candidate_paths": [],
            "selected_path": None,
            "excerpt": None,
            "error": None,
        }

        try:
            tree_url = (
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/"
                f"{urllib.parse.quote(ref, safe='')}?recursive=1"
            )
            tree_payload = self._github_api_get_json(tree_url, token)
            entries = tree_payload.get("tree", []) if isinstance(tree_payload, dict) else []

            file_paths = [
                entry.get("path", "")
                for entry in entries
                if isinstance(entry, dict) and entry.get("type") == "blob" and isinstance(entry.get("path"), str)
            ]

            def score(path: str) -> int:
                lowered = path.lower()
                return sum(2 if lowered.endswith(term) else 1 for term in keywords if term in lowered)

            sorted_candidates = sorted(file_paths, key=lambda p: (score(p), -len(p)), reverse=True)
            candidate_paths = [p for p in sorted_candidates if score(p) > 0][:5]
            if not candidate_paths:
                fallback = [p for p in file_paths if p.lower().endswith("readme.md")]
                candidate_paths = fallback[:1] if fallback else file_paths[:1]

            repo_context["candidate_paths"] = candidate_paths
            if not candidate_paths:
                repo_context["error"] = "No files found in target repository tree."
                return repo_context

            selected_path = candidate_paths[0]
            repo_context["selected_path"] = selected_path

            content_url = (
                f"https://api.github.com/repos/{owner}/{repo}/contents/"
                f"{urllib.parse.quote(selected_path)}?ref={urllib.parse.quote(ref, safe='')}"
            )
            content_payload = self._github_api_get_json(content_url, token)

            if not isinstance(content_payload, dict):
                repo_context["error"] = "Unexpected response while reading selected file."
                return repo_context

            content_text = ""
            if content_payload.get("encoding") == "base64" and isinstance(content_payload.get("content"), str):
                content_text = b64decode(content_payload["content"], validate=False).decode("utf-8", errors="replace")
            elif isinstance(content_payload.get("content"), str):
                content_text = content_payload["content"]

            excerpt_lines = content_text.splitlines()[:80]
            repo_context["excerpt"] = "\n".join(excerpt_lines).strip() if excerpt_lines else ""
            return repo_context

        except urllib.error.HTTPError as exc:
            repo_context["error"] = f"GitHub API HTTP {exc.code}"
            return repo_context
        except Exception as exc:
            repo_context["error"] = str(exc)
            return repo_context

    def _build_workflow_prompt(self, previous_response_text: str, repo_context: dict[str, Any] | None = None) -> str:
        """Build Step 2 planning prompt from Step 1 output plus optional repo context; called by `run()`."""
        behavior_contract = (
            "Behavior contract for this agent:\n"
            "1) Understand the user ask and identify likely target files.\n"
            "2) Search repository paths before reading files.\n"
            "3) Read the best candidate file and ground answers in that content.\n"
            "4) If file evidence is missing, explicitly say what is unknown.\n"
            f"Target repo: {self._config.target_repo_owner}/{self._config.target_repo_name}"
            f" @ {self._config.target_repo_ref}."
        )
        repo_context_block = ""
        if repo_context:
            repo_context_block = (
                "\n\nRepository context fetched for grounding:\n"
                f"- Query terms: {repo_context.get('query')}\n"
                f"- Candidate paths: {repo_context.get('candidate_paths')}\n"
                f"- Selected path: {repo_context.get('selected_path')}\n"
                f"- Fetch error: {repo_context.get('error')}\n"
                "- Selected file excerpt:\n"
                f"{repo_context.get('excerpt') or '[no excerpt available]'}"
            )
        return (
            "Using the previous response, create a simple 4-step workflow that shows how "
            "an agent would proceed. Return a JSON object with keys: steps (array of strings) "
            "and rationale (string).\n\n"
            f"{behavior_contract}\n\n"
            f"{repo_context_block}\n\n"
            "Previous response:\n"
            f"{previous_response_text}"
        )

    @staticmethod
    def _derive_confidence(repo_context: dict[str, Any] | None) -> str:
        """Map repo-context availability to `grounded`/`partial`; called by final prompt/defaulting helpers."""
        if not repo_context:
            return "partial"
        has_excerpt = bool((repo_context.get("excerpt") or "").strip())
        has_error = bool(repo_context.get("error"))
        return "grounded" if has_excerpt and not has_error else "partial"

    def _build_final_prompt(self, workflow_payload: dict[str, Any], repo_context: dict[str, Any] | None = None) -> str:
        """Build Step 3 structured-output prompt with grounding hints; called by `run()`."""
        selected_path = repo_context.get("selected_path") if repo_context else None
        confidence = self._derive_confidence(repo_context)
        return (
            "Using this workflow plan, produce the final structured output as JSON with keys: "
            "steps (array of strings), rationale (string), type (string, must be 'Chat'), "
            "grounding_path (string or null), and confidence (string, either 'grounded' or 'partial').\n\n"
            "Rules:\n"
            "- confidence must be 'grounded' only when you used file evidence.\n"
            "- grounding_path should be the repo file path used as evidence, or null if none.\n\n"
            f"Grounding hints: selected_path={selected_path}, suggested_confidence={confidence}.\n\n"
            f"Workflow plan:\n{workflow_payload}"
        )

    @staticmethod
    def _build_evidence_block(repo_context: dict[str, Any] | None) -> dict[str, Any]:
        """Create compact Step 4 evidence summary for display; called by `run()`."""
        if not repo_context:
            return {
                "grounding_path": None,
                "matched_lines": [],
                "error": "No repository context available.",
            }

        excerpt = (repo_context.get("excerpt") or "").strip()
        query = (repo_context.get("query") or "").strip()
        keywords = [term for term in query.split() if term]

        matched_lines: list[str] = []
        if excerpt and keywords:
            for line in excerpt.splitlines():
                lowered = line.lower()
                if any(term.lower() in lowered for term in keywords):
                    matched_lines.append(line.strip())
                if len(matched_lines) >= 3:
                    break

        return {
            "grounding_path": repo_context.get("selected_path"),
            "matched_lines": matched_lines,
            "error": repo_context.get("error"),
        }

    async def _create_agent(self, data_input: str = "") -> Agent:
        """Instantiate the chat agent from template/config; called once by `run()`."""
        prompt = self._load_prompt_template(self._config.prompt_template_path)
        rendered_instructions = self._render_instructions(
            prompt.get("instructions", "You are a helpful assistant."),
            {"data_input": data_input},
        )
        instructions = (self._config.system_prompt_override or "").strip() or rendered_instructions

        model_block = prompt.get("model", {})
        model_id = model_block.get("id") if isinstance(model_block, dict) else model_block

        chat_client = OpenAIChatClient(model=model_id, credential=AzureCliCredential())
        return Agent(
            client=chat_client,
            instructions=instructions,
            name=prompt.get("name", "Assistant"),
            default_options=ChatOptions(model=model_id, max_tokens=prompt.get("max_tokens")),  # type: ignore[typeddict-item]
            tools=prompt.get("tools", []),
        )

    async def _run_step1(self, agent: Agent, step1_prompt: str) -> str:
        """Execute Step 1 user prompt (streaming or retry path) and return text; called by `run()`."""
        if self._config.stream_output:
            print("Step 1 result (streaming):")
            step1_result = await run_with_stream(agent, step1_prompt)
        else:
            step1_result = await run_with_retry(agent, step1_prompt)
            print("Step 1 result:\n", step1_result.text)
        return step1_result.text if hasattr(step1_result, "text") else str(step1_result)

    def _print_step15_repo_context(self, repo_context: dict[str, Any]) -> None:
        """Print Step 1.5 repo-context diagnostic block; called by `run()`."""
        print("Step 1.5 repo context:\n")
        self._print_json(
            {
                "repo": repo_context.get("repo"),
                "query": repo_context.get("query"),
                "candidate_paths": repo_context.get("candidate_paths"),
                "selected_path": repo_context.get("selected_path"),
                "error": repo_context.get("error"),
            }
        )

    @staticmethod
    def _extract_step3_payload(final_result: Any) -> Any:
        """Extract printable payload from Step 3 result object; called by `run()`."""
        if hasattr(final_result, "model_dump"):
            return final_result.model_dump()
        if hasattr(final_result, "parsed"):
            return final_result.parsed
        return final_result

    def _apply_grounding_defaults(self, final_payload: dict[str, Any], repo_context: dict[str, Any]) -> dict[str, Any]:
        """Backfill missing grounding fields in final payload; called by `run()` before persistence."""
        if "grounding_path" not in final_payload:
            final_payload["grounding_path"] = repo_context.get("selected_path")
        if "confidence" not in final_payload:
            final_payload["confidence"] = self._derive_confidence(repo_context)
        return final_payload

    def _persist_and_print_outputs(self, final_payload: dict[str, Any]) -> None:
        """Persist final output and print DB contents for inspection; called at end of `run()`."""
        output_id = self._store.insert(
            steps=final_payload.get("steps", []),
            rationale=final_payload.get("rationale", ""),
            output_type=final_payload.get("type", "Chat"),
        )
        print(f"Saved structured output with id={output_id}")

        all_structured_outputs = self._store.list_all()
        print("All structured outputs in the database:")
        for output in all_structured_outputs:
            if hasattr(output, "model_dump"):
                payload = output.model_dump()
            elif hasattr(output, "_asdict"):
                payload = output._asdict()
            elif isinstance(output, dict):
                payload = output
            else:
                try:
                    payload = dict(output)
                except Exception:
                    payload = {"value": output}
            print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))

    async def run(self, user_message: str | None = None) -> None:
        """Orchestrate Steps 1 → 4 and persistence; called by module-level `agent1()` wrapper."""
        logger.info("Starting agent1 demo - Hello from agentframework!")
        step1_prompt = self._resolve_step1_prompt(user_message)
        agent = await self._create_agent(data_input=step1_prompt)

        step1_text = await self._run_step1(agent, step1_prompt)
        repo_context = self._fetch_repo_context_from_prompt(step1_text)
        self._print_step15_repo_context(repo_context)

        workflow_prompt = self._build_workflow_prompt(step1_text, repo_context)
        workflow_plan = await run_with_retry(agent, workflow_prompt, response_format=WorkflowPlan)
        print("Step 2 workflow:\n", workflow_plan)

        workflow_payload = self._to_serializable_payload(workflow_plan)
        structured_prompt = self._build_final_prompt(workflow_payload, repo_context)
        final_result = await run_with_retry(agent, structured_prompt, response_format=StructuredOutput)
        print("Step 3 structured output:")
        payload = self._extract_step3_payload(final_result)
        self._print_json(payload)

        final_payload = self._to_serializable_payload(final_result)
        final_payload = self._apply_grounding_defaults(final_payload, repo_context)

        evidence_block = self._build_evidence_block(repo_context)
        print("Step 4 evidence block:")
        self._print_json(evidence_block)

        self._persist_and_print_outputs(final_payload)


def build_agent1_config() -> Agent1Config:
    """Build runtime config from environment variables; called by module-level `agent1()`."""
    prompt_path = Path(
        os.getenv(
            "AGENT1_PROMPT_TEMPLATE_PATH",
            os.getenv(
                "PROMPT_TEMPLATE_PATH",
                str(Path(__file__).parent / "prompts" / "assistant_jinja.yaml"),
            ),
        )
    )
    return Agent1Config(
        prompt_template_path=prompt_path,
        stream_output=os.getenv("STREAM_OUTPUT", "1") == "1",
        system_prompt_override=os.getenv("AGENT1_SYSTEM_PROMPT_OVERRIDE"),
        target_repo_owner=os.getenv("AGENT1_TARGET_REPO_OWNER", "RoyKimYYZ"),
        target_repo_name=os.getenv("AGENT1_TARGET_REPO_NAME", "aks-demos"),
        target_repo_ref=os.getenv("AGENT1_TARGET_REPO_REF", "main"),
    )


async def agent1(user_message: str | None = None) -> None:
    """Public async entrypoint used by CLI and Streamlit Agent1 flow."""
    load_dotenv()
    runner = Agent1DemoRunner(build_agent1_config())
    await runner.run(user_message=user_message)
