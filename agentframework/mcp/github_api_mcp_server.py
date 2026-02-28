from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("github-api-repo-reader")

GITHUB_API_BASE = "https://api.github.com"
MAX_LIST_RESULTS = 5000
MAX_SEARCH_RESULTS = 200
MAX_FILE_BYTES = 1024 * 1024
MAX_RETURN_LINES = 2000


def _get_github_token() -> str:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_PAT")
    if not token:
        raise ValueError(
            "Missing GitHub token. Set GITHUB_TOKEN (recommended) or GH_PAT in environment."
        )
    return token.strip()


def _validate_non_empty(name: str, value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError(f"{name} is required")
    return cleaned


def _validate_path(path: str) -> str:
    normalized = (path or "").strip().lstrip("/")
    if not normalized:
        return ""
    if ".." in normalized.split("/"):
        raise ValueError("path must not contain '..'")
    if "\\" in normalized:
        raise ValueError("path must not contain backslashes")
    return normalized


def _github_request_json(url: str) -> dict[str, Any]:
    token = _get_github_token()
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-api-mcp-server",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        raise RuntimeError(f"GitHub API request failed: {exc.code} {exc.reason} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub API request failed: {exc.reason}") from exc


def _fetch_repo_tree(owner: str, repo: str, ref: str) -> list[dict[str, Any]]:
    encoded_ref = urllib.parse.quote(ref, safe="")
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{encoded_ref}?recursive=1"
    payload = _github_request_json(url)
    tree = payload.get("tree", [])
    if not isinstance(tree, list):
        return []
    return [item for item in tree if isinstance(item, dict)]


def _slice_lines(content: str, start_line: int, end_line: int) -> tuple[list[str], int, int, int]:
    lines = content.splitlines()
    total_lines = len(lines)

    start = max(1, int(start_line))
    requested_end = max(start, int(end_line))
    capped_end = min(requested_end, start + MAX_RETURN_LINES - 1)

    sliced = lines[start - 1 : capped_end]
    return sliced, total_lines, start, capped_end


@mcp.tool()
def list_repo_files(owner: str, repo: str, ref: str = "main", path: str = "") -> dict[str, Any]:
    """List file paths in a GitHub repository tree under an optional path prefix."""
    owner = _validate_non_empty("owner", owner)
    repo = _validate_non_empty("repo", repo)
    ref = _validate_non_empty("ref", ref)
    path_prefix = _validate_path(path)

    tree = _fetch_repo_tree(owner, repo, ref)
    files: list[str] = []

    for item in tree:
        if item.get("type") != "blob":
            continue
        item_path = str(item.get("path") or "")
        if not item_path:
            continue
        if path_prefix and not item_path.startswith(path_prefix.rstrip("/") + "/") and item_path != path_prefix:
            continue
        files.append(item_path)
        if len(files) >= MAX_LIST_RESULTS:
            break

    return {
        "owner": owner,
        "repo": repo,
        "ref": ref,
        "path_prefix": path_prefix,
        "count": len(files),
        "truncated": len(files) >= MAX_LIST_RESULTS,
        "files": files,
    }


@mcp.tool()
def search_repo_paths(owner: str, repo: str, query: str, ref: str = "main") -> dict[str, Any]:
    """Search file paths in a GitHub repository tree by case-insensitive query terms."""
    owner = _validate_non_empty("owner", owner)
    repo = _validate_non_empty("repo", repo)
    ref = _validate_non_empty("ref", ref)
    query = _validate_non_empty("query", query)

    tokens = [token for token in re.split(r"\s+", query.lower().strip()) if token]
    tree = _fetch_repo_tree(owner, repo, ref)

    matches: list[tuple[int, str]] = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        item_path = str(item.get("path") or "")
        if not item_path:
            continue
        path_lower = item_path.lower()
        if not all(token in path_lower for token in tokens):
            continue

        score = 0
        for token in tokens:
            if path_lower.endswith(token):
                score += 4
            if f"/{token}/" in path_lower:
                score += 2
            if token in path_lower:
                score += 1
        matches.append((score, item_path))

    matches.sort(key=lambda pair: (-pair[0], pair[1]))
    limited = matches[:MAX_SEARCH_RESULTS]

    return {
        "owner": owner,
        "repo": repo,
        "ref": ref,
        "query": query,
        "count": len(limited),
        "truncated": len(matches) > MAX_SEARCH_RESULTS,
        "results": [{"path": path, "score": score} for score, path in limited],
    }


@mcp.tool()
def read_repo_file(
    owner: str,
    repo: str,
    path: str,
    ref: str = "main",
    start_line: int = 1,
    end_line: int = 300,
) -> dict[str, Any]:
    """Read a text file from a GitHub repository with line-range support."""
    owner = _validate_non_empty("owner", owner)
    repo = _validate_non_empty("repo", repo)
    ref = _validate_non_empty("ref", ref)
    path = _validate_non_empty("path", _validate_path(path))

    encoded_path = urllib.parse.quote(path, safe="/")
    encoded_ref = urllib.parse.quote(ref, safe="")
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{encoded_path}?ref={encoded_ref}"
    payload = _github_request_json(url)

    if payload.get("type") != "file":
        raise ValueError(f"{path} is not a file")

    encoding = str(payload.get("encoding") or "")
    encoded_content = payload.get("content")
    if encoding != "base64" or not isinstance(encoded_content, str):
        raise ValueError(f"Unsupported file encoding for {path}: {encoding}")

    raw_bytes = base64.b64decode(encoded_content, validate=False)
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise ValueError(f"File too large ({len(raw_bytes)} bytes). Max allowed is {MAX_FILE_BYTES} bytes")

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = raw_bytes.decode("utf-8", errors="replace")

    sliced, total_lines, returned_start, returned_end = _slice_lines(content, start_line, end_line)

    return {
        "owner": owner,
        "repo": repo,
        "ref": ref,
        "path": path,
        "sha": payload.get("sha"),
        "size_bytes": len(raw_bytes),
        "total_lines": total_lines,
        "returned_start_line": returned_start,
        "returned_end_line": returned_end,
        "content": "\n".join(sliced),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
