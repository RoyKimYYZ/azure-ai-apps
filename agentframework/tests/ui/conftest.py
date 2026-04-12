"""Playwright fixtures for Streamlit UI tests.

Starts the Streamlit app as a subprocess before the test session and tears
it down afterwards.  Each test gets a fresh Playwright browser page pointed
at the running app.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from playwright.sync_api import Page

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHATBOT_SCRIPT = "chatbot/chatbot.py"  # relative to _PROJECT_ROOT

# How long to wait for Streamlit to become responsive (seconds).
_STARTUP_TIMEOUT = 30
_POLL_INTERVAL = 0.5


def _free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = _STARTUP_TIMEOUT) -> None:
    """Block until the Streamlit server is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(_POLL_INTERVAL)
    raise TimeoutError(f"Streamlit did not start within {timeout}s on port {port}")


@pytest.fixture(scope="session")
def streamlit_port() -> int:
    return _free_port()


@pytest.fixture(scope="session")
def streamlit_server(streamlit_port: int) -> Generator[subprocess.Popen[bytes]]:
    """Launch Streamlit as a child process for the entire test session.

    Uses ``uv run streamlit run …`` to match the project's normal launch
    method and ensure the ``chatbot`` package is importable correctly.
    """
    uv_bin = shutil.which("uv") or "uv"
    proc = subprocess.Popen(
        [
            uv_bin,
            "run",
            "streamlit",
            "run",
            _CHATBOT_SCRIPT,
            "--server.port",
            str(streamlit_port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_server(streamlit_port)
        yield proc
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="session")
def app_url(streamlit_port: int) -> str:
    return f"http://127.0.0.1:{streamlit_port}"


@pytest.fixture()
def ui_page(
    page: Page,
    streamlit_server: subprocess.Popen[bytes],
    app_url: str,
) -> Page:
    """Navigate to the app and wait for the Streamlit shell to load."""
    page.goto(app_url, wait_until="networkidle")
    # Wait for the Streamlit app frame to be ready.
    page.wait_for_selector("[data-testid='stApp']", timeout=15_000)
    return page
