"""Autonomous .py file watcher -- runs ruff, mypy, pytest on change."""

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

WATCH_DIR = Path(__file__).resolve().parent
EXCLUDE = {".venv", "__pycache__", "build", "dist", ".git", "node_modules"}
DEBOUNCE_SECS = 2.0

CHECKS = [
    ("ruff", ["uv", "run", "ruff", "check", "."]),
    ("mypy", ["uv", "run", "mypy", "."]),
    ("pytest", ["uv", "run", "pytest", "-q"]),
]

BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def scan(root: Path) -> dict[str, float]:
    """Return {filepath: mtime} for all .py files under root."""
    result: dict[str, float] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE]
        for f in filenames:
            if f.endswith(".py") and not f.startswith(".file-watcher"):
                p = os.path.join(dirpath, f)
                with contextlib.suppress(OSError):
                    result[p] = os.stat(p).st_mtime
    return result


def run_checks(changed: list[str]) -> None:
    rel = [os.path.relpath(p, WATCH_DIR) for p in changed]
    print(f"\n{BLUE}{BOLD}{'=' * 60}{RESET}")
    print(f"{BLUE}{BOLD}  Changed: {', '.join(rel)}{RESET}")
    print(f"{BLUE}{BOLD}{'=' * 60}{RESET}\n")

    all_pass = True
    for name, cmd in CHECKS:
        print(f"{YELLOW}{BOLD}▶ {name}{RESET}")
        result = subprocess.run(cmd, cwd=WATCH_DIR, capture_output=True, text=True)
        output = (result.stdout + result.stderr).strip()

        if result.returncode == 0:
            summary = output.split("\n")[-1] if output else "OK"
            print(f"  {GREEN}✔ passed{RESET}  {DIM}{summary}{RESET}\n")
        else:
            all_pass = False
            print(f"  {RED}✘ failed (exit {result.returncode}){RESET}")
            for line in output.split("\n"):
                print(f"  {DIM}{line}{RESET}")
            print()

    if all_pass:
        print(f"{GREEN}{BOLD}✔ All checks passed{RESET}\n")
    else:
        print(f"{RED}{BOLD}✘ Some checks failed{RESET}\n")


def main() -> None:
    print(f"{BLUE}{BOLD}👀 Watching .py files in {WATCH_DIR}{RESET}")
    print(f"{DIM}   Checks: ruff → mypy → pytest{RESET}")
    print(f"{DIM}   Press Ctrl+C to stop{RESET}\n")

    prev = scan(WATCH_DIR)

    try:
        while True:
            time.sleep(1)
            curr = scan(WATCH_DIR)

            changed: list[str] = []
            for p, mtime in curr.items():
                if p not in prev or prev[p] != mtime:
                    changed.append(p)
            for p in prev:
                if p not in curr:
                    changed.append(p)

            if changed:
                time.sleep(DEBOUNCE_SECS)
                curr = scan(WATCH_DIR)
                run_checks(changed)
                prev = curr
    except KeyboardInterrupt:
        print(f"\n{DIM}Watcher stopped.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
