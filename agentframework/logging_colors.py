import os
import re
import sys
from collections.abc import Sequence
import logging


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def supports_color(stream=None) -> bool:
    stream = stream or sys.stdout

    if os.getenv("NO_COLOR") is not None:
        return False

    term = (os.getenv("TERM") or "").lower()
    if term in {"", "dumb"}:
        return False

    try:
        return bool(stream.isatty())
    except Exception:
        return False


class Ansi:
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"

    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"


_DEFAULT_COLORS = [
    Ansi.CYAN,
    Ansi.GREEN,
    Ansi.YELLOW,
    Ansi.MAGENTA,
    Ansi.BLUE,
    Ansi.RED,
]


_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: Ansi.BLUE,
    logging.INFO: Ansi.GREEN,
    logging.WARNING: Ansi.YELLOW,
    logging.ERROR: Ansi.RED,
    logging.CRITICAL: Ansi.RED + Ansi.BOLD,
}


class ColorLogFormatter(logging.Formatter):
    """A logging formatter that colorizes standard columns for terminal output."""

    def __init__(
        self,
        fmt: str,
        datefmt: str | None = None,
        *,
        enable_color: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._enable_color = enable_color

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        text = super().formatTime(record, datefmt)
        if not self._enable_color:
            return text
        return f"{Ansi.DIM}{text}{Ansi.RESET}"

    def format(self, record: logging.LogRecord) -> str:
        if not self._enable_color:
            return super().format(record)

        original_levelname = record.levelname
        original_name = record.name
        try:
            level_color = _LEVEL_COLORS.get(record.levelno, Ansi.GREEN)
            record.levelname = f"{level_color}{original_levelname}{Ansi.RESET}"
            record.name = f"{Ansi.CYAN}{original_name}{Ansi.RESET}"
            return super().format(record)
        finally:
            record.levelname = original_levelname
            record.name = original_name


def colorize_columns(
    columns: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    sep: str = "  ",
    enabled: bool = True,
) -> str:
    if not enabled:
        return sep.join(columns)

    palette = list(colors) if colors is not None else _DEFAULT_COLORS
    if not palette:
        return sep.join(columns)

    parts: list[str] = []
    for index, value in enumerate(columns):
        color = palette[index % len(palette)]
        parts.append(f"{color}{value}{Ansi.RESET}")
    return sep.join(parts)
