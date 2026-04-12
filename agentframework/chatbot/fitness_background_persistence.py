from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FitnessPersistenceRequest:
    user_id: str
    request_id: str
    selected_model: str | None
    backend_cfg: dict[str, Any] | None
    user_prompt: str
    assistant_text: str
    image_bytes: bytes | None
    image_name: str | None
    session_key: str
    agent_name: str
    serialized_thread: dict[str, Any] | None


@dataclass(slots=True)
class FitnessPersistenceHooks:
    persist_async: Callable[[FitnessPersistenceRequest], Awaitable[None]]
    record_log: Callable[[str, str, str], None]
    logger: logging.Logger


def schedule_fitness_persistence(
    request: FitnessPersistenceRequest,
    *,
    hooks: FitnessPersistenceHooks,
) -> None:
    def _worker() -> None:
        try:
            asyncio.run(hooks.persist_async(request))  # type: ignore[arg-type]
            hooks.record_log(
                "Fitness Nutrition",
                f"Background memory persistence completed for {request.user_id}",
                "INFO",
            )
        except Exception as exc:
            hooks.logger.exception("Background fitness persistence failed")
            hooks.record_log(
                "Fitness Nutrition",
                f"Background memory persistence failed for {request.user_id}: {exc}",
                "ERROR",
            )

    threading.Thread(
        target=_worker,
        name=f"fitness-persist-{request.user_id}",
        daemon=True,
    ).start()
