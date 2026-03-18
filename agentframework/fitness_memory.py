from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from agent_framework import ChatMessage, Context, ContextProvider
from pydantic import BaseModel, Field, model_validator

from app_settings import Settings


logger = logging.getLogger(__name__)
_SKIP_UPDATE = object()
SQL_COPT_SS_ACCESS_TOKEN = 1256
AZURE_SQL_TOKEN_SCOPE = "https://database.windows.net/.default"


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _json_default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            pass
    return str(value)


def _to_json_text(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def _row_to_dict(description: Any, row: Any) -> dict[str, Any]:
    return {column[0]: value for column, value in zip(description, row)}


def _import_azure_sql_runtime() -> tuple[Any, Any]:
    try:
        import pyodbc
    except ImportError as exc:
        raise RuntimeError(
            "Azure SQL support requires the pyodbc package and the unixODBC runtime. "
            "Install unixODBC and ODBC Driver 18 for SQL Server before using FITNESS_DB_BACKEND=azuresql."
        ) from exc

    try:
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise RuntimeError(
            "Azure SQL support requires azure-identity. Install project dependencies before using FITNESS_DB_BACKEND=azuresql."
        ) from exc

    return pyodbc, DefaultAzureCredential


def _encode_sql_access_token(token: str) -> bytes:
    encoded = token.encode("utf-16-le")
    return struct.pack(f"<I{len(encoded)}s", len(encoded), encoded)


def _normalize_sql_server_name(server: str) -> str:
    normalized = server.strip()
    if not normalized:
        raise ValueError("AZURE_SQL_SERVER must be set when FITNESS_DB_BACKEND=azuresql")
    if ".database.windows.net" not in normalized:
        return f"{normalized}.database.windows.net"
    return normalized


def _normalize_thread_state(state: dict[str, Any]) -> dict[str, Any]:
    store = state.get("chat_message_store_state")
    if not isinstance(store, dict):
        return state

    messages = store.get("messages")
    if not isinstance(messages, list):
        return state

    normalized_messages: list[Any] = []
    for message in messages:
        if isinstance(message, dict):
            try:
                normalized_messages.append(ChatMessage.from_dict(message))
                continue
            except Exception:
                normalized_messages.append(message)
                continue
        normalized_messages.append(message)

    store["messages"] = normalized_messages
    state["chat_message_store_state"] = store
    return state


class ProfileUpdate(BaseModel):
    field: Literal[
        "name",
        "birthday_mmddyyyy",
        "height_value",
        "height_unit",
        "city",
        "country",
        "sex",
        "timezone",
        "external_user_key",
    ]
    value: str | float | int | None


class MealUpsert(BaseModel):
    occurred_at: str | None = None
    meal_type: Literal["breakfast", "lunch", "dinner", "snack", "other"] | None = None
    source_image_uri: str | None = None
    source_hash: str | None = None
    unit_system: Literal["metric", "imperial"] | None = None
    notes: str | None = None


class MealItemUpsert(BaseModel):
    food_label: str
    quantity_value: float | None = None
    quantity_unit: str | None = None
    confidence: float | None = None
    notes: str | None = None


class MacroEventInsert(BaseModel):
    calories_kcal: float | None = Field(default=None, ge=0)
    protein_g: float | None = Field(default=None, ge=0)
    carbs_g: float | None = Field(default=None, ge=0)
    fat_g: float | None = Field(default=None, ge=0)
    fiber_g: float | None = Field(default=None, ge=0)
    sugar_g: float | None = Field(default=None, ge=0)
    sodium_mg: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    model_name: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    notes: str | None = None


class BodyMetricEventInsert(BaseModel):
    metric_type: Literal["weight", "waist", "blood_pressure"]
    value_primary: float = Field(gt=0)
    value_secondary: float | None = Field(default=None, gt=0)
    unit: Literal["lbs", "kg", "in", "cm", "mmHg"]
    observed_at: str | None = None
    source: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    notes: str | None = None

    # Validation to ensure blood_pressure has value_secondary and unit mmHg, and that weight and waist have appropriate units.
    # @model_validator is used here to perform cross-field validation after the initial model validation.
    @model_validator(mode="after")
    def validate_metric_shape(self) -> "BodyMetricEventInsert":
        if self.metric_type == "blood_pressure":
            if self.value_secondary is None:
                raise ValueError("value_secondary is required for blood_pressure")
            if self.unit != "mmHg":
                raise ValueError("blood_pressure unit must be mmHg")
            return self

        if self.value_secondary is not None:
            raise ValueError("value_secondary must be null unless metric_type is blood_pressure")

        if self.metric_type == "weight" and self.unit not in {"lbs", "kg"}:
            raise ValueError("weight unit must be lbs or kg")
        if self.metric_type == "waist" and self.unit not in {"in", "cm"}:
            raise ValueError("waist unit must be in or cm")
        return self


class PersistenceOp(BaseModel):
    operation: Literal["insert", "update", "upsert"]
    target: str
    idempotency_key: str | None = None


class PhotoSubmissionStructuredOutput(BaseModel):
    profile_updates: list[ProfileUpdate] = Field(default_factory=list)
    meal_upsert: MealUpsert | None = None
    meal_items_upsert: list[MealItemUpsert] = Field(default_factory=list)
    macro_events_insert: list[MacroEventInsert] = Field(default_factory=list)
    body_metric_events_insert: list[BodyMetricEventInsert] = Field(default_factory=list)
    persistence_ops: list[PersistenceOp] = Field(default_factory=list)


class TextTurnStructuredOutput(BaseModel):
    profile_updates: list[ProfileUpdate] = Field(default_factory=list)
    body_metric_events_insert: list[BodyMetricEventInsert] = Field(default_factory=list)
    persistence_ops: list[PersistenceOp] = Field(default_factory=list)


@dataclass
class IngestionRunStart:
    run_id: str
    created_at: str


class FitnessMemoryRepository(Protocol):
    def get_read_model(self, user_id: str, *, metric_limit: int = 10, meal_limit: int = 10) -> dict[str, Any]: ...

    def start_ingestion_run(
        self,
        *,
        user_id: str,
        source_type: str,
        idempotency_key: str | None,
        request_json: dict[str, Any] | None,
    ) -> IngestionRunStart: ...

    def finish_ingestion_run(
        self,
        *,
        run_id: str,
        status: Literal["completed", "failed"],
        response_json: dict[str, Any] | None,
        structured_output_json: dict[str, Any] | None,
        error: str | None,
    ) -> None: ...

    def apply_photo_submission(
        self,
        *,
        user_id: str,
        image_path: str,
        payload: PhotoSubmissionStructuredOutput,
        raw_structured_output: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]: ...

    def apply_text_turn_submission(
        self,
        *,
        user_id: str,
        payload: TextTurnStructuredOutput,
        raw_structured_output: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]: ...

    def load_thread_state(self, *, user_id: str, session_key: str, agent_name: str) -> dict[str, Any] | None: ...

    def upsert_thread_state(
        self,
        *,
        user_id: str,
        session_key: str,
        agent_name: str,
        session_state: dict[str, Any],
        summary_text: str | None,
    ) -> None: ...


class SQLiteFitnessMemoryRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_user(self, conn: sqlite3.Connection, user_id: str) -> None:
        conn.execute(
            """
            INSERT INTO users (user_id, name, created_at, updated_at, is_active)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(user_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (user_id, user_id, utc_now_iso(), utc_now_iso()),
        )

    def _apply_profile_updates(self, conn: sqlite3.Connection, user_id: str, updates: list[ProfileUpdate]) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "input_count": len(updates),
            "applied_count": 0,
            "skipped_count": 0,
            "applied_fields": [],
            "normalized_fields": [],
            "skipped_fields": [],
        }
        if not updates:
            return diagnostics

        normalized_updates: list[tuple[str, Any, Any]] = []
        for update in updates:
            normalized = self._normalize_profile_update_value(update.field, update.value)
            if normalized is _SKIP_UPDATE:
                diagnostics["skipped_count"] += 1
                diagnostics["skipped_fields"].append(update.field)
                continue
            normalized_updates.append((update.field, normalized, update.value))

        if not normalized_updates:
            return diagnostics

        assignments: list[str] = []
        values: list[Any] = []
        for field, value, original_value in normalized_updates:
            assignments.append(f"{field} = ?")
            values.append(value)
            diagnostics["applied_fields"].append(field)
            if value != original_value:
                diagnostics["normalized_fields"].append(field)
        assignments.append("updated_at = ?")
        values.append(utc_now_iso())
        values.append(user_id)
        conn.execute(f"UPDATE users SET {', '.join(assignments)} WHERE user_id = ?", values)
        diagnostics["applied_count"] = len(normalized_updates)
        return diagnostics

    @staticmethod
    def _normalize_profile_update_value(field: str, value: Any) -> Any:
        if field == "birthday_mmddyyyy":
            return SQLiteFitnessMemoryRepository._normalize_birthday(value)

        if field == "height_value":
            if value in (None, ""):
                return None
            try:
                height_value = float(value)
            except (TypeError, ValueError):
                logger.warning("Skipping invalid height_value update: %r", value)
                return _SKIP_UPDATE
            if height_value <= 0:
                logger.warning("Skipping non-positive height_value update: %r", value)
                return _SKIP_UPDATE
            return height_value

        if field == "name":
            if value is None:
                logger.warning("Skipping null name update to preserve NOT NULL constraint")
                return _SKIP_UPDATE
            text = str(value).strip()
            if not text:
                logger.warning("Skipping empty name update to preserve NOT NULL constraint")
                return _SKIP_UPDATE
            return text

        if isinstance(value, str):
            return value.strip()
        return value

    @staticmethod
    def _normalize_birthday(value: Any) -> Any:
        if value in (None, ""):
            return None

        text = str(value).strip()
        parse_formats = [
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%m%d%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]

        for fmt in parse_formats:
            try:
                parsed = datetime.strptime(text, fmt)
                return parsed.strftime("%m/%d/%Y")
            except ValueError:
                continue

        logger.warning("Skipping invalid birthday_mmddyyyy update (unrecognized format): %r", value)
        return _SKIP_UPDATE

    def _insert_body_metrics(self, conn: sqlite3.Connection, user_id: str, events: list[BodyMetricEventInsert]) -> int:
        created = 0
        for event in events:
            conn.execute(
                """
                INSERT INTO body_metric_events (
                    event_id, user_id, metric_type, value_primary, value_secondary,
                    unit, observed_at, source, confidence, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    user_id,
                    event.metric_type,
                    event.value_primary,
                    event.value_secondary,
                    event.unit,
                    event.observed_at or utc_now_iso(),
                    event.source,
                    event.confidence,
                    event.notes,
                    utc_now_iso(),
                ),
            )
            created += 1
        return created

    def _aggregate_macros(self, events: list[MacroEventInsert]) -> MacroEventInsert:
        if not events:
            return MacroEventInsert()
        summed = MacroEventInsert()
        numeric_fields = ["calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg"]
        for field_name in numeric_fields:
            total = sum((getattr(event, field_name) or 0.0) for event in events)
            if total > 0:
                setattr(summed, field_name, total)
        for field_name in ["confidence", "model_name", "model_version", "prompt_version", "notes"]:
            first_value = next((getattr(event, field_name) for event in events if getattr(event, field_name) is not None), None)
            setattr(summed, field_name, first_value)
        return summed

    def _upsert_meal_event(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        image_path: str,
        meal: MealUpsert | None,
        macro_events: list[MacroEventInsert],
        raw_structured_output: dict[str, Any],
    ) -> str | None:
        if meal is None and not macro_events:
            return None

        aggregate = self._aggregate_macros(macro_events)
        meal_row = meal or MealUpsert()
        source_hash = meal_row.source_hash
        if not source_hash:
            source_hash = hashlib.sha256(f"{user_id}:{image_path}".encode("utf-8")).hexdigest()

        existing = conn.execute(
            "SELECT meal_event_id FROM meal_events WHERE user_id = ? AND source_hash = ?",
            (user_id, source_hash),
        ).fetchone()

        if existing:
            meal_event_id = str(existing["meal_event_id"])
            conn.execute(
                """
                UPDATE meal_events
                SET occurred_at = ?, meal_type = ?, source_image_uri = ?,
                    unit_system = ?, calories_kcal = ?, protein_g = ?, carbs_g = ?, fat_g = ?,
                    fiber_g = ?, sugar_g = ?, sodium_mg = ?, confidence = ?, model_name = ?,
                    model_version = ?, prompt_version = ?, llm_structured_output_json = ?, notes = ?
                WHERE meal_event_id = ?
                """,
                (
                    meal_row.occurred_at or utc_now_iso(),
                    meal_row.meal_type,
                    meal_row.source_image_uri or image_path,
                    meal_row.unit_system,
                    aggregate.calories_kcal,
                    aggregate.protein_g,
                    aggregate.carbs_g,
                    aggregate.fat_g,
                    aggregate.fiber_g,
                    aggregate.sugar_g,
                    aggregate.sodium_mg,
                    aggregate.confidence,
                    aggregate.model_name,
                    aggregate.model_version,
                    aggregate.prompt_version,
                    json.dumps(raw_structured_output),
                    aggregate.notes or meal_row.notes,
                    meal_event_id,
                ),
            )
            return meal_event_id

        meal_event_id = str(uuid4())
        conn.execute(
            """
            INSERT INTO meal_events (
                meal_event_id, user_id, occurred_at, meal_type, source_image_uri, source_hash,
                calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg,
                unit_system, confidence, model_name, model_version, prompt_version,
                llm_structured_output_json, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meal_event_id,
                user_id,
                meal_row.occurred_at or utc_now_iso(),
                meal_row.meal_type,
                meal_row.source_image_uri or image_path,
                source_hash,
                aggregate.calories_kcal,
                aggregate.protein_g,
                aggregate.carbs_g,
                aggregate.fat_g,
                aggregate.fiber_g,
                aggregate.sugar_g,
                aggregate.sodium_mg,
                meal_row.unit_system,
                aggregate.confidence,
                aggregate.model_name,
                aggregate.model_version,
                aggregate.prompt_version,
                json.dumps(raw_structured_output),
                aggregate.notes or meal_row.notes,
                utc_now_iso(),
            ),
        )
        return meal_event_id

    def get_read_model(self, user_id: str, *, metric_limit: int = 10, meal_limit: int = 10) -> dict[str, Any]:
        with self._conn() as conn:
            profile_row = conn.execute(
                """
                SELECT user_id, external_user_key, name, birthday_mmddyyyy, height_value, height_unit,
                       city, country, sex, timezone, created_at, updated_at, is_active
                FROM users WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            metric_rows = conn.execute(
                """
                SELECT event_id, metric_type, value_primary, value_secondary, unit, observed_at,
                       source, confidence, notes, created_at
                FROM body_metric_events
                WHERE user_id = ?
                ORDER BY observed_at DESC
                LIMIT ?
                """,
                (user_id, metric_limit),
            ).fetchall()
            meal_rows = conn.execute(
                """
                SELECT meal_event_id, occurred_at, meal_type, source_image_uri, source_hash,
                       calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg,
                       unit_system, confidence, model_name, model_version, prompt_version,
                       llm_structured_output_json, notes, created_at
                FROM meal_events
                WHERE user_id = ?
                ORDER BY occurred_at DESC
                LIMIT ?
                """,
                (user_id, meal_limit),
            ).fetchall()

        profile = dict(profile_row) if profile_row else {}
        metrics = [dict(row) for row in metric_rows]
        meals = [dict(row) for row in meal_rows]
        return {"profile": profile, "recent_body_metrics": metrics, "recent_meals": meals}

    # The following methods implement the protocol for starting and finishing ingestion runs, applying photo submission structured outputs, and loading/upserting thread state for agent sessions. These methods interact with the SQLite database to persist and retrieve the necessary information for the fitness agent's memory and context management.
    def start_ingestion_run(
        self,
        *,
        user_id: str,
        source_type: str,
        idempotency_key: str | None,
        request_json: dict[str, Any] | None,
    ) -> IngestionRunStart:
        run_id = str(uuid4())
        now = utc_now_iso()
        with self._conn() as conn:
            self._ensure_user(conn, user_id)
            conn.execute(
                """
                INSERT INTO ingestion_runs (
                    run_id, user_id, source_type, idempotency_key, request_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, 'started', ?)
                """,
                (run_id, user_id, source_type, idempotency_key, json.dumps(request_json or {}), now),
            )
        return IngestionRunStart(run_id=run_id, created_at=now)

    def finish_ingestion_run(
        self,
        *,
        run_id: str,
        status: Literal["completed", "failed"],
        response_json: dict[str, Any] | None,
        structured_output_json: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE ingestion_runs
                SET status = ?, response_json = ?, structured_output_json = ?, error = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    json.dumps(response_json) if response_json is not None else None,
                    json.dumps(structured_output_json) if structured_output_json is not None else None,
                    error,
                    run_id,
                ),
            )

    def apply_photo_submission(
        self,
        *,
        user_id: str,
        image_path: str,
        payload: PhotoSubmissionStructuredOutput,
        raw_structured_output: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        with self._conn() as conn:
            self._ensure_user(conn, user_id)
            profile_debug = self._apply_profile_updates(conn, user_id, payload.profile_updates)
            body_metric_count = self._insert_body_metrics(conn, user_id, payload.body_metric_events_insert)
            meal_event_id = self._upsert_meal_event(
                conn,
                user_id=user_id,
                image_path=image_path,
                meal=payload.meal_upsert,
                macro_events=payload.macro_events_insert,
                raw_structured_output=raw_structured_output,
            )
        return {
            "idempotency_key": idempotency_key,
            "meal_event_id": meal_event_id,
            "body_metric_count": body_metric_count,
            "profile_update_count": len(payload.profile_updates),
            "meal_items_detected": len(payload.meal_items_upsert),
            "profile_debug": profile_debug,
        }

    def apply_text_turn_submission(
        self,
        *,
        user_id: str,
        payload: TextTurnStructuredOutput,
        raw_structured_output: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        with self._conn() as conn:
            self._ensure_user(conn, user_id)
            profile_debug = self._apply_profile_updates(conn, user_id, payload.profile_updates)
            body_metric_count = self._insert_body_metrics(conn, user_id, payload.body_metric_events_insert)
        return {
            "idempotency_key": idempotency_key,
            "profile_update_count": len(payload.profile_updates),
            "body_metric_count": body_metric_count,
            "structured_output_keys": sorted(raw_structured_output.keys()),
            "profile_debug": profile_debug,
        }

    def load_thread_state(self, *, user_id: str, session_key: str, agent_name: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT session_json
                FROM agent_session_memory
                WHERE user_id = ? AND session_key = ? AND agent_name = ?
                LIMIT 1
                """,
                (user_id, session_key, agent_name),
            ).fetchone()
        if row is None:
            return None
        loaded = _safe_json_loads(row["session_json"])
        return _normalize_thread_state(loaded)

    def upsert_thread_state(
        self,
        *,
        user_id: str,
        session_key: str,
        agent_name: str,
        session_state: dict[str, Any],
        summary_text: str | None,
    ) -> None:
        memory_id = hashlib.sha256(f"{user_id}:{session_key}:{agent_name}".encode("utf-8")).hexdigest()[:64]
        now = utc_now_iso()
        with self._conn() as conn:
            self._ensure_user(conn, user_id)
            existing = conn.execute(
                """
                SELECT memory_id
                FROM agent_session_memory
                WHERE user_id = ? AND session_key = ? AND agent_name = ?
                LIMIT 1
                """,
                (user_id, session_key, agent_name),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE agent_session_memory
                    SET session_json = ?, summary_text = ?, last_event_at = ?
                    WHERE memory_id = ?
                    """,
                    (_to_json_text(session_state), summary_text, now, str(existing["memory_id"])),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO agent_session_memory (
                        memory_id, user_id, session_key, agent_name, session_json,
                        summary_text, last_event_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (memory_id, user_id, session_key, agent_name, _to_json_text(session_state), summary_text, now, now),
                )


class AzureSqlFitnessMemoryRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.server = _normalize_sql_server_name(settings.azure_sql_server)
        self.database = settings.azure_sql_database.strip()
        self.schema = settings.azure_sql_schema.strip() or "dbo"
        self.driver = settings.azure_sql_driver.strip() or "ODBC Driver 18 for SQL Server"
        self.auth_mode = settings.azure_sql_auth_mode
        self.admin_user = settings.azure_sql_admin_user.strip()
        self.admin_password = settings.azure_sql_admin_password
        self.encrypt = settings.azure_sql_encrypt
        self.trust_server_certificate = settings.azure_sql_trust_server_certificate
        self.connection_timeout = settings.azure_sql_connection_timeout

        if not self.database:
            raise ValueError("AZURE_SQL_DATABASE must be set when FITNESS_DB_BACKEND=azuresql")
        if self.auth_mode not in {
            "defaultazurecredential",
            "default-azure-credential",
            "default_azure_credential",
            "adminpassword",
            "sqlpassword",
            "sql-password",
        }:
            raise ValueError(f"Unsupported AZURE_SQL_AUTH_MODE={self.auth_mode}")
        if self.auth_mode in {"adminpassword", "sqlpassword", "sql-password"}:
            if not self.admin_user or not self.admin_password:
                raise ValueError(
                    "AZURE_SQL_ADMIN_USER and AZURE_SQL_ADMIN_PASSWORD must be set when AZURE_SQL_AUTH_MODE uses SQL password auth"
                )

    def _table(self, name: str) -> str:
        return f"[{self.schema}].[{name}]"

    def _conn(self) -> Any:
        pyodbc, DefaultAzureCredential = _import_azure_sql_runtime()
        connection_string = (
            f"Driver={{{self.driver}}};"
            f"Server=tcp:{self.server},1433;"
            f"Database={self.database};"
            f"Encrypt={'yes' if self.encrypt else 'no'};"
            f"TrustServerCertificate={'yes' if self.trust_server_certificate else 'no'};"
            f"Connection Timeout={self.connection_timeout};"
        )
        if self.auth_mode in {"adminpassword", "sqlpassword", "sql-password"}:
            connection_string = f"{connection_string}UID={self.admin_user};PWD={self.admin_password};"
            return pyodbc.connect(connection_string, autocommit=False)

        credential = DefaultAzureCredential(
            managed_identity_client_id=self.settings.azure_client_id or None,
            exclude_interactive_browser_credential=False,
        )
        access_token = credential.get_token(AZURE_SQL_TOKEN_SCOPE).token
        token_struct = _encode_sql_access_token(access_token)
        return pyodbc.connect(connection_string, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct}, autocommit=False)

    def _ensure_user(self, cursor: Any, user_id: str) -> None:
        now = utc_now_iso()
        existing = cursor.execute(
            f"SELECT user_id FROM {self._table('users')} WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if existing:
            cursor.execute(
                f"UPDATE {self._table('users')} SET updated_at = ? WHERE user_id = ?",
                (now, user_id),
            )
            return
        cursor.execute(
            f"""
            INSERT INTO {self._table('users')} (user_id, name, created_at, updated_at, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (user_id, user_id, now, now),
        )

    def _apply_profile_updates(self, cursor: Any, user_id: str, updates: list[ProfileUpdate]) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "input_count": len(updates),
            "applied_count": 0,
            "skipped_count": 0,
            "applied_fields": [],
            "normalized_fields": [],
            "skipped_fields": [],
        }
        if not updates:
            return diagnostics

        normalized_updates: list[tuple[str, Any, Any]] = []
        for update in updates:
            normalized = SQLiteFitnessMemoryRepository._normalize_profile_update_value(update.field, update.value)
            if normalized is _SKIP_UPDATE:
                diagnostics["skipped_count"] += 1
                diagnostics["skipped_fields"].append(update.field)
                continue
            normalized_updates.append((update.field, normalized, update.value))

        if not normalized_updates:
            return diagnostics

        assignments: list[str] = []
        values: list[Any] = []
        for field, value, original_value in normalized_updates:
            assignments.append(f"{field} = ?")
            values.append(value)
            diagnostics["applied_fields"].append(field)
            if value != original_value:
                diagnostics["normalized_fields"].append(field)
        assignments.append("updated_at = ?")
        values.append(utc_now_iso())
        values.append(user_id)
        cursor.execute(f"UPDATE {self._table('users')} SET {', '.join(assignments)} WHERE user_id = ?", values)
        diagnostics["applied_count"] = len(normalized_updates)
        return diagnostics

    def _insert_body_metrics(self, cursor: Any, user_id: str, events: list[BodyMetricEventInsert]) -> int:
        created = 0
        for event in events:
            cursor.execute(
                f"""
                INSERT INTO {self._table('body_metric_events')} (
                    event_id, user_id, metric_type, value_primary, value_secondary,
                    unit, observed_at, source, confidence, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    user_id,
                    event.metric_type,
                    event.value_primary,
                    event.value_secondary,
                    event.unit,
                    event.observed_at or utc_now_iso(),
                    event.source,
                    event.confidence,
                    event.notes,
                    utc_now_iso(),
                ),
            )
            created += 1
        return created

    def _upsert_meal_event(
        self,
        cursor: Any,
        *,
        user_id: str,
        image_path: str,
        meal: MealUpsert | None,
        macro_events: list[MacroEventInsert],
        raw_structured_output: dict[str, Any],
    ) -> str | None:
        if meal is None and not macro_events:
            return None

        aggregate = SQLiteFitnessMemoryRepository._aggregate_macros(self, macro_events)
        meal_row = meal or MealUpsert()
        source_hash = meal_row.source_hash
        if not source_hash:
            source_hash = hashlib.sha256(f"{user_id}:{image_path}".encode("utf-8")).hexdigest()

        existing = cursor.execute(
            f"SELECT meal_event_id FROM {self._table('meal_events')} WHERE user_id = ? AND source_hash = ?",
            (user_id, source_hash),
        ).fetchone()

        if existing:
            meal_event_id = str(existing[0])
            cursor.execute(
                f"""
                UPDATE {self._table('meal_events')}
                SET occurred_at = ?, meal_type = ?, source_image_uri = ?,
                    unit_system = ?, calories_kcal = ?, protein_g = ?, carbs_g = ?, fat_g = ?,
                    fiber_g = ?, sugar_g = ?, sodium_mg = ?, confidence = ?, model_name = ?,
                    model_version = ?, prompt_version = ?, llm_structured_output_json = ?, notes = ?
                WHERE meal_event_id = ?
                """,
                (
                    meal_row.occurred_at or utc_now_iso(),
                    meal_row.meal_type,
                    meal_row.source_image_uri or image_path,
                    meal_row.unit_system,
                    aggregate.calories_kcal,
                    aggregate.protein_g,
                    aggregate.carbs_g,
                    aggregate.fat_g,
                    aggregate.fiber_g,
                    aggregate.sugar_g,
                    aggregate.sodium_mg,
                    aggregate.confidence,
                    aggregate.model_name,
                    aggregate.model_version,
                    aggregate.prompt_version,
                    _to_json_text(raw_structured_output),
                    aggregate.notes or meal_row.notes,
                    meal_event_id,
                ),
            )
            return meal_event_id

        meal_event_id = str(uuid4())
        cursor.execute(
            f"""
            INSERT INTO {self._table('meal_events')} (
                meal_event_id, user_id, occurred_at, meal_type, source_image_uri, source_hash,
                calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg,
                unit_system, confidence, model_name, model_version, prompt_version,
                llm_structured_output_json, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meal_event_id,
                user_id,
                meal_row.occurred_at or utc_now_iso(),
                meal_row.meal_type,
                meal_row.source_image_uri or image_path,
                source_hash,
                aggregate.calories_kcal,
                aggregate.protein_g,
                aggregate.carbs_g,
                aggregate.fat_g,
                aggregate.fiber_g,
                aggregate.sugar_g,
                aggregate.sodium_mg,
                meal_row.unit_system,
                aggregate.confidence,
                aggregate.model_name,
                aggregate.model_version,
                aggregate.prompt_version,
                _to_json_text(raw_structured_output),
                aggregate.notes or meal_row.notes,
                utc_now_iso(),
            ),
        )
        return meal_event_id

    def get_read_model(self, user_id: str, *, metric_limit: int = 10, meal_limit: int = 10) -> dict[str, Any]:
        metric_limit = max(int(metric_limit), 0)
        meal_limit = max(int(meal_limit), 0)
        with self._conn() as conn:
            cursor = conn.cursor()
            profile_cursor = cursor.execute(
                f"""
                SELECT user_id, external_user_key, name, birthday_mmddyyyy, height_value, height_unit,
                       city, country, sex, timezone, created_at, updated_at, is_active
                FROM {self._table('users')} WHERE user_id = ?
                """,
                (user_id,),
            )
            profile_row = profile_cursor.fetchone()
            profile = _row_to_dict(profile_cursor.description, profile_row) if profile_row else {}

            metric_cursor = cursor.execute(
                f"""
                SELECT TOP {metric_limit} event_id, metric_type, value_primary, value_secondary, unit, observed_at,
                       source, confidence, notes, created_at
                FROM {self._table('body_metric_events')}
                WHERE user_id = ?
                ORDER BY observed_at DESC
                """,
                (user_id,),
            )
            metrics = [_row_to_dict(metric_cursor.description, row) for row in metric_cursor.fetchall()]

            meal_cursor = cursor.execute(
                f"""
                SELECT TOP {meal_limit} meal_event_id, occurred_at, meal_type, source_image_uri, source_hash,
                       calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg,
                       unit_system, confidence, model_name, model_version, prompt_version,
                       llm_structured_output_json, notes, created_at
                FROM {self._table('meal_events')}
                WHERE user_id = ?
                ORDER BY occurred_at DESC
                """,
                (user_id,),
            )
            meals = [_row_to_dict(meal_cursor.description, row) for row in meal_cursor.fetchall()]

        return {"profile": profile, "recent_body_metrics": metrics, "recent_meals": meals}

    def start_ingestion_run(
        self,
        *,
        user_id: str,
        source_type: str,
        idempotency_key: str | None,
        request_json: dict[str, Any] | None,
    ) -> IngestionRunStart:
        run_id = str(uuid4())
        now = utc_now_iso()
        with self._conn() as conn:
            cursor = conn.cursor()
            self._ensure_user(cursor, user_id)
            cursor.execute(
                f"""
                INSERT INTO {self._table('ingestion_runs')} (
                    run_id, user_id, source_type, idempotency_key, request_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, 'started', ?)
                """,
                (run_id, user_id, source_type, idempotency_key, _to_json_text(request_json or {}), now),
            )
            conn.commit()
        return IngestionRunStart(run_id=run_id, created_at=now)

    def finish_ingestion_run(
        self,
        *,
        run_id: str,
        status: Literal["completed", "failed"],
        response_json: dict[str, Any] | None,
        structured_output_json: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE {self._table('ingestion_runs')}
                SET status = ?, response_json = ?, structured_output_json = ?, error = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    _to_json_text(response_json) if response_json is not None else None,
                    _to_json_text(structured_output_json) if structured_output_json is not None else None,
                    error,
                    run_id,
                ),
            )
            conn.commit()

    def apply_photo_submission(
        self,
        *,
        user_id: str,
        image_path: str,
        payload: PhotoSubmissionStructuredOutput,
        raw_structured_output: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        with self._conn() as conn:
            cursor = conn.cursor()
            self._ensure_user(cursor, user_id)
            profile_debug = self._apply_profile_updates(cursor, user_id, payload.profile_updates)
            body_metric_count = self._insert_body_metrics(cursor, user_id, payload.body_metric_events_insert)
            meal_event_id = self._upsert_meal_event(
                cursor,
                user_id=user_id,
                image_path=image_path,
                meal=payload.meal_upsert,
                macro_events=payload.macro_events_insert,
                raw_structured_output=raw_structured_output,
            )
            conn.commit()
        return {
            "idempotency_key": idempotency_key,
            "meal_event_id": meal_event_id,
            "body_metric_count": body_metric_count,
            "profile_update_count": len(payload.profile_updates),
            "meal_items_detected": len(payload.meal_items_upsert),
            "profile_debug": profile_debug,
        }

    def apply_text_turn_submission(
        self,
        *,
        user_id: str,
        payload: TextTurnStructuredOutput,
        raw_structured_output: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        with self._conn() as conn:
            cursor = conn.cursor()
            self._ensure_user(cursor, user_id)
            profile_debug = self._apply_profile_updates(cursor, user_id, payload.profile_updates)
            body_metric_count = self._insert_body_metrics(cursor, user_id, payload.body_metric_events_insert)
            conn.commit()
        return {
            "idempotency_key": idempotency_key,
            "profile_update_count": len(payload.profile_updates),
            "body_metric_count": body_metric_count,
            "structured_output_keys": sorted(raw_structured_output.keys()),
            "profile_debug": profile_debug,
        }

    def load_thread_state(self, *, user_id: str, session_key: str, agent_name: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                f"""
                SELECT TOP 1 session_json
                FROM {self._table('agent_session_memory')}
                WHERE user_id = ? AND session_key = ? AND agent_name = ?
                """,
                (user_id, session_key, agent_name),
            ).fetchone()
        if row is None:
            return None
        loaded = _safe_json_loads(str(row[0]))
        return _normalize_thread_state(loaded)

    def upsert_thread_state(
        self,
        *,
        user_id: str,
        session_key: str,
        agent_name: str,
        session_state: dict[str, Any],
        summary_text: str | None,
    ) -> None:
        memory_id = hashlib.sha256(f"{user_id}:{session_key}:{agent_name}".encode("utf-8")).hexdigest()[:64]
        now = utc_now_iso()
        with self._conn() as conn:
            cursor = conn.cursor()
            self._ensure_user(cursor, user_id)
            existing = cursor.execute(
                f"""
                SELECT TOP 1 memory_id
                FROM {self._table('agent_session_memory')}
                WHERE user_id = ? AND session_key = ? AND agent_name = ?
                """,
                (user_id, session_key, agent_name),
            ).fetchone()
            if existing:
                cursor.execute(
                    f"""
                    UPDATE {self._table('agent_session_memory')}
                    SET session_json = ?, summary_text = ?, last_event_at = ?
                    WHERE memory_id = ?
                    """,
                    (_to_json_text(session_state), summary_text, now, str(existing[0])),
                )
            else:
                cursor.execute(
                    f"""
                    INSERT INTO {self._table('agent_session_memory')} (
                        memory_id, user_id, session_key, agent_name, session_json,
                        summary_text, last_event_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (memory_id, user_id, session_key, agent_name, _to_json_text(session_state), summary_text, now, now),
                )
            conn.commit()


class DatabaseContextProvider(ContextProvider):
    def __init__(self, repository: FitnessMemoryRepository, user_id: str, *, metric_limit: int = 5, meal_limit: int = 5) -> None:
        self.repository = repository
        self.user_id = user_id
        self.metric_limit = metric_limit
        self.meal_limit = meal_limit

    async def invoking(self, messages: Any, **kwargs: Any) -> Context:
        model = self.repository.get_read_model(self.user_id, metric_limit=self.metric_limit, meal_limit=self.meal_limit)
        profile = model.get("profile", {})
        metrics = model.get("recent_body_metrics", [])
        meals = model.get("recent_meals", [])
        if not profile and not metrics and not meals:
            return Context()

        # Strip the large raw LLM blob – it is internal structured output, not useful context for the model.
        _STRIP_MEAL_KEYS = {"llm_structured_output_json", "source_image_uri", "source_hash"}
        meals_slim = [{k: v for k, v in m.items() if k not in _STRIP_MEAL_KEYS} for m in meals]

        profile_text = json.dumps(profile, ensure_ascii=False)
        metrics_text = json.dumps(metrics, ensure_ascii=False)
        meals_text = json.dumps(meals_slim, ensure_ascii=False)
        instructions = (
            f"{self.DEFAULT_CONTEXT_PROMPT}\n"
            "Use this durable fitness memory context when responding.\n"
            f"User profile: {profile_text}\n"
            f"Recent body metrics: {metrics_text}\n"
            f"Recent meals/macros: {meals_text}\n"
            "If the user asks health or diet questions, ground your answer in these tracked values."
        )
        return Context(instructions=instructions)


def extract_idempotency_key(payload: PhotoSubmissionStructuredOutput, image_bytes: bytes, user_id: str) -> str:
    op_key = next((op.idempotency_key for op in payload.persistence_ops if op.idempotency_key), None)
    if op_key:
        return op_key
    return hashlib.sha256(image_bytes + user_id.encode("utf-8")).hexdigest()


def get_fitness_repository(db_path: str | Path | None = None, *, settings: Settings | None = None) -> FitnessMemoryRepository:
    active_settings = settings or Settings()
    backend = active_settings.fitness_db_backend
    if backend == "sqlite":
        path = Path(db_path) if db_path is not None else active_settings.db_path
        return SQLiteFitnessMemoryRepository(path)
    if backend in {"azuresql", "azure_sql", "azure-sql"}:
        return AzureSqlFitnessMemoryRepository(active_settings)
    raise ValueError(f"Unsupported FITNESS_DB_BACKEND={backend}")
