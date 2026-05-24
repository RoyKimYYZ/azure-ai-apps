"""Prompt Library — user-managed reusable prompt templates.

A small CRUD layer for storing prompts that the user can pick from a sidebar
selector in the Streamlit chatbot. Mirrors the dual-backend (SQLite / Azure SQL)
pattern used by ``fitness_memory.py``.

Schema is intentionally extensible: new ad-hoc fields can land in the
``extra_json`` column without a migration.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from config import get_config, resolve_env

logger = logging.getLogger(__name__)

SQL_COPT_SS_ACCESS_TOKEN = 1256
AZURE_SQL_TOKEN_SCOPE = "https://database.windows.net/.default"
_TOKEN_REFRESH_SKEW_SECONDS = 300

VISIBILITY_PRIVATE = "private"
VISIBILITY_SHARED = "shared"
VISIBILITY_GLOBAL = "global"
ALLOWED_VISIBILITIES = frozenset({VISIBILITY_PRIVATE, VISIBILITY_SHARED, VISIBILITY_GLOBAL})

_REPO_CACHE: dict[tuple[Any, ...], Any] = {}
_REPO_CACHE_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _new_prompt_id() -> str:
    return uuid4().hex


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        value = parsed
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalize_visibility(value: str | None) -> str:
    v = (value or VISIBILITY_PRIVATE).strip().lower()
    if v not in ALLOWED_VISIBILITIES:
        return VISIBILITY_PRIVATE
    return v


@dataclass
class PromptRecord:
    prompt_id: str
    user_id: str
    title: str
    body: str
    description: str | None = None
    agent_names: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    visibility: str = VISIBILITY_PRIVATE
    is_active: bool = True
    sort_order: int = 0
    usage_count: int = 0
    last_used_at: str | None = None
    created_at: str = ""
    updated_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def is_assigned_to_agent(self, agent_name: str) -> bool:
        if not self.agent_names:
            return True  # empty list == any agent
        return agent_name in self.agent_names

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "title": self.title,
            "body": self.body,
            "description": self.description or "",
            "agent_names": list(self.agent_names),
            "tags": list(self.tags),
            "visibility": self.visibility,
            "is_active": self.is_active,
            "sort_order": self.sort_order,
            "usage_count": self.usage_count,
            "last_used_at": self.last_used_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class PromptLibraryRepository(Protocol):
    def ensure_schema(self) -> None: ...
    def list_visible(self, user_id: str, *, agent_name: str | None = None, include_inactive: bool = False) -> list[PromptRecord]: ...
    def list_owned(self, user_id: str, *, include_inactive: bool = True) -> list[PromptRecord]: ...
    def get(self, prompt_id: str) -> PromptRecord | None: ...
    def create(
        self,
        *,
        user_id: str,
        title: str,
        body: str,
        agent_names: list[str] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visibility: str = VISIBILITY_PRIVATE,
        sort_order: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> PromptRecord: ...
    def update(
        self,
        prompt_id: str,
        *,
        title: str | None = None,
        body: str | None = None,
        agent_names: list[str] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visibility: str | None = None,
        is_active: bool | None = None,
        sort_order: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> PromptRecord | None: ...
    def delete(self, prompt_id: str) -> bool: ...
    def mark_used(self, prompt_id: str) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# SQLite implementation
# ─────────────────────────────────────────────────────────────────────────────

_SQLITE_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS prompt_library (
    prompt_id        TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    title            TEXT NOT NULL,
    body             TEXT NOT NULL,
    description      TEXT,
    agent_names_json TEXT NOT NULL DEFAULT '[]',
    tags_json        TEXT NOT NULL DEFAULT '[]',
    visibility       TEXT NOT NULL DEFAULT 'private'
                     CHECK (visibility IN ('private', 'shared', 'global')),
    is_active        INTEGER NOT NULL DEFAULT 1,
    sort_order       INTEGER NOT NULL DEFAULT 0,
    usage_count      INTEGER NOT NULL DEFAULT 0,
    last_used_at     TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    extra_json       TEXT
);
"""

_SQLITE_INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_prompt_library_user        ON prompt_library(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_prompt_library_visibility  ON prompt_library(visibility);",
    "CREATE INDEX IF NOT EXISTS idx_prompt_library_is_active   ON prompt_library(is_active);",
]


def _row_to_record_sqlite(row: sqlite3.Row) -> PromptRecord:
    # sqlite3.Row supports membership but not .get(); SIM401 suggestion does not apply.
    extra_raw = row["extra_json"] if "extra_json" in row else None  # noqa: SIM401
    try:
        extra = json.loads(extra_raw) if extra_raw else {}
        if not isinstance(extra, dict):
            extra = {}
    except json.JSONDecodeError:
        extra = {}
    return PromptRecord(
        prompt_id=str(row["prompt_id"]),
        user_id=str(row["user_id"]),
        title=str(row["title"]),
        body=str(row["body"]),
        description=row["description"],
        agent_names=_coerce_str_list(row["agent_names_json"]),
        tags=_coerce_str_list(row["tags_json"]),
        visibility=_normalize_visibility(row["visibility"]),
        is_active=bool(row["is_active"]),
        sort_order=int(row["sort_order"] or 0),
        usage_count=int(row["usage_count"] or 0),
        last_used_at=row["last_used_at"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        extra=extra,
    )


class SQLitePromptLibraryRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._schema_ready = False
        self._lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._lock:
            if self._schema_ready:
                return
            with self._conn() as conn:
                conn.executescript(_SQLITE_SCHEMA_DDL)
                for ddl in _SQLITE_INDEX_DDL:
                    conn.execute(ddl)
                conn.commit()
            self._schema_ready = True

    def list_visible(
        self,
        user_id: str,
        *,
        agent_name: str | None = None,
        include_inactive: bool = False,
    ) -> list[PromptRecord]:
        self.ensure_schema()
        clauses = ["(user_id = ? OR visibility = 'global')"]
        params: list[Any] = [user_id]
        if not include_inactive:
            clauses.append("is_active = 1")
        sql = (
            "SELECT * FROM prompt_library WHERE "
            + " AND ".join(clauses)
            + " ORDER BY sort_order ASC, title COLLATE NOCASE ASC"
        )
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        records = [_row_to_record_sqlite(r) for r in rows]
        if agent_name:
            records = [r for r in records if r.is_assigned_to_agent(agent_name)]
        return records

    def list_owned(self, user_id: str, *, include_inactive: bool = True) -> list[PromptRecord]:
        self.ensure_schema()
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if not include_inactive:
            clauses.append("is_active = 1")
        sql = (
            "SELECT * FROM prompt_library WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_at DESC"
        )
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_record_sqlite(r) for r in rows]

    def get(self, prompt_id: str) -> PromptRecord | None:
        self.ensure_schema()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_library WHERE prompt_id = ?",
                (prompt_id,),
            ).fetchone()
        return _row_to_record_sqlite(row) if row else None

    def create(
        self,
        *,
        user_id: str,
        title: str,
        body: str,
        agent_names: list[str] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visibility: str = VISIBILITY_PRIVATE,
        sort_order: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> PromptRecord:
        self.ensure_schema()
        prompt_id = _new_prompt_id()
        now = _utc_now_iso()
        agent_names_clean = _coerce_str_list(agent_names)
        tags_clean = _coerce_str_list(tags)
        visibility_clean = _normalize_visibility(visibility)
        extra_json = json.dumps(extra) if extra else None
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO prompt_library (
                    prompt_id, user_id, title, body, description,
                    agent_names_json, tags_json, visibility,
                    is_active, sort_order, usage_count, last_used_at,
                    created_at, updated_at, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 0, NULL, ?, ?, ?)
                """,
                (
                    prompt_id,
                    user_id,
                    title.strip(),
                    body,
                    (description or "").strip() or None,
                    json.dumps(agent_names_clean),
                    json.dumps(tags_clean),
                    visibility_clean,
                    int(sort_order),
                    now,
                    now,
                    extra_json,
                ),
            )
            conn.commit()
        record = self.get(prompt_id)
        if record is None:
            raise RuntimeError("Failed to read back newly created prompt")
        return record

    def update(
        self,
        prompt_id: str,
        *,
        title: str | None = None,
        body: str | None = None,
        agent_names: list[str] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visibility: str | None = None,
        is_active: bool | None = None,
        sort_order: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> PromptRecord | None:
        self.ensure_schema()
        assignments: list[str] = []
        params: list[Any] = []
        if title is not None:
            assignments.append("title = ?")
            params.append(title.strip())
        if body is not None:
            assignments.append("body = ?")
            params.append(body)
        if description is not None:
            assignments.append("description = ?")
            params.append(description.strip() or None)
        if agent_names is not None:
            assignments.append("agent_names_json = ?")
            params.append(json.dumps(_coerce_str_list(agent_names)))
        if tags is not None:
            assignments.append("tags_json = ?")
            params.append(json.dumps(_coerce_str_list(tags)))
        if visibility is not None:
            assignments.append("visibility = ?")
            params.append(_normalize_visibility(visibility))
        if is_active is not None:
            assignments.append("is_active = ?")
            params.append(1 if is_active else 0)
        if sort_order is not None:
            assignments.append("sort_order = ?")
            params.append(int(sort_order))
        if extra is not None:
            assignments.append("extra_json = ?")
            params.append(json.dumps(extra) if extra else None)

        if not assignments:
            return self.get(prompt_id)

        assignments.append("updated_at = ?")
        params.append(_utc_now_iso())
        params.append(prompt_id)
        sql = f"UPDATE prompt_library SET {', '.join(assignments)} WHERE prompt_id = ?"
        with self._conn() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            if cursor.rowcount == 0:
                return None
        return self.get(prompt_id)

    def delete(self, prompt_id: str) -> bool:
        self.ensure_schema()
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM prompt_library WHERE prompt_id = ?", (prompt_id,))
            conn.commit()
            return cursor.rowcount > 0

    def mark_used(self, prompt_id: str) -> None:
        self.ensure_schema()
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                "UPDATE prompt_library SET usage_count = usage_count + 1, last_used_at = ? WHERE prompt_id = ?",
                (now, prompt_id),
            )
            conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Azure SQL implementation
# ─────────────────────────────────────────────────────────────────────────────

_AZURE_SCHEMA_DDL = """
IF OBJECT_ID('{schema}.prompt_library', 'U') IS NULL
BEGIN
    CREATE TABLE {schema}.prompt_library (
        prompt_id         NVARCHAR(64)   NOT NULL PRIMARY KEY,
        user_id           NVARCHAR(64)   NOT NULL,
        title             NVARCHAR(200)  NOT NULL,
        body              NVARCHAR(MAX)  NOT NULL,
        description       NVARCHAR(1000) NULL,
        agent_names_json  NVARCHAR(MAX)  NOT NULL CONSTRAINT DF_prompt_library_agent_names DEFAULT N'[]',
        tags_json         NVARCHAR(MAX)  NOT NULL CONSTRAINT DF_prompt_library_tags        DEFAULT N'[]',
        visibility        NVARCHAR(16)   NOT NULL CONSTRAINT DF_prompt_library_visibility  DEFAULT N'private',
        is_active         BIT            NOT NULL CONSTRAINT DF_prompt_library_is_active   DEFAULT (1),
        sort_order        INT            NOT NULL CONSTRAINT DF_prompt_library_sort_order  DEFAULT (0),
        usage_count       INT            NOT NULL CONSTRAINT DF_prompt_library_usage_count DEFAULT (0),
        last_used_at      DATETIME2(3)   NULL,
        created_at        DATETIME2(3)   NOT NULL CONSTRAINT DF_prompt_library_created_at  DEFAULT SYSUTCDATETIME(),
        updated_at        DATETIME2(3)   NOT NULL CONSTRAINT DF_prompt_library_updated_at  DEFAULT SYSUTCDATETIME(),
        extra_json        NVARCHAR(MAX)  NULL,
        CONSTRAINT CK_prompt_library_visibility CHECK (visibility IN (N'private', N'shared', N'global'))
    );
END;
"""


def _import_azure_sql_runtime() -> tuple[Any, Any]:
    try:
        import pyodbc  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Azure SQL support requires pyodbc. Install project dependencies before using azuresql backend."
        ) from exc
    try:
        from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Azure SQL support requires azure-identity. Install project dependencies before using azuresql backend."
        ) from exc
    return pyodbc, DefaultAzureCredential


def _encode_sql_access_token(token: str) -> bytes:
    encoded = token.encode("utf-16-le")
    return struct.pack(f"<I{len(encoded)}s", len(encoded), encoded)


def _normalize_sql_server_name(server: str) -> str:
    normalized = (server or "").strip()
    if not normalized:
        raise ValueError("AZURE_SQL_SERVER must be set when using azuresql backend")
    if ".database.windows.net" not in normalized:
        return f"{normalized}.database.windows.net"
    return normalized


def _row_to_record_azuresql(columns: list[str], row: Any) -> PromptRecord:
    rec = dict(zip(columns, row, strict=False))
    extra_raw = rec.get("extra_json")
    try:
        extra = json.loads(extra_raw) if extra_raw else {}
        if not isinstance(extra, dict):
            extra = {}
    except json.JSONDecodeError:
        extra = {}

    def _iso(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    return PromptRecord(
        prompt_id=str(rec["prompt_id"]),
        user_id=str(rec["user_id"]),
        title=str(rec["title"]),
        body=str(rec["body"]),
        description=rec.get("description"),
        agent_names=_coerce_str_list(rec.get("agent_names_json")),
        tags=_coerce_str_list(rec.get("tags_json")),
        visibility=_normalize_visibility(rec.get("visibility")),
        is_active=bool(rec.get("is_active")),
        sort_order=int(rec.get("sort_order") or 0),
        usage_count=int(rec.get("usage_count") or 0),
        last_used_at=_iso(rec.get("last_used_at")) or None,
        created_at=_iso(rec.get("created_at")),
        updated_at=_iso(rec.get("updated_at")),
        extra=extra,
    )


_SELECT_COLUMNS = [
    "prompt_id", "user_id", "title", "body", "description",
    "agent_names_json", "tags_json", "visibility", "is_active",
    "sort_order", "usage_count", "last_used_at",
    "created_at", "updated_at", "extra_json",
]


class AzureSqlPromptLibraryRepository:
    def __init__(self) -> None:
        cfg = get_config()
        az_sql = cfg.database.azure_sql
        self.azure_client_id = resolve_env(cfg.azure.identity.client_id_env)
        self.server = _normalize_sql_server_name(resolve_env(az_sql.server_env))
        self.database = resolve_env(az_sql.database_env).strip()
        self.schema = (az_sql.schema_name or "dbo").strip()
        self.driver = (az_sql.driver or "ODBC Driver 18 for SQL Server").strip()
        self.auth_mode = az_sql.auth_mode
        self.admin_user = resolve_env(az_sql.admin_user_env).strip()
        self.admin_password = resolve_env(az_sql.admin_password_env)
        self.encrypt = az_sql.encrypt
        self.trust_server_certificate = az_sql.trust_server_certificate
        self.connection_timeout = az_sql.connection_timeout

        self._auth_lock = threading.RLock()
        self._credential: Any | None = None
        self._access_token = ""
        self._access_token_expires_on = 0.0
        self._schema_ready = False
        self._schema_lock = threading.Lock()

        self._connection_string = (
            f"Driver={{{self.driver}}};"
            f"Server=tcp:{self.server},1433;"
            f"Database={self.database};"
            f"Encrypt={'yes' if self.encrypt else 'no'};"
            f"TrustServerCertificate={'yes' if self.trust_server_certificate else 'no'};"
            f"Connection Timeout={self.connection_timeout};"
        )

        if not self.database:
            raise ValueError("AZURE_SQL_DATABASE must be set when using azuresql backend")

    # --- connection plumbing (mirrors fitness_memory.py) ----------------

    def _get_credential(self, DefaultAzureCredential: Any) -> Any:
        if self._credential is not None:
            return self._credential
        with self._auth_lock:
            if self._credential is None:
                self._credential = DefaultAzureCredential(
                    managed_identity_client_id=self.azure_client_id or None,
                    exclude_interactive_browser_credential=False,
                )
            return self._credential

    def _get_access_token_struct(self, DefaultAzureCredential: Any) -> bytes:
        now = time.time()
        if self._access_token and now < self._access_token_expires_on - _TOKEN_REFRESH_SKEW_SECONDS:
            return _encode_sql_access_token(self._access_token)
        with self._auth_lock:
            now = time.time()
            if self._access_token and now < self._access_token_expires_on - _TOKEN_REFRESH_SKEW_SECONDS:
                return _encode_sql_access_token(self._access_token)
            credential = self._get_credential(DefaultAzureCredential)
            access_token = credential.get_token(AZURE_SQL_TOKEN_SCOPE)
            self._access_token = access_token.token
            self._access_token_expires_on = float(access_token.expires_on)
            return _encode_sql_access_token(self._access_token)

    def _conn(self) -> Any:
        pyodbc, DefaultAzureCredential = _import_azure_sql_runtime()
        pyodbc.pooling = True
        if self.auth_mode in {"adminpassword", "sqlpassword", "sql-password"}:
            return pyodbc.connect(
                f"{self._connection_string}UID={self.admin_user};PWD={self.admin_password};",
                autocommit=False,
            )
        token_struct = self._get_access_token_struct(DefaultAzureCredential)
        return pyodbc.connect(
            self._connection_string,
            attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct},
            autocommit=False,
        )

    def _table(self) -> str:
        return f"[{self.schema}].[prompt_library]"

    # --- schema --------------------------------------------------------

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            ddl = _AZURE_SCHEMA_DDL.format(schema=self.schema)
            try:
                with self._conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute(ddl)
                    conn.commit()
            except Exception as exc:  # noqa: BLE001
                logger.warning("prompt_library schema bootstrap failed (continuing): %s", exc)
            self._schema_ready = True

    # --- queries -------------------------------------------------------

    def _select_sql(self, where: str) -> str:
        cols = ", ".join(_SELECT_COLUMNS)
        return f"SELECT {cols} FROM {self._table()} WHERE {where}"

    def list_visible(
        self,
        user_id: str,
        *,
        agent_name: str | None = None,
        include_inactive: bool = False,
    ) -> list[PromptRecord]:
        self.ensure_schema()
        clauses = ["(user_id = ? OR visibility = N'global')"]
        params: list[Any] = [user_id]
        if not include_inactive:
            clauses.append("is_active = 1")
        sql = (
            self._select_sql(" AND ".join(clauses))
            + " ORDER BY sort_order ASC, LOWER(title) ASC"
        )
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        records = [_row_to_record_azuresql(_SELECT_COLUMNS, r) for r in rows]
        if agent_name:
            records = [r for r in records if r.is_assigned_to_agent(agent_name)]
        return records

    def list_owned(self, user_id: str, *, include_inactive: bool = True) -> list[PromptRecord]:
        self.ensure_schema()
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if not include_inactive:
            clauses.append("is_active = 1")
        sql = self._select_sql(" AND ".join(clauses)) + " ORDER BY updated_at DESC"
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        return [_row_to_record_azuresql(_SELECT_COLUMNS, r) for r in rows]

    def get(self, prompt_id: str) -> PromptRecord | None:
        self.ensure_schema()
        sql = self._select_sql("prompt_id = ?")
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (prompt_id,))
            row = cursor.fetchone()
        return _row_to_record_azuresql(_SELECT_COLUMNS, row) if row else None

    def create(
        self,
        *,
        user_id: str,
        title: str,
        body: str,
        agent_names: list[str] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visibility: str = VISIBILITY_PRIVATE,
        sort_order: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> PromptRecord:
        self.ensure_schema()
        prompt_id = _new_prompt_id()
        now = _utc_now_iso()
        agent_names_clean = _coerce_str_list(agent_names)
        tags_clean = _coerce_str_list(tags)
        visibility_clean = _normalize_visibility(visibility)
        extra_json = json.dumps(extra) if extra else None

        sql = (
            f"INSERT INTO {self._table()} ("
            "prompt_id, user_id, title, body, description, "
            "agent_names_json, tags_json, visibility, is_active, sort_order, "
            "usage_count, last_used_at, created_at, updated_at, extra_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 0, NULL, ?, ?, ?)"
        )
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                sql,
                (
                    prompt_id,
                    user_id,
                    title.strip(),
                    body,
                    (description or "").strip() or None,
                    json.dumps(agent_names_clean),
                    json.dumps(tags_clean),
                    visibility_clean,
                    int(sort_order),
                    now,
                    now,
                    extra_json,
                ),
            )
            conn.commit()
        record = self.get(prompt_id)
        if record is None:
            raise RuntimeError("Failed to read back newly created prompt")
        return record

    def update(
        self,
        prompt_id: str,
        *,
        title: str | None = None,
        body: str | None = None,
        agent_names: list[str] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visibility: str | None = None,
        is_active: bool | None = None,
        sort_order: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> PromptRecord | None:
        self.ensure_schema()
        assignments: list[str] = []
        params: list[Any] = []
        if title is not None:
            assignments.append("title = ?")
            params.append(title.strip())
        if body is not None:
            assignments.append("body = ?")
            params.append(body)
        if description is not None:
            assignments.append("description = ?")
            params.append(description.strip() or None)
        if agent_names is not None:
            assignments.append("agent_names_json = ?")
            params.append(json.dumps(_coerce_str_list(agent_names)))
        if tags is not None:
            assignments.append("tags_json = ?")
            params.append(json.dumps(_coerce_str_list(tags)))
        if visibility is not None:
            assignments.append("visibility = ?")
            params.append(_normalize_visibility(visibility))
        if is_active is not None:
            assignments.append("is_active = ?")
            params.append(1 if is_active else 0)
        if sort_order is not None:
            assignments.append("sort_order = ?")
            params.append(int(sort_order))
        if extra is not None:
            assignments.append("extra_json = ?")
            params.append(json.dumps(extra) if extra else None)
        if not assignments:
            return self.get(prompt_id)

        assignments.append("updated_at = ?")
        params.append(_utc_now_iso())
        params.append(prompt_id)
        sql = f"UPDATE {self._table()} SET {', '.join(assignments)} WHERE prompt_id = ?"
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            affected = cursor.rowcount
            conn.commit()
        if affected == 0:
            return None
        return self.get(prompt_id)

    def delete(self, prompt_id: str) -> bool:
        self.ensure_schema()
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self._table()} WHERE prompt_id = ?", (prompt_id,))
            affected = cursor.rowcount
            conn.commit()
        return affected > 0

    def mark_used(self, prompt_id: str) -> None:
        self.ensure_schema()
        now = _utc_now_iso()
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {self._table()} "
                "SET usage_count = usage_count + 1, last_used_at = ? "
                "WHERE prompt_id = ?",
                (now, prompt_id),
            )
            conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def get_prompt_repository(db_path: str | Path | None = None) -> PromptLibraryRepository:
    cfg = get_config()
    az_sql = cfg.database.azure_sql
    backend = cfg.database.default_backend

    if backend == "sqlite":
        path = Path(db_path) if db_path is not None else Path(cfg.database.sqlite.path).expanduser()
        cache_key: tuple[Any, ...] = ("sqlite", str(path.expanduser().resolve()))
        with _REPO_CACHE_LOCK:
            cached = _REPO_CACHE.get(cache_key)
            if cached is None:
                cached = SQLitePromptLibraryRepository(path)
                _REPO_CACHE[cache_key] = cached
            return cached

    if backend in {"azuresql", "azure_sql", "azure-sql"}:
        cache_key = (
            "azuresql",
            resolve_env(az_sql.server_env),
            resolve_env(az_sql.database_env),
            az_sql.schema_name,
            az_sql.driver,
            az_sql.auth_mode,
            resolve_env(az_sql.admin_user_env),
            resolve_env(az_sql.admin_password_env),
            az_sql.encrypt,
            az_sql.trust_server_certificate,
            az_sql.connection_timeout,
            resolve_env(cfg.azure.identity.client_id_env),
            resolve_env(cfg.azure.identity.tenant_id_env),
        )
        with _REPO_CACHE_LOCK:
            cached = _REPO_CACHE.get(cache_key)
            if cached is None:
                cached = AzureSqlPromptLibraryRepository()
                _REPO_CACHE[cache_key] = cached
            return cached

    raise ValueError(f"Unsupported database.default_backend={backend}")
