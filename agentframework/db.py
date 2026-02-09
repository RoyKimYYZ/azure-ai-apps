import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).parent / "agentframework.db"


class StructuredOutputStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self._get_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS structured_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    steps_json TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def insert(self, steps: list[str], rationale: str, output_type: str) -> int:
        self.init_db()
        with self._get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO structured_outputs (steps_json, rationale, type, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (json.dumps(steps), rationale, output_type, datetime.utcnow().isoformat()),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_all(self) -> list[dict[str, Any]]:
        self.init_db()
        with self._get_connection() as connection:
            rows = connection.execute(
                "SELECT id, steps_json, rationale, type, created_at FROM structured_outputs"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "steps": json.loads(row["steps_json"]),
                "rationale": row["rationale"],
                "type": row["type"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get(self, output_id: int) -> dict[str, Any] | None:
        self.init_db()
        with self._get_connection() as connection:
            row = connection.execute(
                """
                SELECT id, steps_json, rationale, type, created_at
                FROM structured_outputs
                WHERE id = ?
                """,
                (output_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "steps": json.loads(row["steps_json"]),
            "rationale": row["rationale"],
            "type": row["type"],
            "created_at": row["created_at"],
        }

    def update(self, output_id: int, steps: list[str], rationale: str, output_type: str) -> bool:
        self.init_db()
        with self._get_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE structured_outputs
                SET steps_json = ?, rationale = ?, type = ?
                WHERE id = ?
                """,
                (json.dumps(steps), rationale, output_type, output_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def delete(self, output_id: int) -> bool:
        self.init_db()
        with self._get_connection() as connection:
            cursor = connection.execute(
                "DELETE FROM structured_outputs WHERE id = ?",
                (output_id,),
            )
            connection.commit()
            return cursor.rowcount > 0
