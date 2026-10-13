from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path


class SessionStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session "
                "ON messages(session_id, id)"
            )

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    def history(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append_exchange(self, session_id: str, question: str, answer: str) -> None:
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
                [
                    (session_id, "user", question),
                    (session_id, "assistant", answer),
                ],
            )

    def sessions(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT session_id, COUNT(*) AS messages, MAX(created_at) AS updated_at
                FROM messages
                GROUP BY session_id
                ORDER BY MAX(id) DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def clear(self, session_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
        return cursor.rowcount
