from functools import lru_cache
import os
from threading import Lock
from typing import Any, Literal

import psycopg
from langchain_core.runnables import Runnable

from app.components.retriever import create_qa_chain

_schema_lock = Lock()
_schema_ready = False


@lru_cache(maxsize=1)
def get_qa_chain() -> Runnable[Any, Any]:
    return create_qa_chain()


def _connect() -> psycopg.Connection:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL to your managed PostgreSQL connection URL")
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]

    return psycopg.connect(
        database_url,
        connect_timeout=10,
        sslmode=os.environ.get("PGSSLMODE", "require"),
    )


def initialize_chat_history() -> None:
    global _schema_ready
    if _schema_ready:
        return

    with _schema_lock:
        if _schema_ready:
            return

        with _connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS chat_messages_conversation_id_id_idx
                ON chat_messages (conversation_id, id)
                """
            )

        _schema_ready = True


def load_chat_history(conversation_id: str) -> list[dict[str, str]]:
    initialize_chat_history()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE conversation_id = %s
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def save_chat_message(
    conversation_id: str,
    role: Literal["user", "assistant"],
    content: str,
) -> None:
    initialize_chat_history()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (conversation_id, role, content)
            VALUES (%s, %s, %s)
            """,
            (conversation_id, role, content),
        )


def clear_chat_history(conversation_id: str) -> None:
    initialize_chat_history()
    with _connect() as connection:
        connection.execute(
            "DELETE FROM chat_messages WHERE conversation_id = %s",
            (conversation_id,),
        )
