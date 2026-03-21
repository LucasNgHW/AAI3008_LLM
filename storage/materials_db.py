"""
storage/materials_db.py
-----------------------
Minimal SQLite storage for uploaded course materials.

This stores the original PDF bytes and a few metadata fields.
Qdrant will still be used separately for embeddings and retrieval.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("./data/materials.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    content BLOB NOT NULL,
    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def store_material(
    filename: str,
    content: bytes,
    mime_type: str = "application/pdf",
) -> int:
    """Insert one course material and return its database id."""
    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO materials (filename, mime_type, content)
            VALUES (?, ?, ?)
            """,
            (filename, mime_type, content),
        )
        return int(cursor.lastrowid)


def list_materials() -> list[dict]:
    """Return stored materials without the raw PDF bytes."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, mime_type, uploaded_at
            FROM materials
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_material(material_id: int) -> dict | None:
    """Return one stored material including its raw bytes."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, filename, mime_type, content, uploaded_at
            FROM materials
            WHERE id = ?
            """,
            (material_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_material(material_id: int) -> bool:
    """Delete one stored material from SQLite."""
    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            """
            DELETE FROM materials
            WHERE id = ?
            """,
            (material_id,),
        )
    return cursor.rowcount > 0


def delete_all_materials() -> int:
    """Delete all stored materials from SQLite."""
    init_db()
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM materials")
    return cursor.rowcount


def count_materials() -> int:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM materials").fetchone()
    return int(row["count"])
