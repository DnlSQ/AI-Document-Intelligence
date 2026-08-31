"""
Question/answer history persistence layer (RAG v7.3.1).

Responsible only for:
    - Storing each question/answer exchange in SQLite
    - Loading recent history back, most recent first

Must not contain:
    - Retrieval, scoring, or ranking logic
    - LLM or prompt logic
    - Flask routes or template rendering

Mirrors chunk_store.py's pattern exactly: each function opens a
short-lived connection, does its work, and closes it - no
connection is held open across calls, which keeps this module safe
to use from a multi-threaded Flask app without any extra locking
code (SQLite itself serializes access to the file).
"""
import sqlite3
from datetime import datetime, timezone

from src.config import QA_HISTORY_DB_PATH


def _get_connection(db_path):
    """
    Open a new connection to the Q&A history database, creating the
    schema first if it doesn't exist yet.

    Called by every public function in this module so that no
    function depends on some other function having been called
    first - the schema is always guaranteed to exist.
    """
    connection = sqlite3.connect(db_path)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            asked_at TEXT NOT NULL
        )
        """
    )

    return connection


def init_db(db_path=QA_HISTORY_DB_PATH):
    """
    Explicitly create the Q&A history database and its schema.

    Not strictly required before calling the other functions (they
    all self-heal via _get_connection), but useful as an explicit
    startup step or for tests that want to assert the file exists.
    """
    connection = _get_connection(db_path)
    connection.close()


def save_qa_pair(question, answer, db_path=QA_HISTORY_DB_PATH, asked_at=None):
    """
    Persist one question/answer exchange.

    Args:
        question: The user's question, exactly as asked.
        answer: The grounded answer generated for it (or the
            no-context fallback string - both are legitimate
            history entries; distinguishing them is left to
            whoever reads the history back).
        db_path: Path to the SQLite database file.
        asked_at: ISO-8601 timestamp string for when the question
            was asked. Defaults to the current UTC time - exposed
            as a parameter so tests can pass a fixed value instead
            of depending on wall-clock time.

    Returns:
        The new entry's id.
    """
    if asked_at is None:
        asked_at = datetime.now(timezone.utc).isoformat()

    connection = _get_connection(db_path)

    cursor = connection.execute(
        "INSERT INTO qa_history (question, answer, asked_at) VALUES (?, ?, ?)",
        (question, answer, asked_at)
    )

    connection.commit()
    new_id = cursor.lastrowid
    connection.close()

    return new_id


def load_history(db_path=QA_HISTORY_DB_PATH, limit=None):
    """
    Load past question/answer exchanges, most recent first.

    Args:
        db_path: Path to the SQLite database file.
        limit: Maximum number of entries to return. None (default)
            returns everything.

    Returns:
        List of dicts: {"id", "question", "answer", "asked_at"},
        ordered by id descending (most recently saved first).
    """
    connection = _get_connection(db_path)

    query = "SELECT id, question, answer, asked_at FROM qa_history ORDER BY id DESC"
    params = ()

    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    rows = connection.execute(query, params).fetchall()

    connection.close()

    return [
        {"id": row[0], "question": row[1], "answer": row[2], "asked_at": row[3]}
        for row in rows
    ]
