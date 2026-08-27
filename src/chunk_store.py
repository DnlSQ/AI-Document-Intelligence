"""
Chunk metadata persistence layer.

Responsible only for:
    - Storing chunk metadata (chunk_id, page, text, source) in SQLite
    - Loading all stored chunks back into memory
    - Deleting all chunks belonging to a given document (by source)
    - Computing the next available chunk_id

Must not contain:
    - Retrieval, scoring, or ranking logic
    - LLM or prompt logic
    - PDF extraction, cleaning, or chunking logic

Each function opens a short-lived connection, does its work, and
closes it - no connection is held open across calls. This keeps the
module safe to use from a multi-threaded Flask app without any
extra locking code: SQLite itself serializes access to the file.
"""
import sqlite3

from src.config import CHUNK_DB_PATH


def _get_connection(db_path):
    """
    Open a new connection to the chunk database, creating the
    schema first if it doesn't exist yet.

    Called by every public function in this module so that no
    function depends on some other function having been called
    first - the schema is always guaranteed to exist.
    """
    connection = sqlite3.connect(db_path)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY,
            page INTEGER NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)"
    )

    return connection


def init_db(db_path=CHUNK_DB_PATH):
    """
    Explicitly create the chunk database and its schema.

    Not strictly required before calling the other functions (they
    all self-heal via _get_connection), but useful as an explicit
    startup step or for tests that want to assert the file exists.
    """
    connection = _get_connection(db_path)
    connection.close()


def save_chunks(chunks, db_path=CHUNK_DB_PATH):
    """
    Persist a list of chunks to the database.

    Args:
        chunks: List of chunk dicts, each with
            {"chunk_id", "page", "text", "source"}.
        db_path: Path to the SQLite database file.
    """
    connection = _get_connection(db_path)

    connection.executemany(
        """
        INSERT INTO chunks (chunk_id, page, text, source)
        VALUES (?, ?, ?, ?)
        """,
        [
            (chunk["chunk_id"], chunk["page"], chunk["text"], chunk["source"])
            for chunk in chunks
        ]
    )

    connection.commit()
    connection.close()


def load_all_chunks(db_path=CHUNK_DB_PATH):
    """
    Load every stored chunk, ordered by chunk_id.

    Returns:
        List of chunk dicts, in the same shape produced by
        src.chunker.create_document_chunks.
    """
    connection = _get_connection(db_path)

    rows = connection.execute(
        "SELECT chunk_id, page, text, source FROM chunks ORDER BY chunk_id"
    ).fetchall()

    connection.close()

    return [
        {"chunk_id": row[0], "page": row[1], "text": row[2], "source": row[3]}
        for row in rows
    ]


def delete_chunks_by_source(source, db_path=CHUNK_DB_PATH):
    """
    Delete every chunk belonging to a given document.

    Used when a document is replaced by a newer upload with the
    same filename: its outdated chunks must be removed before the
    new ones are saved, so stale and current information never
    coexist under the same source.

    Args:
        source: The document identifier (e.g. its filename) whose
            chunks should be removed.
    """
    connection = _get_connection(db_path)

    connection.execute(
        "DELETE FROM chunks WHERE source = ?",
        (source,)
    )

    connection.commit()
    connection.close()


def get_next_chunk_id(db_path=CHUNK_DB_PATH):
    """
    Compute the next available chunk_id.

    Chunk IDs must keep increasing across the whole database, never
    restart or renumber - two documents' chunks must never collide,
    and replacing one document must never change the chunk_ids of
    any other document already stored.

    Returns:
        1 if the database is empty, otherwise the current maximum
        chunk_id + 1.
    """
    connection = _get_connection(db_path)

    result = connection.execute(
        "SELECT MAX(chunk_id) FROM chunks"
    ).fetchone()

    connection.close()

    max_chunk_id = result[0]

    return 1 if max_chunk_id is None else max_chunk_id + 1
