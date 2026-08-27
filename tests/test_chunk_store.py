"""
Tests for src.chunk_store: SQLite persistence of chunk metadata.

Every test uses an isolated, temporary database file (via pytest's
tmp_path fixture) so these tests never touch the real
data/chunk_store.db used by the running application.
"""
from src.chunk_store import (
    init_db,
    save_chunks,
    load_all_chunks,
    delete_chunks_by_source,
    get_next_chunk_id,
)


def make_db_path(tmp_path):
    return str(tmp_path / "test_chunks.db")


# ---------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------

def test_init_db_is_idempotent(tmp_path):
    db_path = make_db_path(tmp_path)

    init_db(db_path)
    init_db(db_path)  # must not raise on a second call

    assert load_all_chunks(db_path) == []


# ---------------------------------------------------------------
# SAVE / LOAD ROUNDTRIP
# ---------------------------------------------------------------

def test_load_all_chunks_returns_empty_list_when_db_is_empty(tmp_path):
    db_path = make_db_path(tmp_path)

    assert load_all_chunks(db_path) == []


def test_save_and_load_chunks_roundtrip(tmp_path):
    db_path = make_db_path(tmp_path)

    chunks = [
        {"chunk_id": 1, "page": 1, "text": "First chunk.", "source": "doc.pdf"},
        {"chunk_id": 2, "page": 1, "text": "Second chunk.", "source": "doc.pdf"},
    ]

    save_chunks(chunks, db_path)

    loaded = load_all_chunks(db_path)

    assert loaded == chunks


def test_chunks_are_loaded_in_chunk_id_order(tmp_path):
    db_path = make_db_path(tmp_path)

    # Deliberately saved out of order.
    chunks = [
        {"chunk_id": 3, "page": 2, "text": "Third.", "source": "doc.pdf"},
        {"chunk_id": 1, "page": 1, "text": "First.", "source": "doc.pdf"},
        {"chunk_id": 2, "page": 1, "text": "Second.", "source": "doc.pdf"},
    ]

    save_chunks(chunks, db_path)

    loaded = load_all_chunks(db_path)

    assert [chunk["chunk_id"] for chunk in loaded] == [1, 2, 3]


# ---------------------------------------------------------------
# NEXT CHUNK ID
# ---------------------------------------------------------------

def test_get_next_chunk_id_starts_at_one_when_empty(tmp_path):
    db_path = make_db_path(tmp_path)

    assert get_next_chunk_id(db_path) == 1


def test_get_next_chunk_id_continues_from_max_existing(tmp_path):
    db_path = make_db_path(tmp_path)

    save_chunks(
        [
            {"chunk_id": 1, "page": 1, "text": "First.", "source": "doc.pdf"},
            {"chunk_id": 5, "page": 2, "text": "Fifth.", "source": "doc.pdf"},
        ],
        db_path
    )

    assert get_next_chunk_id(db_path) == 6


# ---------------------------------------------------------------
# REPLACE-BY-SOURCE (DELETE)
# ---------------------------------------------------------------

def test_delete_chunks_by_source_removes_only_that_document(tmp_path):
    db_path = make_db_path(tmp_path)

    save_chunks(
        [
            {"chunk_id": 1, "page": 1, "text": "A1.", "source": "a.pdf"},
            {"chunk_id": 2, "page": 2, "text": "A2.", "source": "a.pdf"},
            {"chunk_id": 3, "page": 1, "text": "B1.", "source": "b.pdf"},
        ],
        db_path
    )

    delete_chunks_by_source("a.pdf", db_path)

    remaining = load_all_chunks(db_path)

    assert [chunk["source"] for chunk in remaining] == ["b.pdf"]


def test_delete_chunks_by_source_does_nothing_when_source_not_found(tmp_path):
    db_path = make_db_path(tmp_path)

    save_chunks(
        [{"chunk_id": 1, "page": 1, "text": "A1.", "source": "a.pdf"}],
        db_path
    )

    delete_chunks_by_source("nonexistent.pdf", db_path)

    assert len(load_all_chunks(db_path)) == 1


# ---------------------------------------------------------------
# MULTI-DOCUMENT GROWTH (simulates a growing library)
# ---------------------------------------------------------------

def test_saving_a_second_document_does_not_affect_the_first(tmp_path):
    db_path = make_db_path(tmp_path)

    save_chunks(
        [{"chunk_id": 1, "page": 1, "text": "A1.", "source": "a.pdf"}],
        db_path
    )

    next_id = get_next_chunk_id(db_path)
    save_chunks(
        [{"chunk_id": next_id, "page": 1, "text": "B1.", "source": "b.pdf"}],
        db_path
    )

    loaded = load_all_chunks(db_path)

    assert len(loaded) == 2
    assert loaded[0]["source"] == "a.pdf"
    assert loaded[1]["source"] == "b.pdf"
    assert loaded[0]["chunk_id"] != loaded[1]["chunk_id"]
    