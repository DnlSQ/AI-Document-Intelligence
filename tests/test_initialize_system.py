"""
Tests for src.main.initialize_system (RAG v4.4): starting the
system from whatever is already persisted, instead of always
reprocessing every PDF from scratch.

External dependencies (chunk_store, ingestion) are mocked so
these tests exercise only the startup decision logic - not real
PDF parsing, embedding, or SQLite I/O, which are already covered
by their own modules' test suites.
"""
from src.main import initialize_system


def test_initialize_system_reuses_persisted_data_when_available(monkeypatch):
    """
    If the chunk store already has data, initialize_system must
    load and return it directly - and must NOT re-run ingestion
    for any document, since that would defeat the whole point of
    persisting in the first place.
    """
    persisted_chunks = [
        {"chunk_id": 1, "page": 1, "text": "already stored", "source": "a.pdf"}
    ]

    ingestion_calls = []

    monkeypatch.setattr("src.main.load_all_chunks", lambda db_path: persisted_chunks)
    monkeypatch.setattr(
        "src.main.add_or_replace_document",
        lambda pdf_path, db_path: ingestion_calls.append(pdf_path)
    )
    monkeypatch.setattr(
        "src.main.replace_document_vectors",
        lambda chunks, source, collection=None: ingestion_calls.append(source)
    )

    chunks, collection = initialize_system(collection="fake-collection")

    assert chunks == persisted_chunks
    assert ingestion_calls == []
    assert collection == "fake-collection"


def test_initialize_system_ingests_every_document_when_store_is_empty(monkeypatch):
    """
    First run ever (nothing persisted yet): every configured PDF
    must go through the full add_or_replace_document +
    replace_document_vectors pipeline, and the combined chunks
    from all documents must be returned.
    """
    chunks_by_path = {
        "a.pdf": [{"chunk_id": 1, "page": 1, "text": "A", "source": "a.pdf"}],
        "b.pdf": [{"chunk_id": 2, "page": 1, "text": "B", "source": "b.pdf"}],
    }
    vector_sync_calls = []

    monkeypatch.setattr("src.main.load_all_chunks", lambda db_path: [])
    monkeypatch.setattr(
        "src.main.add_or_replace_document",
        lambda pdf_path, db_path: chunks_by_path[pdf_path]
    )
    monkeypatch.setattr(
        "src.main.replace_document_vectors",
        lambda chunks, source, collection=None: vector_sync_calls.append((source, chunks, collection))
    )

    chunks, collection = initialize_system(
        document_paths=["a.pdf", "b.pdf"],
        collection="fake-collection"
    )

    assert chunks == chunks_by_path["a.pdf"] + chunks_by_path["b.pdf"]
    assert [call[0] for call in vector_sync_calls] == ["a.pdf", "b.pdf"]
    assert vector_sync_calls[0][1] == chunks_by_path["a.pdf"]
    assert all(call[2] == "fake-collection" for call in vector_sync_calls)


def test_initialize_system_uses_default_collection_when_none_given(monkeypatch):
    monkeypatch.setattr("src.main.load_all_chunks", lambda db_path: [])
    monkeypatch.setattr("src.main.get_collection", lambda: "default-collection")
    monkeypatch.setattr(
        "src.main.add_or_replace_document",
        lambda pdf_path, db_path: []
    )

    captured_collections = []
    monkeypatch.setattr(
        "src.main.replace_document_vectors",
        lambda chunks, source, collection=None: captured_collections.append(collection)
    )

    chunks, collection = initialize_system(document_paths=["a.pdf"])

    assert collection == "default-collection"
    assert captured_collections == ["default-collection"]


def test_initialize_system_uses_configured_defaults_when_not_overridden(monkeypatch):
    """
    Verify initialize_system defaults document_paths/db_path to
    the configured values when not explicitly passed in - same
    convention as build_chunk_repository's own default handling.
    """
    from src.config import DOCUMENT_PATHS, CHUNK_DB_PATH

    captured_db_paths = []

    monkeypatch.setattr(
        "src.main.load_all_chunks",
        lambda db_path: captured_db_paths.append(db_path) or []
    )
    monkeypatch.setattr("src.main.get_collection", lambda: "default-collection")

    captured_paths = []
    monkeypatch.setattr(
        "src.main.add_or_replace_document",
        lambda pdf_path, db_path: captured_paths.append(pdf_path) or []
    )
    monkeypatch.setattr(
        "src.main.replace_document_vectors",
        lambda chunks, source, collection=None: None
    )

    initialize_system()

    assert captured_db_paths == [CHUNK_DB_PATH]
    assert captured_paths == DOCUMENT_PATHS
    