"""
Tests for the V3.2 Vector Store module (src/vector_store.py).

Uses a real ChromaDB EphemeralClient (in-memory, no disk, no
network) instead of a mock: unlike Ollama or the embedding model,
an in-memory Chroma collection is cheap enough to use for real in
tests, so these exercise actual Chroma behavior rather than a
guess at it.
"""
import chromadb
import uuid

from src.vector_store import (
    get_collection,
    add_chunks_to_store,
    get_chunk_count,
    reset_store,
    delete_chunks_by_source,
)


def make_test_collection():
    """
    Fresh in-memory collection, isolated per test. A unique
    collection name (not just a new EphemeralClient) is what
    actually guarantees isolation - see the note in
    vector_store.get_collection.
    """
    client = chromadb.EphemeralClient()
    unique_name = f"test_{uuid.uuid4().hex}"
    return get_collection(client=client, name=unique_name)


def make_embedded_chunk(chunk_id, text="sample text", page=1, source="sample.pdf"):
    return {
        "chunk_id": chunk_id,
        "page": page,
        "text": text,
        "source": source,
        "embedding": [float(chunk_id), 1.0, 0.0, 0.0],
    }


def test_add_chunks_to_store_returns_count():
    collection = make_test_collection()
    chunks = [make_embedded_chunk(1), make_embedded_chunk(2)]

    stored = add_chunks_to_store(chunks, collection=collection)

    assert stored == 2
    assert get_chunk_count(collection=collection) == 2


def test_add_chunks_to_store_with_empty_list_returns_zero():
    collection = make_test_collection()

    stored = add_chunks_to_store([], collection=collection)

    assert stored == 0
    assert get_chunk_count(collection=collection) == 0


def test_add_chunks_to_store_upserts_without_duplicate_error():
    """
    Re-ingesting the same document (e.g. after restarting the
    app) must not crash on duplicate IDs - it should just
    overwrite the existing entry.
    """
    collection = make_test_collection()
    chunk = make_embedded_chunk(1, text="original text")

    add_chunks_to_store([chunk], collection=collection)

    updated_chunk = make_embedded_chunk(1, text="updated text")
    add_chunks_to_store([updated_chunk], collection=collection)

    assert get_chunk_count(collection=collection) == 1

    result = collection.get(ids=["1"])
    assert result["documents"][0] == "updated text"


def test_stored_metadata_preserves_chunk_fields():
    collection = make_test_collection()
    chunk = make_embedded_chunk(5, text="VCEO -50 V", page=3, source="sample.pdf")

    add_chunks_to_store([chunk], collection=collection)

    result = collection.get(ids=["5"])

    assert result["documents"][0] == "VCEO -50 V"
    assert result["metadatas"][0]["page"] == 3
    assert result["metadatas"][0]["source"] == "sample.pdf"
    assert result["metadatas"][0]["chunk_id"] == 5


def test_reset_store_removes_all_chunks():
    collection = make_test_collection()
    chunks = [make_embedded_chunk(1), make_embedded_chunk(2)]
    add_chunks_to_store(chunks, collection=collection)

    reset_store(collection=collection)

    assert get_chunk_count(collection=collection) == 0


def test_reset_store_on_empty_collection_does_not_error():
    collection = make_test_collection()

    reset_store(collection=collection)

    assert get_chunk_count(collection=collection) == 0


def test_delete_chunks_by_source_removes_only_matching_chunks():
    collection = make_test_collection()
    add_chunks_to_store(
        [
            make_embedded_chunk(1, source="a.pdf"),
            make_embedded_chunk(2, source="a.pdf"),
            make_embedded_chunk(3, source="b.pdf"),
        ],
        collection=collection
    )

    delete_chunks_by_source("a.pdf", collection=collection)

    assert get_chunk_count(collection=collection) == 1
    remaining = collection.get()
    assert remaining["metadatas"][0]["source"] == "b.pdf"


def test_delete_chunks_by_source_on_empty_collection_does_not_error():
    collection = make_test_collection()

    delete_chunks_by_source("a.pdf", collection=collection)

    assert get_chunk_count(collection=collection) == 0


def test_delete_chunks_by_source_when_source_not_found_leaves_others_untouched():
    collection = make_test_collection()
    add_chunks_to_store([make_embedded_chunk(1, source="a.pdf")], collection=collection)

    delete_chunks_by_source("nonexistent.pdf", collection=collection)

    assert get_chunk_count(collection=collection) == 1
    