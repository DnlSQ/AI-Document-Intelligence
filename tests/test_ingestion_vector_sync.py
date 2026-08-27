"""
Tests for src.ingestion.replace_document_vectors: keeping the
vector store in sync with a document added or replaced in the
chunk repository (V4.3).

Uses a real, isolated in-memory ChromaDB collection (same pattern
as test_vector_store.py) so these tests exercise real delete/upsert
behavior. Only embedding generation is mocked (it depends on a
downloaded model and isn't what this function is responsible for).
"""
import chromadb
import uuid

from src.ingestion import replace_document_vectors
from src.vector_store import get_collection, add_chunks_to_store, get_chunk_count


def make_test_collection():
    client = chromadb.EphemeralClient()
    unique_name = f"test_{uuid.uuid4().hex}"
    return get_collection(client=client, name=unique_name)


def fake_chunk(chunk_id, source, text="sample text", page=1):
    return {"chunk_id": chunk_id, "page": page, "text": text, "source": source}


def patch_embeddings(monkeypatch):
    """
    Real embedding generation needs a downloaded model - not what
    this function is responsible for testing, so it's replaced
    with a deterministic stand-in.
    """
    def fake_generate_embeddings_for_chunks(chunks):
        embedded = []
        for chunk in chunks:
            embedded_chunk = dict(chunk)
            embedded_chunk["embedding"] = [float(chunk["chunk_id"]), 0.0, 0.0, 0.0]
            embedded.append(embedded_chunk)
        return embedded

    monkeypatch.setattr(
        "src.ingestion.generate_embeddings_for_chunks",
        fake_generate_embeddings_for_chunks
    )


def test_replace_document_vectors_stores_new_chunks(monkeypatch):
    collection = make_test_collection()
    patch_embeddings(monkeypatch)

    chunks = [fake_chunk(1, "a.pdf"), fake_chunk(2, "a.pdf")]
    stored = replace_document_vectors(chunks, "a.pdf", collection=collection)

    assert stored == 2
    assert get_chunk_count(collection=collection) == 2


def test_replace_document_vectors_removes_old_vectors_for_the_same_source(monkeypatch):
    collection = make_test_collection()
    patch_embeddings(monkeypatch)

    # Simulate an earlier version of a.pdf already stored under
    # OLD chunk_ids (1, 2) - a replace assigns fresh ids, so these
    # would never be overwritten by a plain upsert.
    add_chunks_to_store(
        [
            {**fake_chunk(1, "a.pdf"), "embedding": [1.0, 0.0, 0.0, 0.0]},
            {**fake_chunk(2, "a.pdf"), "embedding": [2.0, 0.0, 0.0, 0.0]},
        ],
        collection=collection
    )

    new_chunks = [fake_chunk(5, "a.pdf", text="updated content")]
    replace_document_vectors(new_chunks, "a.pdf", collection=collection)

    remaining = collection.get()
    assert remaining["ids"] == ["5"]
    assert remaining["documents"][0] == "updated content"


def test_replace_document_vectors_does_not_affect_other_documents(monkeypatch):
    collection = make_test_collection()
    patch_embeddings(monkeypatch)

    add_chunks_to_store(
        [{**fake_chunk(1, "b.pdf"), "embedding": [1.0, 0.0, 0.0, 0.0]}],
        collection=collection
    )

    replace_document_vectors([fake_chunk(2, "a.pdf")], "a.pdf", collection=collection)

    assert get_chunk_count(collection=collection) == 2
    sources = {m["source"] for m in collection.get()["metadatas"]}
    assert sources == {"a.pdf", "b.pdf"}


def test_replace_document_vectors_uses_default_collection_when_none_given(monkeypatch):
    """
    Doesn't touch the real persistent store - just checks that
    get_collection() is called to obtain one when collection=None.
    """
    patch_embeddings(monkeypatch)
    fallback_collection = make_test_collection()

    monkeypatch.setattr(
        "src.ingestion.get_collection",
        lambda: fallback_collection
    )

    replace_document_vectors([fake_chunk(1, "a.pdf")], "a.pdf")

    assert get_chunk_count(collection=fallback_collection) == 1
    