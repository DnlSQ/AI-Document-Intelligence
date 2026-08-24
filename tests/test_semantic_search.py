"""
Tests for the V3.3 Semantic Search module (src/semantic_search.py).

Uses a real ChromaDB EphemeralClient (cosine space) with small,
hand-picked vectors instead of real embeddings - same reasoning
as test_vector_store.py: a real in-memory Chroma collection is
cheap, so exercising real query() behavior is better than
mocking it. The embedding MODEL itself is mocked (via
generate_embedding), same pattern as test_embeddings.py, so the
suite doesn't need the real model downloaded.
"""
import uuid

import chromadb

from src.semantic_search import semantic_search
from src.vector_store import get_collection, add_chunks_to_store


def make_test_collection():
    client = chromadb.EphemeralClient()
    unique_name = f"test_{uuid.uuid4().hex}"
    return get_collection(client=client, name=unique_name)


def make_embedded_chunk(chunk_id, embedding, text="sample text", page=1, source="sample.pdf"):
    return {
        "chunk_id": chunk_id,
        "page": page,
        "text": text,
        "source": source,
        "embedding": embedding,
    }


def test_semantic_search_returns_most_similar_chunk_first(monkeypatch):
    collection = make_test_collection()

    # Two very different directions, so similarity is unambiguous.
    add_chunks_to_store([
        make_embedded_chunk(1, [1.0, 0.0, 0.0], text="closely related chunk"),
        make_embedded_chunk(2, [0.0, 1.0, 0.0], text="unrelated chunk"),
    ], collection=collection)

    monkeypatch.setattr(
        "src.semantic_search.generate_embedding",
        lambda question: [1.0, 0.0, 0.0]
    )

    results = semantic_search(
        "irrelevant text, embedding is mocked",
        top_k=2,
        collection=collection
    )

    assert len(results) == 2
    assert results[0]["chunk"]["chunk_id"] == 1
    assert results[0]["score"] > results[1]["score"]


def test_semantic_search_respects_top_k(monkeypatch):
    collection = make_test_collection()

    add_chunks_to_store([
        make_embedded_chunk(1, [1.0, 0.0, 0.0]),
        make_embedded_chunk(2, [0.9, 0.1, 0.0]),
        make_embedded_chunk(3, [0.0, 1.0, 0.0]),
    ], collection=collection)

    monkeypatch.setattr(
        "src.semantic_search.generate_embedding",
        lambda question: [1.0, 0.0, 0.0]
    )

    results = semantic_search("q", top_k=1, collection=collection)

    assert len(results) == 1
    assert results[0]["chunk"]["chunk_id"] == 1


def test_semantic_search_with_empty_store_returns_empty_list(monkeypatch):
    collection = make_test_collection()

    monkeypatch.setattr(
        "src.semantic_search.generate_embedding",
        lambda question: [1.0, 0.0, 0.0]
    )

    results = semantic_search("q", top_k=3, collection=collection)

    assert results == []


def test_semantic_search_result_shape_matches_chunk_fields(monkeypatch):
    collection = make_test_collection()

    add_chunks_to_store([
        make_embedded_chunk(
            5, [1.0, 0.0, 0.0], text="VCEO -50 V", page=3, source="sample.pdf"
        ),
    ], collection=collection)

    monkeypatch.setattr(
        "src.semantic_search.generate_embedding",
        lambda question: [1.0, 0.0, 0.0]
    )

    results = semantic_search("q", top_k=1, collection=collection)

    chunk = results[0]["chunk"]
    assert chunk == {
        "chunk_id": 5,
        "page": 3,
        "text": "VCEO -50 V",
        "source": "sample.pdf"
    }
    assert 0.0 <= results[0]["confidence"] <= 1.0


def test_semantic_search_top_k_larger_than_store_size(monkeypatch):
    """
    Asking for more chunks than exist shouldn't error - just
    return what's available.
    """
    collection = make_test_collection()

    add_chunks_to_store([
        make_embedded_chunk(1, [1.0, 0.0, 0.0]),
    ], collection=collection)

    monkeypatch.setattr(
        "src.semantic_search.generate_embedding",
        lambda question: [1.0, 0.0, 0.0]
    )

    results = semantic_search("q", top_k=10, collection=collection)

    assert len(results) == 1
    