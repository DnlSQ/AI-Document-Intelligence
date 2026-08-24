"""
Tests for the V3.1 Embedding Generation module (src/embeddings.py).

These tests mock the underlying sentence-embedding model at its
boundary (embeddings._get_model), the same way test_generator.py
mocks llm.ask_llm to avoid depending on a running Ollama instance.
This keeps the suite fast and runnable in any environment, without
requiring the real model to be downloaded.

A one-time, non-mocked sanity check that the real model actually
captures semantic similarity (the whole point of RAG v3) is a
manual check, not part of this automated suite - matching how the
project never asserts against the real Ollama model either.
"""
from src.embeddings import (
    generate_embedding,
    generate_embeddings_for_chunks,
)


class FakeModel:
    """
    Stand-in for a sentence-transformers model. Returns short,
    deterministic vectors based on input length so tests can
    verify behavior without loading real weights.
    """

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np

        if isinstance(texts, str):
            return np.array([float(len(texts)), 1.0, 0.0, 0.0])

        return np.array([
            [float(len(text)), 1.0, 0.0, 0.0] for text in texts
        ])


def test_generate_embedding_returns_list_of_floats(monkeypatch):
    monkeypatch.setattr(
        "src.embeddings._get_model",
        lambda: FakeModel()
    )

    text = "VCEO collector-emitter voltage"
    vector = generate_embedding(text)

    assert isinstance(vector, list)
    assert all(isinstance(value, float) for value in vector)
    assert vector == [float(len(text)), 1.0, 0.0, 0.0]


def test_generate_embedding_rejects_empty_text(monkeypatch):
    monkeypatch.setattr(
        "src.embeddings._get_model",
        lambda: FakeModel()
    )

    try:
        generate_embedding("")
        assert False, "expected ValueError for empty text"
    except ValueError:
        pass


def test_generate_embedding_rejects_whitespace_only_text(monkeypatch):
    monkeypatch.setattr(
        "src.embeddings._get_model",
        lambda: FakeModel()
    )

    try:
        generate_embedding("   ")
        assert False, "expected ValueError for whitespace-only text"
    except ValueError:
        pass


def test_generate_embeddings_for_chunks_adds_embedding_key(monkeypatch):
    monkeypatch.setattr(
        "src.embeddings._get_model",
        lambda: FakeModel()
    )

    chunks = [
        {
            "chunk_id": 1,
            "page": 2,
            "text": "VCEO collector-emitter voltage -50 V",
            "source": "data/documents/sample.pdf"
        },
        {
            "chunk_id": 2,
            "page": 3,
            "text": "IO output current -500 mA",
            "source": "data/documents/sample.pdf"
        },
    ]

    embedded_chunks = generate_embeddings_for_chunks(chunks)

    assert len(embedded_chunks) == 2

    for original, embedded in zip(chunks, embedded_chunks):
        assert embedded["chunk_id"] == original["chunk_id"]
        assert embedded["page"] == original["page"]
        assert embedded["text"] == original["text"]
        assert embedded["source"] == original["source"]
        assert isinstance(embedded["embedding"], list)
        assert all(
            isinstance(value, float) for value in embedded["embedding"]
        )


def test_generate_embeddings_for_chunks_does_not_mutate_input(monkeypatch):
    monkeypatch.setattr(
        "src.embeddings._get_model",
        lambda: FakeModel()
    )

    chunks = [
        {
            "chunk_id": 1,
            "page": 1,
            "text": "sample text",
            "source": "sample.pdf"
        }
    ]

    generate_embeddings_for_chunks(chunks)

    assert "embedding" not in chunks[0]


def test_generate_embeddings_for_chunks_with_empty_list(monkeypatch):
    monkeypatch.setattr(
        "src.embeddings._get_model",
        lambda: FakeModel()
    )

    assert generate_embeddings_for_chunks([]) == []


def test_model_is_loaded_lazily_and_only_once(monkeypatch):
    """
    The real model is a real disk + memory cost. It must not be
    loaded until it's actually needed, and must not be reloaded
    on every call.
    """
    load_count = {"count": 0}

    def fake_loader():
        load_count["count"] += 1
        return FakeModel()

    monkeypatch.setattr("src.embeddings._model", None)
    monkeypatch.setattr("src.embeddings._load_model", fake_loader)

    generate_embedding("first call")
    generate_embedding("second call")

    assert load_count["count"] == 1
    