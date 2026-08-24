"""
Tests for the main.py orchestration layer.

These tests verify that main.py correctly wires:

    document_loader -> text_cleaner -> chunker -> embeddings ->
    vector_store -> hybrid_retrieval (V3.4) -> generator

External dependencies (PDF parsing, LLM calls, the real embedding
model) are mocked so these tests do not require pymupdf, Ollama,
or a real PDF file. chromadb itself is used for real (in-memory),
same reasoning as test_vector_store.py - it's cheap enough to
exercise for real rather than mock.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb

from src.main import (
    build_chunk_repository,
    build_vector_store,
    answer_question,
    NO_CONTEXT_ANSWER
)
from src.config import MIN_CONFIDENCE_THRESHOLD
from src.vector_store import get_collection, get_chunk_count, add_chunks_to_store


# ============================================================
# build_chunk_repository - ingestion pipeline wiring
# ============================================================

def test_build_chunk_repository_combines_multiple_documents(monkeypatch):
    """
    Verify that build_chunk_repository loads EACH pdf path,
    cleans every page, tags chunks with their correct source,
    and combines everything into a single repository with
    globally unique, sequential chunk_id values (not restarting
    at 1 per document).
    """

    fake_pages_by_path = {
        "transistor.pdf": [
            {"page": 1, "text": "  VCEO   collector-emitter voltage -50 V  "},
            {"page": 2, "text": "hFE   DC current gain 70"},
        ],
        "plants.pdf": [
            {"page": 1, "text": "  Las gimnospermas presentan semilla desnuda  "},
        ],
    }

    def fake_extract_text_from_pdf(pdf_path):
        return [dict(p) for p in fake_pages_by_path[pdf_path]]

    def fake_clean_text(text):
        return text.strip()

    monkeypatch.setattr(
        "src.main.extract_text_from_pdf",
        fake_extract_text_from_pdf
    )
    monkeypatch.setattr(
        "src.main.clean_text",
        fake_clean_text
    )

    chunks = build_chunk_repository(pdf_paths=["transistor.pdf", "plants.pdf"])

    assert len(chunks) == 3
    assert chunks[0]["source"] == "transistor.pdf"
    assert chunks[1]["source"] == "transistor.pdf"
    assert chunks[2]["source"] == "plants.pdf"
    assert [chunk["chunk_id"] for chunk in chunks] == [1, 2, 3]
    assert chunks[0]["text"] == "VCEO   collector-emitter voltage -50 V"
    assert chunks[2]["text"] == "Las gimnospermas presentan semilla desnuda"


def test_build_chunk_repository_uses_default_document_paths(monkeypatch):
    """
    Verify that build_chunk_repository defaults to the
    configured DOCUMENT_PATHS when no paths are provided, and
    loads every one of them.
    """

    from src.config import DOCUMENT_PATHS

    captured_paths = []

    def fake_extract_text_from_pdf(pdf_path):
        captured_paths.append(pdf_path)
        return []

    monkeypatch.setattr(
        "src.main.extract_text_from_pdf",
        fake_extract_text_from_pdf
    )

    build_chunk_repository()

    assert captured_paths == DOCUMENT_PATHS


# ============================================================
# answer_question - query pipeline wiring (V3.4: hybrid_retrieve)
# ============================================================

def test_answer_question_calls_hybrid_retrieve_and_generator(monkeypatch):
    """
    Verify that answer_question retrieves via hybrid_retrieve
    (V3.4) and passes the results to the generator when
    confidence is high enough.
    """

    chunks = [
        {
            "chunk_id": 1,
            "page": 2,
            "text": "VCEO collector-emitter voltage -50 V",
            "source": "sample.pdf"
        }
    ]

    hybrid_calls = []
    generate_calls = []

    def fake_hybrid_retrieve(question, chunks_arg, collection=None, top_k=3):
        hybrid_calls.append((question, chunks_arg, collection, top_k))
        return [{
            "chunk": chunks[0],
            "rrf_score": 0.03,
            "lexical_rank": 1,
            "semantic_rank": 1,
            "lexical_confidence": 0.9,
            "semantic_confidence": 0.8
        }]

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append((question, retrieved_chunks))
        return "The maximum collector-emitter voltage is -50 V."

    monkeypatch.setattr("src.main.hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr("src.main.generate_answer", fake_generate_answer)

    answer = answer_question(
        "What is the maximum collector-emitter voltage?",
        chunks,
        top_k=3
    )

    assert answer == "The maximum collector-emitter voltage is -50 V."
    assert len(hybrid_calls) == 1
    assert hybrid_calls[0][0] == "What is the maximum collector-emitter voltage?"
    assert hybrid_calls[0][3] == 3
    assert len(generate_calls) == 1
    assert generate_calls[0][0] == "What is the maximum collector-emitter voltage?"


def test_answer_question_returns_grounded_fallback_when_no_chunks_found(monkeypatch):
    """
    When hybrid_retrieve finds no relevant chunks, answer_question
    must return the grounded fallback directly without calling
    the generator (and therefore without calling the LLM).
    """

    generate_calls = []

    def fake_hybrid_retrieve(question, chunks_arg, collection=None, top_k=3):
        return []

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append((question, retrieved_chunks))
        return "This should never be returned."

    monkeypatch.setattr("src.main.hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr("src.main.generate_answer", fake_generate_answer)

    answer = answer_question(
        "What is the color of the package?",
        chunks=[{"chunk_id": 1, "page": 1, "text": "irrelevant", "source": "sample.pdf"}]
    )

    assert answer == NO_CONTEXT_ANSWER
    assert len(generate_calls) == 0


def test_answer_question_uses_default_top_k(monkeypatch):
    """
    Verify that answer_question defaults top_k to the configured
    TOP_K_RESULTS value.
    """

    from src.config import TOP_K_RESULTS

    captured = {}

    def fake_hybrid_retrieve(question, chunks_arg, collection=None, top_k=3):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr("src.main.hybrid_retrieve", fake_hybrid_retrieve)

    answer_question("any question", chunks=[])

    assert captured["top_k"] == TOP_K_RESULTS


# ============================================================
# V3.4 - No-Answer Detection now uses
# max(lexical_confidence, semantic_confidence) from the fused
# hybrid result, instead of one RRF-derived confidence. Why: RRF's
# own score scale is compressed by RRF_K=60 and would make
# MIN_CONFIDENCE_THRESHOLD nearly meaningless if used directly -
# so the gate still trusts each method's OWN already-calibrated
# confidence, while RRF only decides ordering.
# ============================================================

def test_answer_question_returns_fallback_when_both_confidences_too_low(monkeypatch):
    """
    If neither method was individually confident, the LLM must
    not be called.
    """

    generate_calls = []

    def fake_hybrid_retrieve(question, chunks_arg, collection=None, top_k=3):
        weak_chunk = {"chunk_id": 1, "page": 1, "text": "barely related", "source": "sample.pdf"}
        return [{
            "chunk": weak_chunk,
            "rrf_score": 0.001,
            "lexical_rank": 5,
            "semantic_rank": 5,
            "lexical_confidence": MIN_CONFIDENCE_THRESHOLD - 0.05,
            "semantic_confidence": MIN_CONFIDENCE_THRESHOLD - 0.05
        }]

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append((question, retrieved_chunks))
        return "This should never be returned."

    monkeypatch.setattr("src.main.hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr("src.main.generate_answer", fake_generate_answer)

    answer = answer_question(
        "some vague question",
        chunks=[{"chunk_id": 1, "page": 1, "text": "x", "source": "s"}]
    )

    assert answer == NO_CONTEXT_ANSWER
    assert len(generate_calls) == 0


def test_answer_question_proceeds_when_only_lexical_confidence_is_high(monkeypatch):
    """
    If lexical alone is confident (a strong technical-term match
    semantic search happened to miss - the real "DC current gain"
    case), the answer must still go through: the gate uses the
    BEST of the two, not an average or lexical-only.
    """

    generate_calls = []

    def fake_hybrid_retrieve(question, chunks_arg, collection=None, top_k=3):
        chunk = {"chunk_id": 1, "page": 1, "text": "some match", "source": "sample.pdf"}
        return [{
            "chunk": chunk,
            "rrf_score": 0.016,
            "lexical_rank": 1,
            "semantic_rank": None,
            "lexical_confidence": MIN_CONFIDENCE_THRESHOLD + 0.1,
            "semantic_confidence": 0.0
        }]

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append((question, retrieved_chunks))
        return "An answer."

    monkeypatch.setattr("src.main.hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr("src.main.generate_answer", fake_generate_answer)

    answer = answer_question("a question", chunks=[{"chunk_id": 1, "page": 1, "text": "x", "source": "s"}])

    assert answer == "An answer."
    assert len(generate_calls) == 1


def test_answer_question_proceeds_when_only_semantic_confidence_is_high(monkeypatch):
    """
    Mirror of the above: if semantic alone is confident (a
    paraphrased question lexical missed), the answer must still
    go through.
    """

    generate_calls = []

    def fake_hybrid_retrieve(question, chunks_arg, collection=None, top_k=3):
        chunk = {"chunk_id": 1, "page": 1, "text": "some match", "source": "sample.pdf"}
        return [{
            "chunk": chunk,
            "rrf_score": 0.016,
            "lexical_rank": None,
            "semantic_rank": 1,
            "lexical_confidence": 0.0,
            "semantic_confidence": MIN_CONFIDENCE_THRESHOLD + 0.1
        }]

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append((question, retrieved_chunks))
        return "An answer."

    monkeypatch.setattr("src.main.hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr("src.main.generate_answer", fake_generate_answer)

    answer = answer_question("a question", chunks=[{"chunk_id": 1, "page": 1, "text": "x", "source": "s"}])

    assert answer == "An answer."
    assert len(generate_calls) == 1


def test_answer_question_proceeds_when_confidence_meets_threshold_exactly(monkeypatch):
    """
    A confidence exactly equal to MIN_CONFIDENCE_THRESHOLD must
    be accepted (strictly-less-than for rejection), so the
    boundary itself is treated as trustworthy.
    """

    generate_calls = []

    def fake_hybrid_retrieve(question, chunks_arg, collection=None, top_k=3):
        chunk = {"chunk_id": 1, "page": 1, "text": "some match", "source": "sample.pdf"}
        return [{
            "chunk": chunk,
            "rrf_score": 0.016,
            "lexical_rank": 1,
            "semantic_rank": None,
            "lexical_confidence": MIN_CONFIDENCE_THRESHOLD,
            "semantic_confidence": 0.0
        }]

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append((question, retrieved_chunks))
        return "An answer."

    monkeypatch.setattr("src.main.hybrid_retrieve", fake_hybrid_retrieve)
    monkeypatch.setattr("src.main.generate_answer", fake_generate_answer)

    answer = answer_question("a question", chunks=[{"chunk_id": 1, "page": 1, "text": "x", "source": "s"}])

    assert answer == "An answer."
    assert len(generate_calls) == 1


def test_answer_question_uses_real_hybrid_retrieval_end_to_end(monkeypatch):
    """
    Integration-style check: real lexical retrieval + real (but
    empty) semantic search, no mocking of hybrid_retrieve itself.
    A real, known-good lexical match must still clear the
    no-answer gate through lexical_confidence alone, since the
    vector store here has nothing in it.
    """

    empty_collection = get_collection(
        client=chromadb.EphemeralClient(),
        name="test_answer_question_empty"
    )

    chunks = [
        {
            "chunk_id": 1,
            "page": 2,
            "text": "VCEO collector-emitter voltage open base -50 V",
            "source": "sample.pdf"
        }
    ]

    generate_calls = []

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append(retrieved_chunks)
        return "The maximum collector-emitter voltage is -50 V."

    monkeypatch.setattr("src.main.generate_answer", fake_generate_answer)

    answer = answer_question(
        "What is the maximum collector-emitter voltage?",
        chunks,
        collection=empty_collection
    )

    assert answer == "The maximum collector-emitter voltage is -50 V."
    assert len(generate_calls) == 1


# ============================================================
# build_vector_store - V3.4 ingestion wiring
# ============================================================

def test_build_vector_store_embeds_and_stores_all_chunks(monkeypatch):
    """
    Verify build_vector_store generates an embedding for every
    chunk and stores all of them, without needing the real
    embedding model or a real persistent store.
    """

    chunks = [
        {"chunk_id": 1, "page": 1, "text": "a", "source": "s.pdf"},
        {"chunk_id": 2, "page": 1, "text": "b", "source": "s.pdf"},
    ]

    def fake_generate_embeddings_for_chunks(chunks_arg):
        return [dict(chunk, embedding=[0.0, 0.0]) for chunk in chunks_arg]

    monkeypatch.setattr(
        "src.main.generate_embeddings_for_chunks",
        fake_generate_embeddings_for_chunks
    )

    collection = get_collection(
        client=chromadb.EphemeralClient(),
        name="test_build_vector_store"
    )

    returned_collection = build_vector_store(chunks, collection=collection)

    assert returned_collection is collection
    assert get_chunk_count(collection=collection) == 2


def test_build_vector_store_resets_before_writing(monkeypatch):
    """
    A rebuild must not leave stale chunks from a previous run
    behind (e.g. a document removed from DOCUMENT_PATHS) - see
    vector_store.reset_store's own docstring for why upsert alone
    is not enough.
    """

    collection = get_collection(
        client=chromadb.EphemeralClient(),
        name="test_build_vector_store_reset"
    )

    stale_chunk = {
        "chunk_id": 999,
        "page": 1,
        "text": "stale",
        "source": "old.pdf",
        "embedding": [0.0, 0.0]
    }
    add_chunks_to_store([stale_chunk], collection=collection)
    assert get_chunk_count(collection=collection) == 1

    def fake_generate_embeddings_for_chunks(chunks_arg):
        return [dict(chunk, embedding=[0.0, 0.0]) for chunk in chunks_arg]

    monkeypatch.setattr(
        "src.main.generate_embeddings_for_chunks",
        fake_generate_embeddings_for_chunks
    )

    new_chunks = [{"chunk_id": 1, "page": 1, "text": "fresh", "source": "new.pdf"}]
    build_vector_store(new_chunks, collection=collection)

    assert get_chunk_count(collection=collection) == 1
    result = collection.get(ids=["1"])
    assert result["documents"][0] == "fresh"
    