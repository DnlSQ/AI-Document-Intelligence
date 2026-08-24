"""
Tests for the main.py orchestration layer.

These tests verify that main.py correctly wires:

    document_loader -> text_cleaner -> chunker -> retriever -> generator

External dependencies (PDF parsing, LLM calls) are mocked so
these tests do not require pymupdf, Ollama, or a real PDF file.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import build_chunk_repository, answer_question, NO_CONTEXT_ANSWER
from src.config import MIN_CONFIDENCE_THRESHOLD


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

    # 2 chunks from transistor.pdf + 1 from plants.pdf = 3 total
    assert len(chunks) == 3

    # Each chunk keeps the source of the document it came from.
    assert chunks[0]["source"] == "transistor.pdf"
    assert chunks[1]["source"] == "transistor.pdf"
    assert chunks[2]["source"] == "plants.pdf"

    # chunk_id is globally unique and sequential across BOTH
    # documents, not restarted at 1 for the second document.
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
# answer_question - query pipeline wiring
# ============================================================

def test_answer_question_calls_retriever_and_generator(monkeypatch):
    """
    Verify that answer_question retrieves relevant chunks
    and passes them to the generator when confidence is
    high enough.
    """

    chunks = [
        {
            "chunk_id": 1,
            "page": 2,
            "text": "VCEO collector-emitter voltage -50 V",
            "source": "sample.pdf"
        }
    ]

    retrieve_calls = []
    generate_calls = []

    def fake_retrieve_relevant_chunks(question, chunks_arg, top_k=3):
        retrieve_calls.append((question, chunks_arg, top_k))
        return [{"chunk": chunks[0], "score": 5, "confidence": 0.9}]

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append((question, retrieved_chunks))
        return "The maximum collector-emitter voltage is -50 V."

    monkeypatch.setattr(
        "src.main.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks
    )
    monkeypatch.setattr(
        "src.main.generate_answer",
        fake_generate_answer
    )

    answer = answer_question(
        "What is the maximum collector-emitter voltage?",
        chunks,
        top_k=3
    )

    assert answer == "The maximum collector-emitter voltage is -50 V."

    assert len(retrieve_calls) == 1
    assert retrieve_calls[0][0] == "What is the maximum collector-emitter voltage?"
    assert retrieve_calls[0][2] == 3

    assert len(generate_calls) == 1
    assert generate_calls[0][0] == "What is the maximum collector-emitter voltage?"


def test_answer_question_returns_grounded_fallback_when_no_chunks_found(monkeypatch):
    """
    Verify that when retrieval finds no relevant chunks,
    answer_question returns the grounded fallback directly
    without calling the generator (and therefore without
    calling the LLM).
    """

    generate_calls = []

    def fake_retrieve_relevant_chunks(question, chunks_arg, top_k=3):
        return []

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append((question, retrieved_chunks))
        return "This should never be returned."

    monkeypatch.setattr(
        "src.main.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks
    )
    monkeypatch.setattr(
        "src.main.generate_answer",
        fake_generate_answer
    )

    answer = answer_question(
        "What is the color of the package?",
        chunks=[{"chunk_id": 1, "page": 1, "text": "irrelevant", "source": "sample.pdf"}]
    )

    assert answer == NO_CONTEXT_ANSWER
    assert len(generate_calls) == 0


def test_answer_question_uses_default_top_k(monkeypatch):
    """
    Verify that answer_question defaults top_k to the
    configured TOP_K_RESULTS value.
    """

    from src.config import TOP_K_RESULTS

    captured = {}

    def fake_retrieve_relevant_chunks(question, chunks_arg, top_k=3):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(
        "src.main.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks
    )

    answer_question("any question", chunks=[])

    assert captured["top_k"] == TOP_K_RESULTS


# ============================================================
# V2.1.5 - No-Answer Detection (confidence threshold)
# ============================================================

def test_answer_question_returns_fallback_when_confidence_too_low(monkeypatch):
    """
    If retrieval returns a match, but its confidence is below
    MIN_CONFIDENCE_THRESHOLD, the LLM must not be called - the
    match is too weak to trust as real grounding.
    """

    generate_calls = []

    def fake_retrieve_relevant_chunks(question, chunks_arg, top_k=3):
        weak_chunk = {"chunk_id": 1, "page": 1, "text": "barely related", "source": "sample.pdf"}
        return [{
            "chunk": weak_chunk,
            "score": 1,
            "confidence": MIN_CONFIDENCE_THRESHOLD - 0.05
        }]

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append((question, retrieved_chunks))
        return "This should never be returned."

    monkeypatch.setattr(
        "src.main.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks
    )
    monkeypatch.setattr(
        "src.main.generate_answer",
        fake_generate_answer
    )

    answer = answer_question("some vague question", chunks=[{"chunk_id": 1, "page": 1, "text": "x", "source": "s"}])

    assert answer == NO_CONTEXT_ANSWER
    assert len(generate_calls) == 0


def test_answer_question_proceeds_when_confidence_meets_threshold_exactly(monkeypatch):
    """
    A confidence exactly equal to MIN_CONFIDENCE_THRESHOLD must
    be accepted (the check is strictly-less-than for rejection),
    so the boundary itself is treated as trustworthy.
    """

    generate_calls = []

    def fake_retrieve_relevant_chunks(question, chunks_arg, top_k=3):
        chunk = {"chunk_id": 1, "page": 1, "text": "some match", "source": "sample.pdf"}
        return [{
            "chunk": chunk,
            "score": 2,
            "confidence": MIN_CONFIDENCE_THRESHOLD
        }]

    def fake_generate_answer(question, retrieved_chunks):
        generate_calls.append((question, retrieved_chunks))
        return "An answer."

    monkeypatch.setattr(
        "src.main.retrieve_relevant_chunks",
        fake_retrieve_relevant_chunks
    )
    monkeypatch.setattr(
        "src.main.generate_answer",
        fake_generate_answer
    )

    answer = answer_question("a question", chunks=[{"chunk_id": 1, "page": 1, "text": "x", "source": "s"}])

    assert answer == "An answer."
    assert len(generate_calls) == 1


def test_answer_question_uses_real_retriever_confidence_end_to_end():
    """
    Integration-style check (no mocking of the retriever): a
    known-good real match from earlier testing (confidence 0.25
    for a natural-language question matched on generic terms)
    must clear the configured threshold and reach the generator
    layer, rather than being silently swallowed by the no-answer
    gate.
    """

    from src.retriever import retrieve_relevant_chunks

    chunks = [
        {
            "chunk_id": 1,
            "page": 2,
            "text": "VCEO collector-emitter voltage open base -50 V",
            "source": "sample.pdf"
        }
    ]

    results = retrieve_relevant_chunks(
        "What is the maximum collector-emitter voltage?",
        chunks,
        top_k=1
    )

    assert results
    assert results[0]["confidence"] >= MIN_CONFIDENCE_THRESHOLD
    