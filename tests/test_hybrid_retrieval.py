"""
Tests for the V3.4 Hybrid Retrieval module (src/hybrid_retrieval.py).

Mocks retrieve_relevant_chunks and semantic_search directly - the
same boundary-mocking pattern used throughout this project
(test_generator.py mocks ask_llm, test_semantic_search.py mocks
generate_embedding). This module's own job is fusion logic, not
retrieval itself, so tests control exactly what each method
"found" and verify the RRF math against it.
"""
from src.hybrid_retrieval import hybrid_retrieve, _rank_positions, RRF_K


def make_result(chunk_id, text="sample text", page=1, source="sample.pdf"):
    return {
        "chunk": {
            "chunk_id": chunk_id,
            "page": page,
            "text": text,
            "source": source
        },
        "score": 0,
        "confidence": 0.0
    }


def test_rank_positions_maps_chunk_id_to_one_indexed_rank():
    results = [make_result(5), make_result(2), make_result(9)]

    ranks = _rank_positions(results)

    assert ranks == {5: 1, 2: 2, 9: 3}


def test_hybrid_retrieve_boosts_chunk_found_by_both_methods(monkeypatch):
    """
    A chunk that ranks well in BOTH methods should outrank a
    chunk that ranks #1 in only one method - this mirrors the
    real "How much voltage can it withstand..." example from the
    V3.3 manual comparison.
    """
    lexical_results = [make_result(10), make_result(1)]   # chunk 1 is rank 2
    semantic_results = [make_result(1), make_result(20)]  # chunk 1 is rank 1

    monkeypatch.setattr(
        "src.hybrid_retrieval.retrieve_relevant_chunks",
        lambda question, chunks, top_k: lexical_results
    )
    monkeypatch.setattr(
        "src.hybrid_retrieval.semantic_search",
        lambda question, top_k, collection: semantic_results
    )

    results = hybrid_retrieve("question", chunks=[], collection=None, top_k=3)

    assert [r["chunk"]["chunk_id"] for r in results] == [1, 10, 20]

    chunk_1_result = results[0]
    expected_score = round(1 / (RRF_K + 2) + 1 / (RRF_K + 1), 8)
    assert round(chunk_1_result["rrf_score"], 8) == expected_score
    assert chunk_1_result["lexical_rank"] == 2
    assert chunk_1_result["semantic_rank"] == 1


def test_hybrid_retrieve_keeps_chunk_found_by_only_one_method(monkeypatch):
    """
    A chunk one method never found at all (the real "DC current
    gain" case, where semantic search found nothing relevant)
    must still surface, not get zeroed out.
    """
    lexical_results = [make_result(4)]
    semantic_results = []

    monkeypatch.setattr(
        "src.hybrid_retrieval.retrieve_relevant_chunks",
        lambda question, chunks, top_k: lexical_results
    )
    monkeypatch.setattr(
        "src.hybrid_retrieval.semantic_search",
        lambda question, top_k, collection: semantic_results
    )

    results = hybrid_retrieve("question", chunks=[], collection=None, top_k=3)

    assert len(results) == 1
    assert results[0]["chunk"]["chunk_id"] == 4
    assert results[0]["lexical_rank"] == 1
    assert results[0]["semantic_rank"] is None


def test_hybrid_retrieve_respects_top_k(monkeypatch):
    lexical_results = [make_result(1), make_result(2), make_result(3)]
    semantic_results = []

    monkeypatch.setattr(
        "src.hybrid_retrieval.retrieve_relevant_chunks",
        lambda question, chunks, top_k: lexical_results
    )
    monkeypatch.setattr(
        "src.hybrid_retrieval.semantic_search",
        lambda question, top_k, collection: semantic_results
    )

    results = hybrid_retrieve("question", chunks=[], collection=None, top_k=2)

    assert len(results) == 2


def test_hybrid_retrieve_with_no_results_from_either_method_returns_empty_list(monkeypatch):
    monkeypatch.setattr(
        "src.hybrid_retrieval.retrieve_relevant_chunks",
        lambda question, chunks, top_k: []
    )
    monkeypatch.setattr(
        "src.hybrid_retrieval.semantic_search",
        lambda question, top_k, collection: []
    )

    results = hybrid_retrieve("question", chunks=[], collection=None, top_k=3)

    assert results == []


def test_hybrid_retrieve_breaks_ties_by_ascending_chunk_id(monkeypatch):
    """
    Two chunks each found by only one method, both at rank 1,
    produce identical RRF scores. Ties must resolve
    deterministically, matching retriever.py's own tie-break
    convention (smaller chunk_id wins).
    """
    lexical_results = [make_result(50)]
    semantic_results = [make_result(7)]

    monkeypatch.setattr(
        "src.hybrid_retrieval.retrieve_relevant_chunks",
        lambda question, chunks, top_k: lexical_results
    )
    monkeypatch.setattr(
        "src.hybrid_retrieval.semantic_search",
        lambda question, top_k, collection: semantic_results
    )

    results = hybrid_retrieve("question", chunks=[], collection=None, top_k=2)

    assert [r["chunk"]["chunk_id"] for r in results] == [7, 50]
    