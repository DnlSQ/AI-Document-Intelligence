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


def make_result(chunk_id, text="sample text", page=1, source="sample.pdf", confidence=0.0, score=0):
    return {
        "chunk": {
            "chunk_id": chunk_id,
            "page": page,
            "text": text,
            "source": source
        },
        "score": score,
        "confidence": confidence
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


# ---------------------------------------------------------------
# Lexical safety net (V6.1) - confirmed real-world case from
# NE555N.pdf, 2026-08-30: a chunk that scores highest of ANY
# chunk on lexical relevance, but is invisible to semantic search,
# must not be silently dropped by RRF fusion.
# ---------------------------------------------------------------

def test_lexical_safety_net_rescues_high_confidence_chunk_dropped_by_fusion(monkeypatch):
    """
    Chunk 99 is the single best lexical match (confidence 0.44,
    above the safety-net threshold) but semantic search never
    finds it at all. Chunks 5 and 6 score only moderately on BOTH
    methods, which without the safety net would be enough to push
    chunk 99 out of a top_k=2 result entirely - exactly what
    happened with the real NE555N.pdf "operating supply voltage
    range" question.
    """
    lexical_results = [
        make_result(99, confidence=0.44),
        make_result(5),
        make_result(6),
    ]
    semantic_results = [make_result(5), make_result(6)]

    monkeypatch.setattr(
        "src.hybrid_retrieval.retrieve_relevant_chunks",
        lambda question, chunks, top_k: lexical_results
    )
    monkeypatch.setattr(
        "src.hybrid_retrieval.semantic_search",
        lambda question, top_k, collection: semantic_results
    )

    results = hybrid_retrieve("question", chunks=[], collection=None, top_k=2)

    result_ids = [r["chunk"]["chunk_id"] for r in results]
    assert result_ids == [5, 99], "chunk 6 (the weakest survivor) should be replaced by chunk 99"


def test_lexical_safety_net_does_not_duplicate_already_included_chunk(monkeypatch):
    """
    When the best lexical chunk already survives the fusion
    naturally (top_k large enough here), the safety net must be a
    no-op - it never duplicates an entry.
    """
    lexical_results = [
        make_result(99, confidence=0.44),
        make_result(5),
        make_result(6),
    ]
    semantic_results = [make_result(5), make_result(6)]

    monkeypatch.setattr(
        "src.hybrid_retrieval.retrieve_relevant_chunks",
        lambda question, chunks, top_k: lexical_results
    )
    monkeypatch.setattr(
        "src.hybrid_retrieval.semantic_search",
        lambda question, top_k, collection: semantic_results
    )

    results = hybrid_retrieve("question", chunks=[], collection=None, top_k=3)

    result_ids = [r["chunk"]["chunk_id"] for r in results]
    assert len(result_ids) == len(set(result_ids)), "no duplicate chunks"
    assert 99 in result_ids


def test_lexical_safety_net_ignores_low_confidence_lexical_match(monkeypatch):
    """
    A lexical-only chunk with LOW confidence must stay excluded -
    the safety net only rescues chunks lexical retrieval is
    genuinely confident about, never a weak, coincidental match.
    """
    lexical_results = [
        make_result(99, confidence=0.2),
        make_result(5),
        make_result(6),
    ]
    semantic_results = [make_result(5), make_result(6)]

    monkeypatch.setattr(
        "src.hybrid_retrieval.retrieve_relevant_chunks",
        lambda question, chunks, top_k: lexical_results
    )
    monkeypatch.setattr(
        "src.hybrid_retrieval.semantic_search",
        lambda question, top_k, collection: semantic_results
    )

    results = hybrid_retrieve("question", chunks=[], collection=None, top_k=2)

    result_ids = [r["chunk"]["chunk_id"] for r in results]
    assert 99 not in result_ids
    assert result_ids == [5, 6]

def test_lexical_safety_net_rescues_tied_lexical_sibling_when_tie_break_winner_already_included(monkeypatch):
    """
    Chunk 33 and chunk 41 tie exactly on lexical score/confidence
    (score=7, confidence=0.47) - the real NE555N.pdf "turn off
    time" case, 2026-08-30: chunk 33 is a feature-list bullet
    that only NAMES the spec ("Low turn-off time"), chunk 41 is
    the data table entry that actually HOLDS its value. Lexical
    retrieval's own tie-break (ascending chunk_id) makes 33 "rank
    1" over 41 purely by coincidence of ID ordering - it has no
    way to know which one actually answers the question.

    Chunk 33 makes the fused top_k on its own (it also ties with
    chunk 21's semantic rank 1, and wins THAT tie by chunk_id
    convention), so a safety net that only ever looks at
    lexical_results[0] would consider its job done and never
    rescue chunk 41 - even though 41 is equally confident and is
    the chunk that actually contains the answer. The safety net
    must check every chunk tied for the best lexical score, not
    just the first one.
    """
    lexical_results = [
        make_result(33, score=7, confidence=0.47),
        make_result(41, score=7, confidence=0.47),
        make_result(42, score=5, confidence=0.56),
    ]
    semantic_results = [
        make_result(21, confidence=0.29),
        make_result(17, confidence=0.26),
    ]

    monkeypatch.setattr(
        "src.hybrid_retrieval.retrieve_relevant_chunks",
        lambda question, chunks, top_k: lexical_results
    )
    monkeypatch.setattr(
        "src.hybrid_retrieval.semantic_search",
        lambda question, top_k, collection: semantic_results
    )

    results = hybrid_retrieve("question", chunks=[], collection=None, top_k=3)

    result_ids = [r["chunk"]["chunk_id"] for r in results]
    assert result_ids == [21, 33, 41], "chunk 17 (the weakest survivor) should be replaced by chunk 41"
    