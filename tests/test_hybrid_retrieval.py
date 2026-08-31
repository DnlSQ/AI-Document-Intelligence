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

def test_hybrid_retrieve_applies_custom_weights_to_rrf_formula(monkeypatch):
    """
    RAG v7.2: hybrid_retrieve accepts optional lexical_weight/
    semantic_weight to scale each method's RRF contribution before
    summing - the mechanism the V7.2 measurement script needs to
    compare candidate weightings against the golden dataset before
    picking a default. Both default to 1.0 (LEXICAL_WEIGHT/
    SEMANTIC_WEIGHT), reproducing the original unweighted formula
    exactly - every other test in this file still passes unchanged
    with the defaults.
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

    results = hybrid_retrieve(
        "question", chunks=[], collection=None, top_k=3,
        lexical_weight=2.0, semantic_weight=0.5,
    )

    chunk_1_result = next(r for r in results if r["chunk"]["chunk_id"] == 1)
    expected_score = round(2.0 * (1 / (RRF_K + 2)) + 0.5 * (1 / (RRF_K + 1)), 8)
    assert round(chunk_1_result["rrf_score"], 8) == expected_score


def test_hybrid_retrieve_weighting_can_change_the_final_ranking(monkeypatch):
    """
    The whole point of weighting: a chunk found by BOTH methods
    (lexical rank 1 + a weak semantic rank) should lose the top
    spot to a chunk found ONLY by semantic search at its best rank,
    once semantic_weight is raised enough - proving the weight
    genuinely participates in ranking, not just cosmetic score
    reporting.

    chunk 99 is an unrelated filler chunk occupying semantic rank
    2, purely so chunk 1's semantic appearance lands at rank 3
    (not rank 1) - this keeps the unweighted comparison a clean,
    non-tied margin instead of the coincidental tie that a naive
    "both chunks are someone's rank 1" setup would produce.
    """
    lexical_results = [make_result(1)]
    semantic_results = [make_result(3), make_result(99), make_result(1)]

    monkeypatch.setattr(
        "src.hybrid_retrieval.retrieve_relevant_chunks",
        lambda question, chunks, top_k: lexical_results
    )
    monkeypatch.setattr(
        "src.hybrid_retrieval.semantic_search",
        lambda question, top_k, collection: semantic_results
    )

    unweighted = hybrid_retrieve("question", chunks=[], collection=None, top_k=1)
    assert unweighted[0]["chunk"]["chunk_id"] == 1

    heavily_semantic = hybrid_retrieve(
        "question", chunks=[], collection=None, top_k=1,
        lexical_weight=0.1, semantic_weight=5.0,
    )
    assert heavily_semantic[0]["chunk"]["chunk_id"] == 3

def test_lexical_safety_net_does_not_evict_a_result_that_is_itself_tied_for_best_score(monkeypatch):
    """
    Real case, NE555N.pdf "output fall time", 2026-08-31: chunk 47
    (the correctly reconstructed table fact) and chunk 56 (an
    unrelated page whose lexical score is inflated purely by a
    boilerplate header repeated on nearly every page) tie for the
    best raw lexical score. Fusion naturally seats one tied member
    at the top and another tied member at the bottom of top_k,
    with a decent-but-not-tied semantic performer in between.

    The safety net must never evict a top_result that is ITSELF
    tied for the best score (chunk 10 or chunk 40 below) just to
    admit yet another member of that same evidence class (chunk
    30) - that trade doesn't fix anything. It must instead sacrifice
    the one slot that is NOT part of that tied class (chunk 20).
    """
    lexical_results = [
        make_result(10, score=7, confidence=0.5),  # rank 1 - tied, in top_k naturally
        make_result(30, score=7, confidence=0.5),  # rank 2 - tied, dropped by fusion - the rescue candidate
        make_result(20, score=4, confidence=0.6),  # rank 3 - NOT tied, decent semantic, should be sacrificed
        make_result(40, score=7, confidence=0.5),  # rank 4 - tied, kept in top_k by a strong semantic rank
    ]
    semantic_results = [
        make_result(40),  # rank 1
        make_result(20),  # rank 2
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
    assert 20 not in result_ids, "the non-tied chunk should be the one sacrificed"
    assert set(result_ids) == {10, 30, 40}, "both original tied members must survive alongside the rescued one"


def test_lexical_safety_net_declines_to_rescue_when_every_slot_is_tied_for_best_score(monkeypatch):
    """
    Real case, NE555N.pdf "output rise time", 2026-08-31: all three
    fused top_k slots happen to be tied for the best lexical score
    (chunks 53, 45, 47 in the real trace). A fourth tied chunk (56)
    is dropped by fusion and would technically qualify for rescue,
    but there is no slot left that ISN'T itself part of the same
    tied, high-confidence evidence class - sacrificing any of them
    to admit an equally-tied peer would not be a rescue, just a
    pointless swap. The safety net must leave the results
    untouched in this case.
    """
    lexical_results = [
        make_result(1, score=7, confidence=0.5),
        make_result(2, score=7, confidence=0.5),
        make_result(3, score=7, confidence=0.5),
        make_result(4, score=7, confidence=0.5),  # dropped by fusion, would-be rescue candidate
    ]
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

    result_ids = [r["chunk"]["chunk_id"] for r in results]
    assert result_ids == [1, 2, 3], "no safe slot to sacrifice, so no rescue happens"
    