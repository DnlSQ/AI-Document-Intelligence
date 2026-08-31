"""
V3.4 Hybrid Retrieval.

Responsible only for:

    - Combining lexical retrieval (retriever.py) and semantic
      search (semantic_search.py) results into a single ranked
      list, using Reciprocal Rank Fusion (RRF).

Must not contain:

    - Lexical scoring logic (see retriever.py).
    - Embedding generation or vector search logic (see
      embeddings.py, vector_store.py, semantic_search.py).
    - LLM logic or prompt construction (see generator.py, llm.py).

Why RRF, not a weighted average of scores: the two methods'
scores are not on comparable scales (retriever.py's score is an
unbounded integer; semantic_search.py's score is a cosine
similarity in [-1, 1]), so averaging them directly would be
meaningless. RRF instead combines each method's RANK POSITION for
a chunk, which is always comparable regardless of how that rank
was computed. This is a standard, published technique (Cormack
et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods", SIGIR 2009), not an invented
heuristic.
"""
from src.retriever import retrieve_relevant_chunks
from src.semantic_search import semantic_search

# Standard RRF damping constant. 60 is the value used in the
# original RRF paper and the default most production hybrid
# search systems (Elasticsearch, Weaviate, etc.) use. It controls
# how much a chunk's exact rank matters - kept at the literature
# default rather than inventing a new number without evidence to
# tune it against.
RRF_K = 60

# Number of results pulled from EACH method before fusing - not
# the final number returned. Must be larger than top_k: a chunk
# ranked low in one method (e.g. #8) can still end up near the
# top of the fused result if it also ranks high in the other
# method, but only if it was actually in each method's candidate
# pool to begin with.
CANDIDATE_POOL_SIZE = 10

# Minimum lexical confidence (see retriever.calculate_confidence)
# for a chunk to be force-included in the final results via the
# lexical safety net below, even when Reciprocal Rank Fusion would
# otherwise drop it entirely. Calibrated from two real chunks
# confirmed (via diagnose_retrieval.py, 2026-08-30) to hold the
# complete, correct answer to a real question about a
# previously-unseen document (NE555N.pdf) - lexical confidence
# 0.44 and 0.47 respectively - set comfortably below both so
# similar future cases are still caught, while staying well above
# MIN_CONFIDENCE_THRESHOLD (0.15) so a merely-adequate lexical
# match isn't forced in without real conviction.
# Default multipliers applied to each method's RRF contribution
# before summing (see hybrid_retrieve). Both default to 1.0,
# reproducing the original unweighted formula exactly - these are
# the values used when the caller does not override them. Not
# tuned yet: RAG v7.2 introduces the mechanism first, then
# test_hybrid_weighting_manual.py measures real candidate weights
# against the golden dataset before any default changes.
LEXICAL_WEIGHT = 1.0
SEMANTIC_WEIGHT = 1.0
LEXICAL_SAFETY_NET_THRESHOLD = 0.35


def _rank_positions(results):
    """
    Map each chunk_id in a ranked result list to its 1-indexed
    position (1 = best).

    Args:
        results: Either retriever.py or semantic_search.py output
            - a list of {"chunk": {...with chunk_id...}, ...},
            already sorted best-first.

    Returns:
        Dict[int, int]: chunk_id -> rank (1-indexed).
    """
    return {
        result["chunk"]["chunk_id"]: rank
        for rank, result in enumerate(results, start=1)
    }


def _apply_lexical_safety_net(top_results, fused_results, lexical_results):
    """
    Force-include a strong lexical match that Reciprocal Rank
    Fusion would otherwise drop entirely from the final results.

    Looks at every lexical result TIED for the single best raw
    score - not just lexical_results[0] - because two chunks can
    score identically (e.g. a feature-list bullet mentioning a
    term by name vs. the data table entry that actually holds its
    value), and retriever.py's own tie-break (ascending chunk_id)
    has no way to know which one actually answers the question.
    Rescuing only lexical_results[0] would silently accept
    whichever twin happened to win that arbitrary tie-break, even
    when its tied sibling - equally confident, and still missing
    from the final results - is the one with the real answer.
    Confirmed with a real question (NE555N.pdf "turn off time",
    2026-08-30): chunk 33 (a feature-list mention) and chunk 41
    (the data table entry) tied at score=7/confidence 0.47; chunk
    33 already made the fused top_k on its own merits, so the
    original single-best version of this function considered its
    job done and never looked at chunk 41 at all.

    Among the tied, still-missing candidates that clear
    LEXICAL_SAFETY_NET_THRESHOLD, the first one (in
    lexical_results' own deterministic order) is rescued by
    replacing the weakest (last) entry in top_results - never by
    fabricating data, always by looking up that chunk's real fused
    entry in fused_results first.

    Args:
        top_results: Final fused results, already truncated to
            top_k (see hybrid_retrieve).
        fused_results: The FULL fused list, before truncation -
            needed to look up a rescued chunk's real RRF entry.
        lexical_results: Raw lexical retrieval results (best
            first), as returned by retrieve_relevant_chunks.

    Returns:
        top_results, or a copy with its last entry replaced by
        the rescued chunk's real fused result.
    """
    if not lexical_results:
        return top_results

    top_result_ids = {result["chunk"]["chunk_id"] for result in top_results}

    best_score = lexical_results[0]["score"]

    candidates = [
        result for result in lexical_results
        if result["score"] == best_score
        and result["confidence"] >= LEXICAL_SAFETY_NET_THRESHOLD
        and result["chunk"]["chunk_id"] not in top_result_ids
    ]

    if not candidates:
        return top_results

    rescued_id = candidates[0]["chunk"]["chunk_id"]

    fused_by_id = {
        result["chunk"]["chunk_id"]: result for result in fused_results
    }
    safety_net_result = fused_by_id.get(rescued_id)

    if safety_net_result is None:
        return top_results

    if not top_results:
        return [safety_net_result]

    return top_results[:-1] + [safety_net_result]


def hybrid_retrieve(
    question, chunks, collection=None, top_k=3,
    lexical_weight=LEXICAL_WEIGHT, semantic_weight=SEMANTIC_WEIGHT,
):
    """
    Retrieve the most relevant chunks for a question by fusing
    lexical retrieval and semantic search with Reciprocal Rank
    Fusion, then applying a lexical safety net (see
    _apply_lexical_safety_net) so a chunk lexical retrieval is
    highly confident about is never silently dropped just because
    semantic search ignored it.

    Args:
        question: User question.
        chunks: Document chunk repository (for lexical retrieval).
        collection: Optional vector store collection (for tests).
            Defaults to the real persistent collection.
        top_k: Number of chunks to return after fusion.
        lexical_weight: Multiplier applied to lexical retrieval's
            RRF contribution before summing. Defaults to
            LEXICAL_WEIGHT (1.0), reproducing the original
            unweighted formula.
        semantic_weight: Multiplier applied to semantic search's
            RRF contribution before summing. Defaults to
            SEMANTIC_WEIGHT (1.0), reproducing the original
            unweighted formula.

    Returns:
        List of results, ranked by fused score (best first), each:
            {
                "chunk": {...},
                "rrf_score": float,
                "lexical_rank": int or None,
                "semantic_rank": int or None,
                "lexical_confidence": float,  # 0.0 if lexical never found it
                "semantic_confidence": float  # 0.0 if semantic never found it
            }
        A chunk found by only one method still appears, with the
        other method's rank as None - it is not penalized for the
        other method's silence.
    """
    lexical_results = retrieve_relevant_chunks(
        question, chunks, top_k=CANDIDATE_POOL_SIZE
    )
    semantic_results = semantic_search(
        question, top_k=CANDIDATE_POOL_SIZE, collection=collection
    )

    lexical_ranks = _rank_positions(lexical_results)
    semantic_ranks = _rank_positions(semantic_results)

    lexical_confidence_by_id = {
        result["chunk"]["chunk_id"]: result["confidence"]
        for result in lexical_results
    }
    semantic_confidence_by_id = {
        result["chunk"]["chunk_id"]: result["confidence"]
        for result in semantic_results
    }

    chunk_by_id = {}
    for result in lexical_results + semantic_results:
        chunk_by_id[result["chunk"]["chunk_id"]] = result["chunk"]

    all_chunk_ids = set(lexical_ranks) | set(semantic_ranks)

    fused_results = []

    for chunk_id in all_chunk_ids:
        rrf_score = 0.0

        lexical_rank = lexical_ranks.get(chunk_id)
        if lexical_rank is not None:
            rrf_score += lexical_weight * (1 / (RRF_K + lexical_rank))

        semantic_rank = semantic_ranks.get(chunk_id)
        if semantic_rank is not None:
            rrf_score += semantic_weight * (1 / (RRF_K + semantic_rank))

        fused_results.append({
            "chunk": chunk_by_id[chunk_id],
            "rrf_score": rrf_score,
            "lexical_rank": lexical_rank,
            "semantic_rank": semantic_rank,
            "lexical_confidence": lexical_confidence_by_id.get(chunk_id, 0.0),
            "semantic_confidence": semantic_confidence_by_id.get(chunk_id, 0.0)
        })

    fused_results.sort(
        key=lambda result: (result["rrf_score"], -result["chunk"]["chunk_id"]),
        reverse=True
    )

    top_results = fused_results[:top_k]

    top_results = _apply_lexical_safety_net(
        top_results, fused_results, lexical_results
    )

    return top_results

