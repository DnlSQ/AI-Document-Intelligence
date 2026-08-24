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


def hybrid_retrieve(question, chunks, collection=None, top_k=3):
    """
    Retrieve the most relevant chunks for a question by fusing
    lexical retrieval and semantic search with Reciprocal Rank
    Fusion.

    Args:
        question: User question.
        chunks: Document chunk repository (for lexical retrieval).
        collection: Optional vector store collection (for tests).
            Defaults to the real persistent collection.
        top_k: Number of chunks to return after fusion.

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
            rrf_score += 1 / (RRF_K + lexical_rank)

        semantic_rank = semantic_ranks.get(chunk_id)
        if semantic_rank is not None:
            rrf_score += 1 / (RRF_K + semantic_rank)

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

    return fused_results[:top_k]
