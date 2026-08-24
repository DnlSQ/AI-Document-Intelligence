"""
Tests for V2.1.3 Ranking Improvements.

Focus: query-aware, fully deterministic tie-breaking.

The old tie-break checked whether a chunk contained ANY technical
term anywhere in its text, regardless of whether that term was
part of the user's query. This let an unrelated technical mention
(e.g. "ICBO" appearing in a chunk about "output current") win a
tie it had no real claim to.
"""
from src.retriever import (
    count_technical_term_matches,
    retrieve_relevant_chunks,
)


PDF_PATH = "data/documents/sample.pdf"


# ---------------------------------------------------------------------
# count_technical_term_matches
# ---------------------------------------------------------------------

def test_counts_only_query_relevant_technical_terms():
    query = "VCEO ICBO"
    text = "VCEO collector-emitter voltage -50 V"

    # Only VCEO is present; ICBO is not, so count must be 1.
    assert count_technical_term_matches(query, text) == 1


def test_counts_zero_when_query_has_no_technical_terms():
    query = "output current"
    text = "The ICBO leakage current is also mentioned here."

    # ICBO is technical, but it is not part of the query, so it
    # must not be counted.
    assert count_technical_term_matches(query, text) == 0


def test_counts_multiple_matching_technical_terms():
    query = "VCEO ICBO"
    text = "VCEO is -50 V and ICBO is 100 nA"

    assert count_technical_term_matches(query, text) == 2


# ---------------------------------------------------------------------
# retrieve_relevant_chunks - tie-break must be query-aware
# ---------------------------------------------------------------------

def test_unrelated_technical_term_does_not_win_a_tie():
    """
    Regression test for the old buggy tie-break.

    Both chunks match the same query terms with the same score.
    Chunk B additionally mentions an unrelated technical term
    (ICBO) that has nothing to do with the query. That mention
    must NOT cause chunk B to be ranked above chunk A.
    """

    query = "output current"

    chunks = [
        {
            "chunk_id": 2,
            "page": 1,
            "text": "the current output rating also mentions ICBO leakage",
            "source": PDF_PATH,
        },
        {
            "chunk_id": 1,
            "page": 1,
            "text": "the current output value is high",
            "source": PDF_PATH,
        },
    ]

    results = retrieve_relevant_chunks(query, chunks, top_k=2)

    assert len(results) == 2
    assert results[0]["score"] == results[1]["score"]

    # With equal scores and zero query-relevant technical matches
    # on both sides, the deterministic chunk_id tie-break must
    # decide: chunk_id 1 comes before chunk_id 2.
    assert results[0]["chunk"]["chunk_id"] == 1
    assert results[1]["chunk"]["chunk_id"] == 2


def test_tie_break_is_deterministic_regardless_of_input_order():
    """
    Two chunks with identical text (and therefore identical score
    and identical technical-match count) must always be ordered
    by chunk_id, no matter the order they were passed in.
    """

    query = "voltage rating"
    text = "voltage rating information for this component"

    chunks_order_a = [
        {"chunk_id": 5, "page": 1, "text": text, "source": PDF_PATH},
        {"chunk_id": 3, "page": 1, "text": text, "source": PDF_PATH},
    ]
    chunks_order_b = [
        {"chunk_id": 3, "page": 1, "text": text, "source": PDF_PATH},
        {"chunk_id": 5, "page": 1, "text": text, "source": PDF_PATH},
    ]

    results_a = retrieve_relevant_chunks(query, chunks_order_a, top_k=2)
    results_b = retrieve_relevant_chunks(query, chunks_order_b, top_k=2)

    assert [r["chunk"]["chunk_id"] for r in results_a] == [3, 5]
    assert [r["chunk"]["chunk_id"] for r in results_b] == [3, 5]


def test_query_relevant_technical_match_still_wins_a_tie():
    """
    When two chunks are tied on score, the chunk where a
    query-relevant technical term matches should be preferred
    over a chunk with no technical relevance to the query at all.
    """

    query = "VCEO rating hint"

    chunks = [
        {
            "chunk_id": 1,
            "page": 1,
            "text": "rating hint without the identifier",
            "source": PDF_PATH,
        },
        {
            "chunk_id": 2,
            "page": 1,
            "text": "VCEO value shown as a hint",
            "source": PDF_PATH,
        },
    ]

    results = retrieve_relevant_chunks(query, chunks, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk"]["chunk_id"] == 2
    