"""
Tests for V2.1.4 Confidence Scoring.

Confidence is a normalized value in [0.0, 1.0] derived from the
raw relevance score, computed per (query, chunk) pair as:

    confidence = score / max_possible_score(query, text)

where max_possible_score represents the score a "perfect" match
would achieve for the query tokens that actually appear in THIS
chunk (matched as technical, plus the exact phrase and chunk
bonuses). Query tokens that never appear in the chunk at all
(e.g. stopwords like "what"/"is"/"the" in a natural-language
question) are excluded from the denominator, since they never
had a chance to match in the first place - counting them would
unfairly penalize confidence for otherwise correct matches.
"""
from src.retriever import (
    calculate_max_possible_score,
    calculate_confidence,
    calculate_relevance_score,
    retrieve_relevant_chunks,
)


PDF_PATH = "data/documents/sample.pdf"


# ---------------------------------------------------------------------
# calculate_max_possible_score
# ---------------------------------------------------------------------

def test_max_possible_score_is_zero_for_empty_query():
    assert calculate_max_possible_score("", "VCEO rating -50 V") == 0


def test_max_possible_score_is_zero_when_no_query_token_matches():
    assert calculate_max_possible_score("VCEO", "unrelated text about packaging") == 0


def test_max_possible_score_scales_with_number_of_matched_tokens():
    text = "VCEO ICBO information here"

    single_match = calculate_max_possible_score("VCEO", text)
    two_matches = calculate_max_possible_score("VCEO ICBO", text)

    assert two_matches > single_match


def test_max_possible_score_ignores_query_tokens_not_in_chunk():
    """
    A query token that never appears in the chunk (like a
    stopword in a natural-language question) must not inflate
    the denominator.
    """
    text = "VCEO rating -50 V"

    with_stopwords = calculate_max_possible_score("what is the VCEO rating", text)
    without_stopwords = calculate_max_possible_score("VCEO rating", text)

    assert with_stopwords == without_stopwords


# ---------------------------------------------------------------------
# calculate_confidence
# ---------------------------------------------------------------------

def test_confidence_is_zero_for_empty_query():
    assert calculate_confidence(score=5, query="", text="VCEO rating -50 V") == 0.0


def test_confidence_is_zero_when_nothing_matched():
    assert calculate_confidence(score=0, query="VCEO", text="unrelated text") == 0.0


def test_confidence_is_between_zero_and_one():
    query = "VCEO ICBO output"
    text = "VCEO output information here"

    confidence = calculate_confidence(
        score=calculate_relevance_score(query, text),
        query=query,
        text=text
    )

    assert 0.0 <= confidence <= 1.0


def test_perfect_match_reaches_full_confidence():
    query = "VCEO"
    text = "VCEO rating -50 V"

    score = calculate_relevance_score(query, text)
    confidence = calculate_confidence(score, query, text)

    assert confidence == 1.0


def test_natural_language_question_no_longer_penalized_by_stopwords():
    """
    Regression test for the calibration bug: a natural-language
    question with several stopwords must not have those stopwords
    count against its confidence, since they never had a chance
    to match this (or any) chunk in the first place.
    """

    query = "What is the maximum collector-emitter voltage?"
    text = "VCEO collector-emitter voltage open base -50 V"

    score = calculate_relevance_score(query, text)
    confidence = calculate_confidence(score, query, text)

    # 3 generic matches (collector, emitter, voltage), no exact
    # phrase, no technical hit -> score 3.
    # max = 3 matched tokens * TECHNICAL_TERM_WEIGHT(3)
    #     + EXACT_PHRASE_BONUS(2) + TECHNICAL_CHUNK_BONUS(1) = 12
    # confidence = 3 / 12 = 0.25
    assert score == 3
    assert confidence == 0.25


def test_higher_score_yields_higher_confidence_for_same_pair():
    query = "VCEO ICBO"
    text = "VCEO ICBO details listed here"

    low_confidence = calculate_confidence(score=3, query=query, text=text)
    high_confidence = calculate_confidence(score=6, query=query, text=text)

    assert high_confidence > low_confidence


def test_confidence_never_exceeds_one_even_with_repeated_query_terms():
    query = "VCEO VCEO"
    text = "VCEO VCEO rating -50 V"

    score = calculate_relevance_score(query, text)
    confidence = calculate_confidence(score, query, text)

    assert confidence <= 1.0


# ---------------------------------------------------------------------
# retrieve_relevant_chunks - confidence must be included per result
# ---------------------------------------------------------------------

def test_retrieve_relevant_chunks_includes_confidence():
    chunks = [
        {
            "chunk_id": 1,
            "page": 2,
            "text": "VCEO collector-emitter voltage -50 V",
            "source": PDF_PATH,
        }
    ]

    results = retrieve_relevant_chunks("What is the VCEO voltage?", chunks, top_k=1)

    assert len(results) == 1
    assert "confidence" in results[0]
    assert 0.0 <= results[0]["confidence"] <= 1.0


def test_higher_ranked_chunk_has_greater_or_equal_confidence():
    query = "VCEO rating"

    chunks = [
        {"chunk_id": 1, "page": 1, "text": "rating information only", "source": PDF_PATH},
        {"chunk_id": 2, "page": 1, "text": "VCEO rating -50 V exact match here", "source": PDF_PATH},
    ]

    results = retrieve_relevant_chunks(query, chunks, top_k=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["confidence"] >= results[1]["confidence"]
    