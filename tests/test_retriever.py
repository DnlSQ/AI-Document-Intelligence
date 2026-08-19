from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks

from src.retriever import (
    tokenize,
    calculate_relevance_score,
    retrieve_relevant_chunks
)


PDF_PATH = "data/documents/sample.pdf"


def load_chunks():
    """
    Load sample.pdf, clean its text and create document chunks.
    """

    pages = extract_text_from_pdf(PDF_PATH)

    for page in pages:
        page["text"] = clean_text(page["text"])

    return create_document_chunks(
        pages,
        chunk_size=1000,
        chunk_overlap=150,
        source=PDF_PATH
    )


# ============================================================
# EXISTING COMPATIBILITY TESTS
# ============================================================


def test_retriever_returns_results():
    """
    Verify that the retriever returns at least one result.
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the maximum output current?",
        chunks
    )

    assert results


def test_retriever_returns_scores():
    """
    Verify that retrieved results contain a relevance score.
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the maximum output current?",
        chunks
    )

    for result in results:
        assert "chunk" in result
        assert "score" in result

        assert isinstance(result["score"], (int, float))
        assert result["score"] > 0


def test_question_1_output_current():
    """
    Question 1:

    What is the maximum output current of the PDTB113ZT?

    Expected:
    500 mA
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the maximum output current of the PDTB113ZT?",
        chunks
    )

    combined_text = " ".join(
        result["chunk"]["text"]
        for result in results
    )

    assert "output current" in combined_text.lower()
    assert "500 mA" in combined_text


def test_question_2_collector_emitter_voltage():
    """
    Question 2:

    What is the maximum collector-emitter voltage?

    Expected:
    VCEO = -50 V
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the maximum collector-emitter voltage?",
        chunks
    )

    combined_text = " ".join(
        result["chunk"]["text"]
        for result in results
    )

    assert "VCEO" in combined_text
    assert "-50" in combined_text


def test_question_3_input_voltage():
    """
    Question 3:

    What is the maximum input voltage?

    Expected:
    +5 V / -10 V
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the maximum input voltage?",
        chunks
    )

    combined_text = " ".join(
        result["chunk"]["text"]
        for result in results
    )

    assert "input voltage" in combined_text.lower()
    assert "5" in combined_text
    assert "-10" in combined_text


def test_question_4_dc_current_gain():
    """
    Question 4:

    What is the DC current gain?

    Expected:
    hFE = 70
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the DC current gain?",
        chunks
    )

    combined_text = " ".join(
        result["chunk"]["text"]
        for result in results
    )

    assert "hFE" in combined_text
    assert "DC current gain" in combined_text
    assert "70" in combined_text


# ============================================================
# V2.1 - IMPROVED TOKENIZATION
# ============================================================


def test_tokenize_preserves_technical_identifiers():
    """
    V2.1:

    Technical identifiers such as VCEO, hFE, R1 and R2
    must remain identifiable after tokenization.
    """

    text = (
        "VCEO collector-emitter voltage "
        "hFE DC current gain R1 = 1 kΩ R2 = 10 kΩ"
    )

    tokens = tokenize(text)

    assert "vceo" in tokens
    assert "hfe" in tokens
    assert "r1" in tokens
    assert "r2" in tokens


def test_tokenize_handles_hyphenated_terms():
    """
    V2.1:

    Hyphenated technical terminology should preserve
    useful components for retrieval.

    Example:
        collector-emitter

    should allow retrieval of:
        collector
        emitter
    """

    tokens = tokenize(
        "collector-emitter voltage"
    )

    assert "collector" in tokens
    assert "emitter" in tokens
    assert "voltage" in tokens


def test_tokenize_normalizes_case():
    """
    V2.1:

    Tokenization should be case-insensitive.
    """

    tokens = tokenize(
        "VCEO vceo VCEO"
    )

    assert tokens.count("vceo") == 3


def test_tokenize_does_not_create_empty_tokens():
    """
    V2.1:

    Tokenization must not return empty strings.
    """

    tokens = tokenize(
        "VCEO   collector-emitter   voltage"
    )

    assert "" not in tokens


# ============================================================
# V2.1 - TECHNICAL TERM WEIGHTING
# ============================================================


def test_technical_identifier_is_more_relevant_than_generic_term():
    """
    V2.1:

    A technical identifier such as VCEO should contribute
    more relevance than a generic term such as voltage.

    This test intentionally compares two chunks:

        Chunk A:
            VCEO collector-emitter voltage

        Chunk B:
            voltage

    Chunk A must receive the higher score.
    """

    query = "What is the VCEO voltage?"

    score_technical = calculate_relevance_score(
        query,
        "VCEO collector-emitter voltage"
    )

    score_generic = calculate_relevance_score(
        query,
        "voltage"
    )

    assert score_technical > score_generic


def test_hfe_is_treated_as_technical_term():
    """
    V2.1:

    hFE should contribute meaningful relevance when
    searching for DC current gain.
    """

    query = "What is the hFE DC current gain?"

    score_with_hfe = calculate_relevance_score(
        query,
        "hFE DC current gain"
    )

    score_without_hfe = calculate_relevance_score(
        query,
        "DC current gain"
    )

    assert score_with_hfe > score_without_hfe


# ============================================================
# V2.1 - PHRASE MATCHING
# ============================================================


def test_exact_phrase_match_scores_higher():
    """
    V2.1:

    An exact phrase match should receive a higher score
    than a chunk containing the same terms separately.
    """

    query = "collector-emitter voltage"

    exact_phrase_score = calculate_relevance_score(
        query,
        "VCEO collector-emitter voltage open base -50 V"
    )

    separate_terms_score = calculate_relevance_score(
        query,
        "collector voltage and emitter current information"
    )

    assert exact_phrase_score > separate_terms_score


def test_output_current_phrase_match():
    """
    V2.1:

    The phrase "output current" should improve relevance
    for the correct chunk.
    """

    query = "maximum output current"

    relevant_score = calculate_relevance_score(
        query,
        "IO output current -500 mA"
    )

    weak_score = calculate_relevance_score(
        query,
        "collector voltage information"
    )

    assert relevant_score > weak_score


# ============================================================
# V2.1 - RANKING
# ============================================================


def test_retriever_ranks_exact_technical_match_first():
    """
    V2.1:

    The chunk containing the strongest technical and phrase
    match must be ranked first.
    """

    chunks = [
        {
            "chunk_id": 1,
            "page": 2,
            "text": "Voltage information for the device.",
            "source": PDF_PATH
        },
        {
            "chunk_id": 2,
            "page": 2,
            "text": (
                "VCEO collector-emitter voltage "
                "open base -50 V"
            ),
            "source": PDF_PATH
        },
        {
            "chunk_id": 3,
            "page": 3,
            "text": (
                "Collector current characteristics "
                "and typical values."
            ),
            "source": PDF_PATH
        }
    ]

    results = retrieve_relevant_chunks(
        "What is the maximum collector-emitter voltage?",
        chunks,
        top_k=3
    )

    assert results

    assert results[0]["chunk"]["chunk_id"] == 2


def test_retriever_ranks_output_current_chunk_first():
    """
    V2.1:

    The chunk containing the exact output-current information
    must be ranked first.
    """

    chunks = [
        {
            "chunk_id": 1,
            "page": 2,
            "text": "Input voltage information.",
            "source": PDF_PATH
        },
        {
            "chunk_id": 2,
            "page": 2,
            "text": "IO output current -500 mA.",
            "source": PDF_PATH
        },
        {
            "chunk_id": 3,
            "page": 3,
            "text": "Collector-emitter voltage -50 V.",
            "source": PDF_PATH
        }
    ]

    results = retrieve_relevant_chunks(
        "What is the maximum output current?",
        chunks,
        top_k=3
    )

    assert results

    assert results[0]["chunk"]["chunk_id"] == 2


def test_retriever_respects_top_k():
    """
    V2.1:

    The retriever must never return more than top_k results.
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "voltage",
        chunks,
        top_k=2
    )

    assert len(results) <= 2


def test_retriever_results_are_sorted_by_score():
    """
    V2.1:

    Results must be returned from highest relevance
    score to lowest relevance score.
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the maximum collector-emitter voltage?",
        chunks,
        top_k=3
    )

    scores = [
        result["score"]
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True
    )


# ============================================================
# V2.1 - RELEVANCE FILTERING
# ============================================================


def test_retriever_ignores_irrelevant_chunks():
    """
    V2.1:

    Chunks with no meaningful query overlap should not
    be returned by the retriever.
    """

    chunks = [
        {
            "chunk_id": 1,
            "page": 1,
            "text": "Package dimensions and mechanical outline.",
            "source": PDF_PATH
        },
        {
            "chunk_id": 2,
            "page": 2,
            "text": "VCEO collector-emitter voltage -50 V.",
            "source": PDF_PATH
        }
    ]

    results = retrieve_relevant_chunks(
        "collector-emitter voltage",
        chunks,
        top_k=3
    )

    assert results

    returned_ids = [
        result["chunk"]["chunk_id"]
        for result in results
    ]

    assert 2 in returned_ids


def test_retriever_returns_empty_for_unrelated_query():
    """
    V2.1:

    A completely unrelated query should return no results
    when there are no matching terms.
    """

    chunks = [
        {
            "chunk_id": 1,
            "page": 1,
            "text": "VCEO collector-emitter voltage -50 V.",
            "source": PDF_PATH
        },
        {
            "chunk_id": 2,
            "page": 2,
            "text": "IO output current -500 mA.",
            "source": PDF_PATH
        }
    ]

    results = retrieve_relevant_chunks(
        "What is the color of the package?",
        chunks,
        top_k=3
    )

    assert results == []
    