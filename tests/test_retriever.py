from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.retriever import (
    tokenize,
    normalize_text,
    calculate_relevance_score,
    calculate_confidence,
    retrieve_relevant_chunks,
    is_stopword
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
# BASIC TOKENIZATION TESTS
# ============================================================

def test_tokenize_returns_lowercase_tokens():
    """
    Verify that tokenization converts text to lowercase tokens.
    """

    tokens = tokenize("Collector-Emitter Voltage")

    assert tokens == [
        "collector",
        "emitter",
        "voltage"
    ]


def test_tokenize_handles_numbers():
    """
    Verify that numeric values are preserved as tokens.
    """

    tokens = tokenize("Maximum current is 500 mA")

    assert "500" in tokens
    assert "ma" in tokens


# ============================================================
# BASIC RELEVANCE TESTS
# ============================================================

def test_relevance_score_counts_matching_terms():
    """
    Verify that matching query terms contribute to the score.
    """

    score = calculate_relevance_score(
        "maximum output current",
        "maximum output current is 500 mA"
    )

    assert score >= 3


def test_relevance_score_zero_when_no_terms_match():
    """
    Verify that unrelated text receives a zero score.
    """

    score = calculate_relevance_score(
        "collector-emitter voltage",
        "package dimensions and thermal resistance"
    )

    assert score == 0


# ============================================================
# RETRIEVER BASIC TESTS
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
        assert isinstance(result["score"], int)
        assert result["score"] > 0


# ============================================================
# DOCUMENT QUESTIONS
# ============================================================

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
# V2.1.1 — EXACT PHRASE MATCHING
# ============================================================

def test_normalize_text_lowercases_text():
    """
    V2.1.1:

    Verify that normalize_text converts text to lowercase.
    """

    result = normalize_text(
        "Collector-Emitter Voltage"
    )

    assert result == "collector-emitter voltage"


def test_normalize_text_removes_extra_whitespace():
    """
    V2.1.1:

    Verify that normalize_text removes repeated whitespace.
    """

    result = normalize_text(
        "collector-emitter    voltage"
    )

    assert result == "collector-emitter voltage"


def test_exact_phrase_match_scores_higher():
    """
    V2.1.1:

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


def test_exact_phrase_match_is_detected_with_different_case():
    """
    V2.1.1:

    Phrase matching should be case-insensitive.
    """

    score_lowercase = calculate_relevance_score(
        "collector-emitter voltage",
        "collector-emitter voltage"
    )

    score_uppercase = calculate_relevance_score(
        "collector-emitter voltage",
        "COLLECTOR-EMITTER VOLTAGE"
    )

    assert score_lowercase == score_uppercase


def test_exact_phrase_bonus_does_not_break_term_matching():
    """
    V2.1.1:

    A chunk should still receive a relevance score when
    only individual query terms match.
    """

    score = calculate_relevance_score(
        "collector-emitter voltage",
        "collector voltage information"
    )

    assert score > 0


# ============================================================
# V2.1.2 — TECHNICAL TERM WEIGHTING
# ============================================================

def test_technical_term_match_scores_higher():
    """
    V2.1.2:

    A chunk containing a technical term from the query
    should receive a higher score than a chunk containing
    only generic words from the same query.

    Example:

    Query:
        What is the VCEO voltage?

    Technical chunk:
        VCEO collector-emitter voltage open base -50 V

    Generic chunk:
        What is the voltage information for this component?
    """

    query = "What is the VCEO voltage?"

    technical_score = calculate_relevance_score(
        query,
        "VCEO collector-emitter voltage open base -50 V"
    )

    generic_score = calculate_relevance_score(
        query,
        "What is the voltage information for this component?"
    )

    assert technical_score > generic_score


def test_technical_term_receives_additional_weight():
    """
    V2.1.2:

    Verify that a technical term contributes more than
    a normal matching term.

    The exact numerical weight is intentionally not fixed
    here. The test only verifies that the technical term
    provides an additional advantage.
    """

    technical_score = calculate_relevance_score(
        "VCEO",
        "VCEO collector-emitter voltage"
    )

    generic_score = calculate_relevance_score(
        "voltage",
        "VCEO collector-emitter voltage"
    )

    assert technical_score > generic_score


def test_multiple_technical_terms_receive_weight():
    """
    V2.1.2:

    A chunk matching multiple technical terms should score
    higher than a chunk matching only one technical term.
    """

    query = "VCEO hFE"

    multiple_technical_terms_score = calculate_relevance_score(
        query,
        "VCEO collector-emitter voltage hFE 70"
    )

    single_technical_term_score = calculate_relevance_score(
        query,
        "VCEO collector-emitter voltage"
    )

    assert multiple_technical_terms_score > single_technical_term_score


def test_technical_term_weighting_preserves_normal_terms():
    """
    V2.1.2:

    Technical weighting must not remove the contribution
    of normal matching terms.
    """

    score_with_generic_term = calculate_relevance_score(
        "VCEO voltage",
        "VCEO voltage"
    )

    score_without_generic_term = calculate_relevance_score(
        "VCEO",
        "VCEO voltage"
    )

    assert score_with_generic_term > score_without_generic_term


def test_technical_terms_are_case_insensitive_for_matching():
    """
    V2.1.2:

    Technical term matching should remain case-insensitive.

    VCEO and vceo should refer to the same token for
    retrieval purposes.
    """

    uppercase_score = calculate_relevance_score(
        "VCEO",
        "VCEO collector-emitter voltage"
    )

    lowercase_score = calculate_relevance_score(
        "vceo",
        "VCEO collector-emitter voltage"
    )

    assert uppercase_score == lowercase_score


# ============================================================
# V2.1.2 — RANKING
# ============================================================

def test_technical_chunk_is_ranked_above_generic_chunk():
    """
    V2.1.2:

    A chunk containing the technical query term should
    rank above a generic chunk containing only common terms.
    """

    chunks = [
        {
            "chunk_id": 1,
            "page": 1,
            "text": (
                "What is the voltage information "
                "for this component?"
            ),
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
        }
    ]

    results = retrieve_relevant_chunks(
        "What is the VCEO voltage?",
        chunks,
        top_k=2
    )

    assert len(results) == 2

    assert results[0]["chunk"]["chunk_id"] == 2
    assert results[1]["chunk"]["chunk_id"] == 1

    assert results[0]["score"] > results[1]["score"]


def test_retriever_preserves_score_order():
    """
    V2.1.2:

    Retrieved results must be ordered from highest
    relevance score to lowest relevance score.
    """

    chunks = [
        {
            "chunk_id": 1,
            "page": 1,
            "text": "voltage information",
            "source": PDF_PATH
        },
        {
            "chunk_id": 2,
            "page": 2,
            "text": "VCEO voltage",
            "source": PDF_PATH
        },
        {
            "chunk_id": 3,
            "page": 3,
            "text": "VCEO collector-emitter voltage",
            "source": PDF_PATH
        }
    ]

    results = retrieve_relevant_chunks(
        "What is the VCEO voltage?",
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


def test_retriever_technical_term_ranking_with_real_document():
    """
    V2.1.2:

    Verify that a query containing a technical term
    retrieves the corresponding technical document chunk.
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the VCEO voltage?",
        chunks
    )

    assert results

    top_chunk = results[0]["chunk"]

    assert "VCEO" in top_chunk["text"]


# ============================================================
# V2.1.2 — COMPATIBILITY
# ============================================================

def test_existing_output_current_retrieval_still_works():
    """
    V2.1.2 compatibility test:

    Technical weighting must not break the existing
    output-current retrieval behavior.
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the maximum output current?",
        chunks
    )

    assert results

    combined_text = " ".join(
        result["chunk"]["text"]
        for result in results
    )

    assert "output current" in combined_text.lower()
    assert "500 mA" in combined_text


def test_existing_collector_emitter_retrieval_still_works():
    """
    V2.1.2 compatibility test:

    Technical weighting must not break the existing
    collector-emitter retrieval behavior.
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the maximum collector-emitter voltage?",
        chunks
    )

    assert results

    combined_text = " ".join(
        result["chunk"]["text"]
        for result in results
    )

    assert "VCEO" in combined_text
    assert "-50" in combined_text


def test_existing_input_voltage_retrieval_still_works():
    """
    V2.1.2 compatibility test:

    Technical weighting must not break the existing
    input-voltage retrieval behavior.
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the maximum input voltage?",
        chunks
    )

    assert results

    combined_text = " ".join(
        result["chunk"]["text"]
        for result in results
    )

    assert "input voltage" in combined_text.lower()
    assert "5" in combined_text
    assert "-10" in combined_text


def test_existing_dc_current_gain_retrieval_still_works():
    """
    V2.1.2 compatibility test:

    Technical weighting must not break the existing
    DC-current-gain retrieval behavior.
    """

    chunks = load_chunks()

    results = retrieve_relevant_chunks(
        "What is the DC current gain?",
        chunks
    )

    assert results

    combined_text = " ".join(
        result["chunk"]["text"]
        for result in results
    )

    assert "hFE" in combined_text
    assert "DC current gain" in combined_text
    assert "70" in combined_text


# ============================================================
# STOPWORD-AWARE SCORING
# ============================================================

def test_is_stopword_recognizes_common_english_words():
    assert is_stopword("the")
    assert is_stopword("is")
    assert is_stopword("what")


def test_is_stopword_recognizes_common_spanish_words():
    assert is_stopword("es")
    assert is_stopword("mi")
    assert is_stopword("cual")


def test_is_stopword_rejects_technical_and_content_words():
    assert not is_stopword("voltage")
    assert not is_stopword("vceo")
    assert not is_stopword("collector")


def test_stopwords_do_not_contribute_to_relevance_score():
    """
    A query that ONLY matches via stopwords must score 0 - this
    is the real "cual es mi nombre?" vs. a Spanish document case
    observed during the V3.4 end-to-end test.
    """
    query = "cual es mi nombre"
    text = "el tallo es la parte principal de la planta"

    score = calculate_relevance_score(query, text)

    assert score == 0


def test_stopwords_do_not_change_score_of_a_real_technical_match():
    """
    A real technical question scores the same with or without its
    stopwords ("what", "is", "the") - they neither add to the
    score nor subtract from it, the meaningful terms still carry
    it entirely.
    """
    query_with_stopwords = "what is the maximum collector-emitter voltage"
    query_without_stopwords = "maximum collector-emitter voltage"
    text = "VCEO collector-emitter voltage open base -50 V"

    score_with = calculate_relevance_score(query_with_stopwords, text)
    score_without = calculate_relevance_score(query_without_stopwords, text)

    assert score_with == score_without


def test_confidence_is_zero_when_only_stopwords_match():
    query = "cual es mi nombre"
    text = "el tallo es la parte principal de la planta"

    score = calculate_relevance_score(query, text)
    confidence = calculate_confidence(score, query, text)

    assert confidence == 0.0
    