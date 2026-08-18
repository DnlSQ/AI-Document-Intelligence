from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.retriever import retrieve_relevant_chunks


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
    