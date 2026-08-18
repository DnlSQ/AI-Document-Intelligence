from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks


PDF_PATH = "data/documents/sample.pdf"


def load_chunks():
    """
    Load sample.pdf, clean its text and create document chunks.
    """

    pages = extract_text_from_pdf(PDF_PATH)

    for page in pages:
        page["text"] = clean_text(page["text"])

    chunks = create_document_chunks(
        pages,
        chunk_size=1000,
        chunk_overlap=150,
        source=PDF_PATH
    )

    return chunks


def test_chunks_are_created():
    """
    Verify that the PDF produces at least one chunk.
    """

    chunks = load_chunks()

    assert len(chunks) > 0


def test_chunk_metadata():
    """
    Verify that every chunk contains the required metadata.
    """

    chunks = load_chunks()

    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "page" in chunk
        assert "text" in chunk
        assert "source" in chunk

        assert isinstance(chunk["chunk_id"], int)
        assert isinstance(chunk["page"], int)

        assert chunk["chunk_id"] > 0
        assert chunk["page"] > 0

        assert isinstance(chunk["text"], str)
        assert chunk["text"].strip()

        assert chunk["source"] == PDF_PATH


def test_question_1_maximum_output_current():
    """
    Question 1:

    What is the maximum output current of the PDTB113ZT?

    Expected answer:
    500 mA
    """

    chunks = load_chunks()

    relevant_chunks = []

    for chunk in chunks:
        text = chunk["text"].lower()

        if (
            "output current" in text
            and "500 ma" in text
        ):
            relevant_chunks.append(chunk)

    assert relevant_chunks, (
        "No chunk containing "
        "'output current' and '500 mA' was found."
    )

    combined_text = " ".join(
        chunk["text"]
        for chunk in relevant_chunks
    )

    assert "500 mA" in combined_text


def test_question_2_maximum_collector_emitter_voltage():
    """
    Question 2:

    What is the maximum collector-emitter voltage?

    Expected answer:
    VCEO = -50 V
    """

    chunks = load_chunks()

    relevant_chunks = []

    for chunk in chunks:
        text = chunk["text"]

        if (
            "VCEO" in text
            and "-50" in text
        ):
            relevant_chunks.append(chunk)

    assert relevant_chunks, (
        "No chunk containing "
        "'VCEO' and '-50 V' was found."
    )

    combined_text = " ".join(
        chunk["text"]
        for chunk in relevant_chunks
    )

    assert "VCEO" in combined_text
    assert "-50" in combined_text

def test_question_3_maximum_input_voltage():
    """
    Question 3:

    What is the maximum input voltage of the PDTB113ZT?

    Expected answer:
    Positive: 5 V
    Negative: -10 V
    """

    chunks = load_chunks()

    relevant_chunks = []

    for chunk in chunks:
        text = chunk["text"]

        if (
            "input voltage" in text
            and "5" in text
            and "-10" in text
        ):
            relevant_chunks.append(chunk)

    assert relevant_chunks, (
        "No chunk containing "
        "'input voltage', '5 V' and '-10 V' was found."
    )

    combined_text = " ".join(
        chunk["text"]
        for chunk in relevant_chunks
    )

    assert "input voltage" in combined_text
    assert "5" in combined_text
    assert "-10" in combined_text
def test_question_4_dc_current_gain():
    """
    Question 4:

    What is the typical DC current gain (hFE) of the PDTB113ZT?

    Expected answer:
    hFE = 70 minimum
    """

    chunks = load_chunks()

    relevant_chunks = []

    for chunk in chunks:
        text = chunk["text"]

        if (
            "hFE" in text
            and "DC current gain" in text
            and "70" in text
        ):
            relevant_chunks.append(chunk)

    assert relevant_chunks, (
        "No chunk containing "
        "'hFE', 'DC current gain' and '70' was found."
    )

    combined_text = " ".join(
        chunk["text"]
        for chunk in relevant_chunks
    )

    assert "hFE" in combined_text
    assert "DC current gain" in combined_text
    assert "70" in combined_text
