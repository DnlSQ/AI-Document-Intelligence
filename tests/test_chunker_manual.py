from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks


PDF_PATH = "data/documents/sample.pdf"


def load_chunks():
    pages = extract_text_from_pdf(PDF_PATH)

    for page in pages:
        page["text"] = clean_text(page["text"])

    chunks = create_document_chunks(
        pages,
        chunk_size=1000,
        chunk_overlap=150
    )

    return chunks


def test_chunks_are_created():
    chunks = load_chunks()

    assert len(chunks) > 0


def test_chunk_metadata():
    chunks = load_chunks()

    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "page" in chunk
        assert "text" in chunk
        assert "source" in chunk


def test_question_1_maximum_output_current():
    chunks = load_chunks()

    matching_chunks = [
        chunk
        for chunk in chunks
        if "output current" in chunk["text"]
    ]

    assert matching_chunks, "No chunk contains the output current information."

    combined_text = "\n".join(
        chunk["text"]
        for chunk in matching_chunks
    )

    assert "500 mA" in combined_text


def test_question_2_maximum_collector_emitter_voltage():
    chunks = load_chunks()

    matching_chunks = [
        chunk
        for chunk in chunks
        if "VCEO" in chunk["text"]
    ]

    assert matching_chunks, "No chunk contains VCEO information."

    combined_text = "\n".join(
        chunk["text"]
        for chunk in matching_chunks
    )

    assert "-50" in combined_text


def test_question_3_maximum_input_voltage():
    chunks = load_chunks()

    matching_chunks = [
        chunk
        for chunk in chunks
        if "VI" in chunk["text"]
        and "input voltage" in chunk["text"]
    ]

    assert matching_chunks, "No chunk contains the input voltage information."

    combined_text = "\n".join(
        chunk["text"]
        for chunk in matching_chunks
    )

    assert "5" in combined_text
    assert "-10" in combined_text
    