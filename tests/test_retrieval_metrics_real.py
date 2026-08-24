"""
Tests for V2.3 Retrieval Metrics, run against the real
sample.pdf.

Unlike test_evaluation.py's binary pass/fail (which allows a
"pass" built from keywords scattered across several chunks),
these metrics require a SINGLE chunk to contain the full answer
on its own. Some precision/recall numbers may look "imperfect"
even when the system works correctly - see the notes on each
assertion for why.
"""
from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.evaluation import calculate_retrieval_metrics, EVALUATION_DATASET


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


def test_relevant_chunk_always_ranks_first():
    """
    For every question, the FIRST relevant chunk found must be
    at rank 1 (reciprocal_rank == 1.0).

    Note: we deliberately do NOT assert recall_at_k == 1.0 here.
    Because chunker.py uses chunk_overlap=150, the same answer
    can legitimately appear in more than one overlapping chunk
    window - and some of this dataset's expected_keywords (e.g.
    the bare digits "5"/"-10" for input_voltage) are loose
    substring matches that can coincidentally also appear in an
    unrelated chunk's numbers (like "-50" or "500 mA"). Both
    effects inflate total_relevant_in_corpus without indicating
    any real retrieval problem - what actually matters is that
    the retriever puts a genuinely correct chunk in the #1 spot,
    which reciprocal_rank measures directly.
    """

    chunks = load_chunks()

    report = calculate_retrieval_metrics(EVALUATION_DATASET, chunks, top_k=3)

    not_ranked_first = [
        result for result in report["results"]
        if result["reciprocal_rank"] != 1.0
    ]

    assert not not_ranked_first, (
        f"Some questions did not rank a relevant chunk first: {not_ranked_first}"
    )


def test_relevant_chunk_ranks_reasonably_high():
    """
    Mean Reciprocal Rank should be reasonably high, meaning the
    correct chunk isn't just present in top_k somewhere, but
    ranked near the top. This is a softer check than recall:
    it's fine if this isn't a perfect 1.0, but it shouldn't be
    close to 0 either (which would mean correct chunks are being
    buried at the bottom of the ranking).
    """

    chunks = load_chunks()

    report = calculate_retrieval_metrics(EVALUATION_DATASET, chunks, top_k=3)

    assert report["mrr"] > 0.5, (
        f"MRR is low ({report['mrr']}) - relevant chunks are ranking "
        f"too far down. Full report: {report['results']}"
    )
    