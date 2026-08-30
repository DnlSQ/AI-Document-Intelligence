"""
Golden-dataset regression tests for NE555N.pdf (RAG v6.3).

Kept in a dedicated file, mirroring test_evaluation.py and
test_retrieval_metrics_real.py's existing pattern for sample.pdf,
rather than merged into EVALUATION_DATASET - see the comment above
NE555N_EVALUATION_DATASET in src/evaluation.py for why.

These two questions are the exact ones RAG v5.5 found broken and
RAG v6.2 fixed. Before v6.2, "turn off time" would have failed here
too (the reconstructed table facts didn't exist yet) - this file is
what turns that into a permanent regression check instead of a
manual browser smoke test someone has to remember to repeat.
"""
from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.evaluation import (
    evaluate_retrieval,
    calculate_retrieval_metrics,
    NE555N_EVALUATION_DATASET,
)


PDF_PATH = "data/documents/NE555N.pdf"


def load_chunks():
    """
    Load NE555N.pdf, clean its text and create document chunks -
    identical pipeline to test_evaluation.py's load_chunks(), just
    pointed at a different real PDF.
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


def test_ne555n_golden_dataset_achieves_full_accuracy():
    """
    Regression safety net for RAG v6.2: both questions must still
    be answerable from the real document's retrieved chunks.
    """
    chunks = load_chunks()

    report = evaluate_retrieval(NE555N_EVALUATION_DATASET, chunks, top_k=3)

    failed = [result for result in report["results"] if not result["passed"]]

    assert not failed, f"NE555N.pdf evaluation regressions found: {failed}"
    assert report["accuracy"] == 1.0


def test_ne555n_relevant_chunk_is_found_for_both_questions():
    """
    Stricter than the accuracy check above: calculate_retrieval_metrics
    requires ALL expected_keywords in a SINGLE chunk (see
    is_relevant_chunk in evaluation.py), not scattered across the
    combined top_k text. A reciprocal_rank of 0.0 here would mean no
    single chunk actually holds the complete fact together - exactly
    the kind of gap v5.5/v6.2 were built to catch.
    """
    chunks = load_chunks()

    report = calculate_retrieval_metrics(NE555N_EVALUATION_DATASET, chunks, top_k=3)

    not_found_at_all = [
        result for result in report["results"]
        if result["reciprocal_rank"] == 0.0
    ]

    assert not not_found_at_all, (
        f"Some NE555N.pdf questions found no single chunk containing "
        f"the full answer: {not_found_at_all}"
    )
    