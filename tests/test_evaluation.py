"""
Tests for the V2.2 Evaluation Framework, run against the real
sample.pdf.

This acts as a regression safety net: if a future change to the
retriever breaks one of the four known-good real-world answers,
this test fails immediately with a clear report of which case
broke and why - instead of relying on someone remembering to
manually re-check each question by hand.
"""
from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.evaluation import evaluate_retrieval, EVALUATION_DATASET


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


def test_golden_dataset_achieves_full_accuracy_on_real_document():
    """
    Regression safety net: the retriever must still correctly
    surface all four known real-world answers.
    """

    chunks = load_chunks()

    report = evaluate_retrieval(EVALUATION_DATASET, chunks, top_k=3)

    failed = [result for result in report["results"] if not result["passed"]]

    assert not failed, f"Evaluation regressions found: {failed}"
    assert report["accuracy"] == 1.0


def test_evaluation_report_includes_score_and_confidence_per_case():
    """
    Verify that each per-question result carries diagnostic
    score/confidence info, useful for spotting weak matches even
    when the keyword check technically passes.
    """

    chunks = load_chunks()

    report = evaluate_retrieval(EVALUATION_DATASET, chunks, top_k=3)

    for result in report["results"]:
        assert "top_score" in result
        assert "top_confidence" in result
        assert result["top_score"] >= 0
        assert 0.0 <= result["top_confidence"] <= 1.0