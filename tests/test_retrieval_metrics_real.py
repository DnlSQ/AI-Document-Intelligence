"""
Tests for V2.3 Retrieval Metrics, run against the real
sample.pdf.

Unlike test_evaluation.py's binary pass/fail (which allows a
"pass" built from keywords scattered across several chunks),
these metrics require a SINGLE chunk to contain the full answer
on its own. Some precision/recall numbers may look "imperfect"
even when the system works correctly - see the notes on each
assertion for why.

Since the 4->16 dataset expansion (2026-08-25), EVALUATION_DATASET
entries carry a "style" tag ("literal" or "paraphrase"). The
rank-1 invariant is checked separately per style below, because
it holds for literal questions but not for paraphrased ones - see
rag-v3-progress.md's Comparative Retrieval Evaluation and
Confidence Gate Risk Check for the full measurement and the
reasoning for why that's an accepted trade-off, not a regression.
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


def test_relevant_chunk_always_ranks_first_for_literal_questions():
    """
    For every LITERAL question (matching the datasheet's own
    vocabulary), the first relevant chunk found must be at rank 1
    (reciprocal_rank == 1.0). This invariant has held since the
    original 4-question dataset and still holds across all 12
    literal questions after the 4->16 dataset expansion
    (2026-08-25) - see rag-v3-progress.md.

    Paraphrased questions are checked separately, with a looser
    bar - see test_relevant_chunk_is_found_for_paraphrased_questions
    below. rag-v3-progress.md's "Confidence Gate Risk Check" is why
    that looser bar is an accepted, documented trade-off rather
    than an unnoticed regression: it confirmed a paraphrased
    question never gets falsely rejected by the no-answer gate
    just because its relevant chunk isn't ranked first.
    """

    chunks = load_chunks()
    literal_cases = [
        case for case in EVALUATION_DATASET if case.get("style") == "literal"
    ]

    report = calculate_retrieval_metrics(literal_cases, chunks, top_k=3)

    not_ranked_first = [
        result for result in report["results"]
        if result["reciprocal_rank"] != 1.0
    ]

    assert not not_ranked_first, (
        f"Some literal questions did not rank a relevant chunk first: {not_ranked_first}"
    )


def test_relevant_chunk_is_found_for_paraphrased_questions():
    """
    Paraphrased questions (natural-language wording, not the
    datasheet's own vocabulary) are held to a looser bar: the
    relevant chunk must be FOUND somewhere in top_k
    (reciprocal_rank > 0), not necessarily ranked first.

    This is a measured, accepted trade-off (see
    rag-v3-progress.md's Comparative Retrieval Evaluation and
    Confidence Gate Risk Check) - lexical alone doesn't always
    rank the right chunk #1 on paraphrases, but it's still found,
    and that was confirmed to never cause a real answerable
    question to be rejected by the no-answer gate.
    """

    chunks = load_chunks()
    paraphrase_cases = [
        case for case in EVALUATION_DATASET if case.get("style") == "paraphrase"
    ]

    report = calculate_retrieval_metrics(paraphrase_cases, chunks, top_k=3)

    not_found_at_all = [
        result for result in report["results"]
        if result["reciprocal_rank"] == 0.0
    ]

    assert not not_found_at_all, (
        f"Some paraphrased questions found NO relevant chunk at all "
        f"in top_k: {not_found_at_all}"
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
    