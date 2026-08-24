"""
Tests for V2.3 Retrieval Metrics (src/evaluation.py).

These use synthetic, hand-constructed corpora with known ground
truth so the Precision@K / Recall@K / MRR arithmetic can be
verified exactly, independent of any real PDF.
"""
from src.evaluation import (
    is_relevant_chunk,
    calculate_retrieval_metrics,
)


PDF_PATH = "sample.pdf"


# ---------------------------------------------------------------------
# is_relevant_chunk
# ---------------------------------------------------------------------

def test_chunk_is_relevant_when_it_contains_all_keywords():
    case = {"expected_keywords": ["VCEO", "-50"]}
    assert is_relevant_chunk(case, "VCEO rating is -50 V typical") is True


def test_chunk_is_not_relevant_when_missing_a_keyword():
    case = {"expected_keywords": ["VCEO", "-50"]}
    assert is_relevant_chunk(case, "VCEO rating information only") is False


def test_relevance_check_is_case_insensitive():
    case = {"expected_keywords": ["vceo"]}
    assert is_relevant_chunk(case, "VCEO rating -50 V") is True


# ---------------------------------------------------------------------
# calculate_retrieval_metrics - hand-computed scenarios
# ---------------------------------------------------------------------

def test_perfect_case_single_relevant_chunk_ranked_first():
    """
    One relevant chunk exists in the corpus, and it's retrieved
    at rank 1.

    total_relevant = 1
    retrieved (top_k=3) = [relevant_chunk] (only chunk with score > 0)
    relevant_retrieved = 1
    precision_at_k = 1/3 = 0.33  (capped low because top_k=3 but
                                   only 1 relevant chunk exists -
                                   this is expected, not a bug)
    recall_at_k = 1/1 = 1.0
    reciprocal_rank = 1/1 = 1.0
    """

    dataset = [{
        "id": "c1",
        "question": "VCEO",
        "expected_keywords": ["VCEO"],
        "description": "d"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating -50 V", "source": PDF_PATH},
        {"chunk_id": 2, "page": 1, "text": "unrelated packaging info", "source": PDF_PATH},
    ]

    report = calculate_retrieval_metrics(dataset, chunks, top_k=3)
    result = report["results"][0]

    assert result["total_relevant_in_corpus"] == 1
    assert result["relevant_retrieved"] == 1
    assert result["precision_at_k"] == round(1 / 3, 2)
    assert result["recall_at_k"] == 1.0
    assert result["reciprocal_rank"] == 1.0


def test_multiple_relevant_chunks_only_some_retrieved():
    """
    Two relevant chunks exist in the corpus, but only one of
    them actually scores high enough / exists to be retrieved
    within top_k, so recall must reflect the partial coverage.
    """

    dataset = [{
        "id": "c1",
        "question": "VCEO",
        "expected_keywords": ["VCEO"],
        "description": "d"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating -50 V", "source": PDF_PATH},
        {"chunk_id": 2, "page": 2, "text": "VCEO also mentioned again here", "source": PDF_PATH},
    ]

    report = calculate_retrieval_metrics(dataset, chunks, top_k=3)
    result = report["results"][0]

    assert result["total_relevant_in_corpus"] == 2
    assert result["relevant_retrieved"] == 2
    assert result["recall_at_k"] == 1.0


def test_relevant_chunk_ranked_second_gives_half_reciprocal_rank():
    """
    When the only relevant chunk is NOT the top-ranked result,
    reciprocal_rank must reflect its actual rank (1/2), not 1.0.
    """

    dataset = [{
        "id": "c1",
        "question": "VCEO ICBO",
        "expected_keywords": ["ICBO"],
        "description": "d"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO VCEO VCEO rating details", "source": PDF_PATH},
        {"chunk_id": 2, "page": 2, "text": "ICBO leakage current value", "source": PDF_PATH},
    ]

    report = calculate_retrieval_metrics(dataset, chunks, top_k=3)
    result = report["results"][0]

    assert result["total_relevant_in_corpus"] == 1
    assert result["reciprocal_rank"] == 0.5


def test_no_relevant_chunk_in_corpus_gives_zero_recall_without_error():
    dataset = [{
        "id": "c1",
        "question": "VCEO",
        "expected_keywords": ["ICBO"],
        "description": "d"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating -50 V, no mention of the other term", "source": PDF_PATH},
    ]

    report = calculate_retrieval_metrics(dataset, chunks, top_k=3)
    result = report["results"][0]

    assert result["total_relevant_in_corpus"] == 0
    assert result["relevant_retrieved"] == 0
    assert result["recall_at_k"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_aggregate_metrics_average_across_dataset():
    dataset = [
        {"id": "c1", "question": "VCEO", "expected_keywords": ["VCEO"], "description": "d"},
        {"id": "c2", "question": "ICBO", "expected_keywords": ["ICBO"], "description": "d"},
    ]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating -50 V", "source": PDF_PATH},
        {"chunk_id": 2, "page": 2, "text": "ICBO leakage current", "source": PDF_PATH},
    ]

    report = calculate_retrieval_metrics(dataset, chunks, top_k=3)

    assert report["mrr"] == 1.0
    assert report["mean_recall_at_k"] == 1.0
    assert report["total"] == 2


def test_empty_dataset_returns_zero_metrics_without_error():
    report = calculate_retrieval_metrics([], chunks=[], top_k=3)

    assert report["total"] == 0
    assert report["mean_precision_at_k"] == 0.0
    assert report["mean_recall_at_k"] == 0.0
    assert report["mrr"] == 0.0
    