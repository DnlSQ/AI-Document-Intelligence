"""
Tests for the V2.2 Evaluation Framework logic (src/evaluation.py).

These tests use synthetic chunks so they can run in any
environment without the real PDF or its ingestion pipeline.
The real-document regression check lives in test_evaluation.py.
"""
from src.evaluation import evaluate_retrieval


PDF_PATH = "sample.pdf"


def test_case_passes_when_all_keywords_present():
    dataset = [{
        "id": "case1",
        "question": "What is the VCEO rating?",
        "expected_keywords": ["VCEO", "-50"],
        "description": "test case"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating is -50 V typical", "source": PDF_PATH}
    ]

    report = evaluate_retrieval(dataset, chunks, top_k=3)

    assert report["total"] == 1
    assert report["passed"] == 1
    assert report["accuracy"] == 1.0
    assert report["results"][0]["passed"] is True
    assert report["results"][0]["missing_keywords"] == []


def test_case_fails_when_keyword_missing():
    dataset = [{
        "id": "case1",
        "question": "What is the ICBO rating?",
        "expected_keywords": ["ICBO"],
        "description": "test case"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "unrelated content about packaging", "source": PDF_PATH}
    ]

    report = evaluate_retrieval(dataset, chunks, top_k=3)

    assert report["total"] == 1
    assert report["passed"] == 0
    assert report["accuracy"] == 0.0
    assert report["results"][0]["passed"] is False
    assert "ICBO" in report["results"][0]["missing_keywords"]


def test_keyword_matching_is_case_insensitive():
    """
    Unlike the strict, case-sensitive pytest checks elsewhere in
    the project, this framework's keyword matching is
    intentionally case-insensitive: it's a diagnostic tool, not
    a replacement for the sharp unit tests.
    """
    dataset = [{
        "id": "case1",
        "question": "voltage",
        "expected_keywords": ["VOLTAGE"],
        "description": "test case"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "the voltage rating is listed here", "source": PDF_PATH}
    ]

    report = evaluate_retrieval(dataset, chunks, top_k=3)

    assert report["results"][0]["passed"] is True


def test_accuracy_computed_across_mixed_pass_fail_cases():
    dataset = [
        {"id": "a", "question": "VCEO", "expected_keywords": ["VCEO"], "description": "d"},
        {"id": "b", "question": "ICBO", "expected_keywords": ["ICBO"], "description": "d"},
    ]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating here", "source": PDF_PATH},
    ]

    report = evaluate_retrieval(dataset, chunks, top_k=3)

    assert report["total"] == 2
    assert report["passed"] == 1
    assert report["accuracy"] == 0.5


def test_empty_dataset_returns_zero_accuracy_without_error():
    report = evaluate_retrieval([], chunks=[], top_k=3)

    assert report["total"] == 0
    assert report["passed"] == 0
    assert report["accuracy"] == 0.0


def test_case_with_no_retrieved_chunks_reports_zero_score_and_confidence():
    dataset = [{
        "id": "case1",
        "question": "VCEO",
        "expected_keywords": ["VCEO"],
        "description": "test case"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "completely unrelated text", "source": PDF_PATH}
    ]

    report = evaluate_retrieval(dataset, chunks, top_k=3)

    result = report["results"][0]
    assert result["top_score"] == 0
    assert result["top_confidence"] == 0.0


def test_result_includes_diagnostic_fields():
    dataset = [{
        "id": "case1",
        "question": "VCEO",
        "expected_keywords": ["VCEO"],
        "description": "test case"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating -50 V", "source": PDF_PATH}
    ]

    report = evaluate_retrieval(dataset, chunks, top_k=3)
    result = report["results"][0]

    assert "top_score" in result
    assert "top_confidence" in result
    assert result["top_score"] > 0
    assert 0.0 <= result["top_confidence"] <= 1.0
    