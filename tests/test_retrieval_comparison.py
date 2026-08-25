"""
Tests for comparative retrieval evaluation (src/evaluation.py):
comparing lexical, semantic, and hybrid retrieval on the same
golden dataset via calculate_retrieval_metrics / compare_retrieval_methods.

Synthetic, fully-mocked data only - no real PDF, no real embedding
model, no real ChromaDB. This feature's job is to wire together
the three retrieval strategies and reuse the existing
Precision/Recall/MRR arithmetic identically for each, so these
tests mock retrieval itself and verify the wiring is correct - the
same boundary-mocking pattern test_hybrid_retrieval.py uses for
fusion logic. The real, non-mocked side-by-side numbers against
sample.pdf live in tests/test_retrieval_comparison_manual.py.
"""
from src.evaluation import calculate_retrieval_metrics, compare_retrieval_methods


PDF_PATH = "sample.pdf"


def make_result(chunk_id, text):
    return {"chunk": {"chunk_id": chunk_id, "page": 1, "text": text, "source": PDF_PATH}}


# ---------------------------------------------------------------------
# calculate_retrieval_metrics - retrieve_fn override
# ---------------------------------------------------------------------

def test_calculate_retrieval_metrics_defaults_to_lexical_retrieval():
    """
    Backward-compatibility regression: calling without retrieve_fn
    must behave exactly as before this feature existed.
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
    assert result["reciprocal_rank"] == 1.0


def test_calculate_retrieval_metrics_uses_custom_retrieve_fn_instead_of_lexical():
    """
    A custom retrieve_fn must actually be used - not just accepted
    and ignored. Here the chunk that WOULD match lexically is
    deliberately excluded by the fake retrieve_fn, so the metrics
    must reflect finding NOTHING, proving lexical scoring was
    bypassed entirely.
    """
    dataset = [{
        "id": "c1",
        "question": "VCEO",
        "expected_keywords": ["VCEO"],
        "description": "d"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating -50 V", "source": PDF_PATH},
    ]

    def empty_retrieve_fn(question, top_k):
        return []

    report = calculate_retrieval_metrics(
        dataset, chunks, top_k=3, retrieve_fn=empty_retrieve_fn
    )
    result = report["results"][0]

    # Ground truth still finds the relevant chunk in the corpus...
    assert result["total_relevant_in_corpus"] == 1
    # ...but retrieve_fn "found" nothing, so nothing counts as retrieved.
    assert result["relevant_retrieved"] == 0
    assert result["recall_at_k"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_calculate_retrieval_metrics_works_with_result_shape_lacking_score_or_confidence():
    """
    hybrid_retrieve's results carry rrf_score/lexical_rank/
    semantic_rank instead of score/confidence. Metrics only ever
    read result["chunk"]["text"], so a retrieve_fn returning this
    leaner shape must work without error.
    """
    dataset = [{
        "id": "c1",
        "question": "VCEO",
        "expected_keywords": ["VCEO"],
        "description": "d"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating -50 V", "source": PDF_PATH},
    ]

    def hybrid_shaped_retrieve_fn(question, top_k):
        return [{
            "chunk": chunks[0],
            "rrf_score": 0.03,
            "lexical_rank": 1,
            "semantic_rank": None,
            "lexical_confidence": 0.8,
            "semantic_confidence": 0.0
        }]

    report = calculate_retrieval_metrics(
        dataset, chunks, top_k=3, retrieve_fn=hybrid_shaped_retrieve_fn
    )
    result = report["results"][0]

    assert result["relevant_retrieved"] == 1
    assert result["reciprocal_rank"] == 1.0


# ---------------------------------------------------------------------
# compare_retrieval_methods
# ---------------------------------------------------------------------

def test_compare_retrieval_methods_returns_a_report_per_method(monkeypatch):
    dataset = [{
        "id": "c1",
        "question": "VCEO",
        "expected_keywords": ["VCEO"],
        "description": "d"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating -50 V", "source": PDF_PATH},
    ]

    monkeypatch.setattr(
        "src.evaluation.retrieve_relevant_chunks",
        lambda question, chunks_arg, top_k: [make_result(1, "VCEO rating -50 V")]
    )
    monkeypatch.setattr(
        "src.evaluation.semantic_search",
        lambda question, top_k, collection: []
    )
    monkeypatch.setattr(
        "src.evaluation.hybrid_retrieve",
        lambda question, chunks_arg, collection, top_k: [make_result(1, "VCEO rating -50 V")]
    )

    comparison = compare_retrieval_methods(dataset, chunks, collection=None, top_k=3)

    assert set(comparison.keys()) == {"lexical", "semantic", "hybrid"}

    for report in comparison.values():
        assert "mean_precision_at_k" in report
        assert "mean_recall_at_k" in report
        assert "mrr" in report


def test_compare_retrieval_methods_reflects_each_methods_own_results(monkeypatch):
    """
    The three reports must genuinely come from their own mocked
    function, not all collapse to the same one - lexical "finds"
    the answer here, semantic and hybrid do not, so their MRR
    must differ accordingly.
    """
    dataset = [{
        "id": "c1",
        "question": "VCEO",
        "expected_keywords": ["VCEO"],
        "description": "d"
    }]
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "VCEO rating -50 V", "source": PDF_PATH},
    ]

    monkeypatch.setattr(
        "src.evaluation.retrieve_relevant_chunks",
        lambda question, chunks_arg, top_k: [make_result(1, "VCEO rating -50 V")]
    )
    monkeypatch.setattr(
        "src.evaluation.semantic_search",
        lambda question, top_k, collection: []
    )
    monkeypatch.setattr(
        "src.evaluation.hybrid_retrieve",
        lambda question, chunks_arg, collection, top_k: []
    )

    comparison = compare_retrieval_methods(dataset, chunks, collection=None, top_k=3)

    assert comparison["lexical"]["mrr"] == 1.0
    assert comparison["semantic"]["mrr"] == 0.0
    assert comparison["hybrid"]["mrr"] == 0.0
    