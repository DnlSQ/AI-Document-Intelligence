"""
V2.2 Evaluation Framework.

Provides a small "golden dataset" of question / expected-content
pairs and a runner that checks the retriever's real output
against it, producing a structured pass/fail report.

This complements (does not replace) the individual pytest
question tests in test_retriever.py: those catch regressions
with sharp, specific, case-sensitive assertions; this framework
gives a single summary view and makes it easy to grow the
evaluation set over time without writing a new test function for
every question.
"""

from src.retriever import retrieve_relevant_chunks


# Each entry describes one evaluation case:
#
#   id: short unique identifier
#   question: the user question to ask
#   expected_keywords: strings that must ALL appear somewhere in
#       the combined text of the retrieved top_k chunks for the
#       case to pass (matched case-insensitively)
#   description: human-readable summary, for reporting
EVALUATION_DATASET = [
    {
        "id": "output_current",
        "question": "What is the maximum output current of the PDTB113ZT?",
        "expected_keywords": ["output current", "500 mA"],
        "description": "Maximum output current",
    },
    {
        "id": "collector_emitter_voltage",
        "question": "What is the maximum collector-emitter voltage?",
        "expected_keywords": ["VCEO", "-50"],
        "description": "Maximum collector-emitter voltage",
    },
    {
        "id": "input_voltage",
        "question": "What is the maximum input voltage?",
        "expected_keywords": ["input voltage", "5", "-10"],
        "description": "Maximum input voltage",
    },
    {
        "id": "dc_current_gain",
        "question": "What is the DC current gain?",
        "expected_keywords": ["hFE", "DC current gain", "70"],
        "description": "DC current gain (hFE)",
    },
]


def evaluate_retrieval(dataset, chunks, top_k=3):
    """
    Run every question in the dataset through the retriever and
    check whether all of its expected_keywords appear somewhere
    in the combined text of the retrieved chunks.

    This evaluates the RETRIEVAL layer only (no LLM call), so it
    can run in any environment without Ollama or a live model.

    Args:
        dataset: List of evaluation cases (see EVALUATION_DATASET).
        chunks: Document chunk repository to search.
        top_k: Number of chunks to retrieve per question.

    Returns:
        dict with:
            "results": list of per-question result dicts, each
                containing id, question, description, passed,
                missing_keywords, top_score, top_confidence.
            "total": number of cases evaluated.
            "passed": number of cases that passed.
            "accuracy": passed / total (0.0 if dataset is empty).
    """

    results = []

    for case in dataset:
        retrieved_chunks = retrieve_relevant_chunks(
            case["question"],
            chunks,
            top_k=top_k
        )

        combined_text = " ".join(
            result["chunk"]["text"]
            for result in retrieved_chunks
        )

        missing_keywords = [
            keyword
            for keyword in case["expected_keywords"]
            if keyword.lower() not in combined_text.lower()
        ]

        passed = not missing_keywords

        results.append({
            "id": case["id"],
            "question": case["question"],
            "description": case["description"],
            "passed": passed,
            "missing_keywords": missing_keywords,
            "top_score": retrieved_chunks[0]["score"] if retrieved_chunks else 0,
            "top_confidence": retrieved_chunks[0]["confidence"] if retrieved_chunks else 0.0,
        })

    total = len(results)
    passed_count = sum(1 for result in results if result["passed"])

    return {
        "results": results,
        "total": total,
        "passed": passed_count,
        "accuracy": round(passed_count / total, 2) if total else 0.0,
    }


def is_relevant_chunk(case, chunk_text):
    """
    Determine whether a single chunk, on its own, actually
    answers an evaluation case.

    Stricter than evaluate_retrieval's keyword check: here ALL
    expected_keywords must appear in THIS chunk alone (not
    scattered across the combined text of several chunks). This
    matters for V2.3, since Precision/Recall/MRR need a real,
    per-chunk ground-truth judgment - a "pass" built from
    keywords split across unrelated chunks isn't a chunk that
    actually answers the question.

    Args:
        case: Evaluation case dict with "expected_keywords".
        chunk_text: Text of a single chunk.

    Returns:
        True if this chunk alone contains every expected keyword.
    """
    return all(
        keyword.lower() in chunk_text.lower()
        for keyword in case["expected_keywords"]
    )


def calculate_retrieval_metrics(dataset, chunks, top_k=3):
    """
    Compute standard information-retrieval quality metrics for
    each question in the dataset, plus their aggregate averages.

    For each question:
        - total_relevant_in_corpus: how many chunks in the WHOLE
          document collection would, on their own, answer this
          question (the ground truth for Recall).
        - relevant_retrieved: how many of the top_k retrieved
          chunks are actually relevant (per is_relevant_chunk).
        - precision_at_k: relevant_retrieved / top_k. Note this
          is structurally capped below 1.0 whenever top_k is
          larger than total_relevant_in_corpus - that's expected,
          not a bug (e.g. 1 relevant chunk out of top_k=3 gives
          a maximum possible precision of 0.33).
        - recall_at_k: relevant_retrieved / total_relevant_in_corpus.
          1.0 means every relevant chunk in the whole document
          was successfully retrieved in the top_k.
        - reciprocal_rank: 1 / rank of the FIRST relevant chunk
          in the ranked results (0.0 if none was found). Rewards
          ranking the right chunk near the top, not just
          including it somewhere in top_k.

    Args:
        dataset: List of evaluation cases (see EVALUATION_DATASET).
        chunks: Full document chunk repository (used both to
            establish ground truth and to run retrieval).
        top_k: Number of chunks to retrieve per question.

    Returns:
        dict with "results" (per-question metrics), "total",
        and the dataset-wide averages "mean_precision_at_k",
        "mean_recall_at_k", "mrr".
    """

    results = []

    for case in dataset:
        total_relevant = sum(
            1 for chunk in chunks if is_relevant_chunk(case, chunk["text"])
        )

        retrieved = retrieve_relevant_chunks(
            case["question"],
            chunks,
            top_k=top_k
        )

        relevance_flags = [
            is_relevant_chunk(case, result["chunk"]["text"])
            for result in retrieved
        ]

        relevant_retrieved = sum(relevance_flags)

        precision_at_k = relevant_retrieved / top_k if top_k else 0.0
        recall_at_k = (
            relevant_retrieved / total_relevant if total_relevant else 0.0
        )

        reciprocal_rank = 0.0
        for rank, is_relevant in enumerate(relevance_flags, start=1):
            if is_relevant:
                reciprocal_rank = 1 / rank
                break

        results.append({
            "id": case["id"],
            "question": case["question"],
            "total_relevant_in_corpus": total_relevant,
            "relevant_retrieved": relevant_retrieved,
            "precision_at_k": round(precision_at_k, 2),
            "recall_at_k": round(recall_at_k, 2),
            "reciprocal_rank": round(reciprocal_rank, 2),
        })

    total = len(results)

    mean_precision_at_k = (
        round(sum(r["precision_at_k"] for r in results) / total, 2)
        if total else 0.0
    )
    mean_recall_at_k = (
        round(sum(r["recall_at_k"] for r in results) / total, 2)
        if total else 0.0
    )
    mrr = (
        round(sum(r["reciprocal_rank"] for r in results) / total, 2)
        if total else 0.0
    )

    return {
        "results": results,
        "total": total,
        "mean_precision_at_k": mean_precision_at_k,
        "mean_recall_at_k": mean_recall_at_k,
        "mrr": mrr,
    }


def print_retrieval_metrics_report(report):
    """
    Print a human-readable summary of a retrieval metrics report
    produced by calculate_retrieval_metrics.

    Args:
        report: dict returned by calculate_retrieval_metrics.
    """

    print("=" * 60)
    print("RETRIEVAL METRICS REPORT")
    print("=" * 60)

    for result in report["results"]:
        print(f"\n{result['id']} - {result['question']}")
        print(f"  Relevant chunks in corpus: {result['total_relevant_in_corpus']}")
        print(f"  Relevant retrieved (top_k): {result['relevant_retrieved']}")
        print(
            f"  Precision@K: {result['precision_at_k']} | "
            f"Recall@K: {result['recall_at_k']} | "
            f"Reciprocal Rank: {result['reciprocal_rank']}"
        )

    print("\n" + "=" * 60)
    print(f"Mean Precision@K: {report['mean_precision_at_k']}")
    print(f"Mean Recall@K:    {report['mean_recall_at_k']}")
    print(f"MRR:              {report['mrr']}")
    print("=" * 60)


def print_evaluation_report(report):
    """
    Print a human-readable summary of an evaluation report
    produced by evaluate_retrieval.

    Args:
        report: dict returned by evaluate_retrieval.
    """

    print("=" * 60)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 60)

    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{status}] {result['id']} - {result['description']}")
        print(f"  Question: {result['question']}")
        print(
            f"  Top score: {result['top_score']} | "
            f"Top confidence: {result['top_confidence']}"
        )

        if result["missing_keywords"]:
            print(f"  Missing keywords: {result['missing_keywords']}")

    print("\n" + "=" * 60)
    accuracy_pct = report["accuracy"] * 100
    print(f"TOTAL: {report['passed']}/{report['total']} passed ({accuracy_pct:.0f}%)")
    print("=" * 60)
    