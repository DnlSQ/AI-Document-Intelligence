"""
V2.2 Evaluation Framework.

Provides a small "golden dataset" of question / expected-content
pairs and a runner that checks a retrieval method's real output
against it, producing a structured pass/fail report.

This complements (does not replace) the individual pytest
question tests in test_retriever.py: those catch regressions
with sharp, specific, case-sensitive assertions; this framework
gives a single summary view and makes it easy to grow the
evaluation set over time without writing a new test function for
every question.

Also provides V2.3 Retrieval Metrics (Precision@K, Recall@K, MRR)
and a comparative extension that runs those same metrics across
lexical, semantic, and hybrid retrieval on the same dataset and
corpus, so the three strategies can be measured side by side
instead of eyeballed from separate manual runs.
"""

from src.retriever import retrieve_relevant_chunks
from src.semantic_search import semantic_search
from src.hybrid_retrieval import hybrid_retrieve


# Each entry describes one evaluation case:
#
#   id: short unique identifier
#   question: the user question to ask
#   expected_keywords: strings that must ALL appear somewhere in
#       the combined text of the retrieved top_k chunks for the
#       case to pass (matched case-insensitively)
#   description: human-readable summary, for reporting
EVALUATION_DATASET = [
    # --- Original 4, unchanged content, tagged for the new breakdown ---
    {
        "id": "output_current",
        "question": "What is the maximum output current of the PDTB113ZT?",
        "expected_keywords": ["output current", "500 mA"],
        "description": "Maximum output current",
        "style": "literal",
    },
    {
        "id": "collector_emitter_voltage",
        "question": "What is the maximum collector-emitter voltage?",
        "expected_keywords": ["VCEO", "-50"],
        "description": "Maximum collector-emitter voltage",
        "style": "literal",
    },
    {
        "id": "input_voltage",
        "question": "What is the maximum input voltage?",
        "expected_keywords": ["input voltage", "5", "-10"],
        "description": "Maximum input voltage",
        "style": "literal",
    },
    {
        "id": "dc_current_gain",
        "question": "What is the DC current gain?",
        "expected_keywords": ["hFE", "DC current gain", "70"],
        "description": "DC current gain (hFE)",
        "style": "literal",
    },

    # --- New literal cases, verified against the real PDF's Limiting
    # values / Thermal characteristics / Characteristics tables ---
    {
        "id": "collector_base_voltage",
        "question": "What is the maximum collector-base voltage?",
        "expected_keywords": ["VCBO", "-50"],
        "description": "Maximum collector-base voltage (VCBO)",
        "style": "literal",
    },
    {
        "id": "emitter_base_voltage",
        "question": "What is the maximum emitter-base voltage?",
        "expected_keywords": ["VEBO", "-5"],
        "description": "Maximum emitter-base voltage (VEBO)",
        "style": "literal",
    },
    {
        "id": "total_power_dissipation",
        "question": "What is the total power dissipation?",
        "expected_keywords": ["Ptot", "250"],
        "description": "Total power dissipation (Ptot)",
        "style": "literal",
    },
    {
        "id": "storage_temperature",
        "question": "What is the storage temperature range?",
        "expected_keywords": ["storage temperature", "-65"],
        "description": "Storage temperature range (Tstg)",
        "style": "literal",
    },
    {
        "id": "thermal_resistance",
        "question": "What is the thermal resistance from junction to ambient?",
        "expected_keywords": ["thermal resistance", "500"],
        "description": "Thermal resistance junction-to-ambient (Rth(j-a))",
        "style": "literal",
    },
    {
        "id": "saturation_voltage",
        "question": "What is the collector-emitter saturation voltage?",
        "expected_keywords": ["VCEsat", "-300"],
        "description": "Collector-emitter saturation voltage (VCEsat)",
        "style": "literal",
    },
    {
        "id": "package_type",
        "question": "What package does the PDTB113ZT come in?",
        "expected_keywords": ["SOT23", "TO-236AB"],
        "description": "Package type (SOT23 / TO-236AB)",
        "style": "literal",
    },
    {
        "id": "npn_complement",
        "question": "What is the NPN complement of the PDTB113ZT?",
        "expected_keywords": ["PDTD113ZT"],
        "description": "NPN complement part number",
        "style": "literal",
    },

    # --- Paraphrased cases: same underlying facts, natural-language
    # wording far from the datasheet's own vocabulary - the terrain
    # where semantic search is supposed to have an edge. ---
    {
        "id": "collector_base_voltage_paraphrase",
        "question": "How much voltage can this transistor withstand between its collector and base before breaking down?",
        "expected_keywords": ["VCBO", "-50"],
        "description": "Maximum collector-base voltage, paraphrased",
        "style": "paraphrase",
    },
    {
        "id": "total_power_dissipation_paraphrase",
        "question": "How much power can this component safely dissipate?",
        "expected_keywords": ["Ptot", "250"],
        "description": "Total power dissipation, paraphrased",
        "style": "paraphrase",
    },
    {
        "id": "storage_temperature_paraphrase",
        "question": "What temperature range can this part be stored in without damage?",
        "expected_keywords": ["storage temperature", "-65"],
        "description": "Storage temperature range, paraphrased",
        "style": "paraphrase",
    },
    {
        "id": "saturation_voltage_paraphrase",
        "question": "What is the voltage drop across the transistor when it's fully switched on?",
        "expected_keywords": ["VCEsat", "-300"],
        "description": "Collector-emitter saturation voltage, paraphrased",
        "style": "paraphrase",
    },
]

# Separate golden dataset for NE555N.pdf (RAG v6.3). Kept apart from
# EVALUATION_DATASET on purpose: test_evaluation.py and
# test_retrieval_metrics_real.py build their chunk corpus from
# sample.pdf ALONE and assert perfect accuracy/MRR over the full
# EVALUATION_DATASET list - mixing NE555N.pdf questions into that
# same list would fail those tests immediately, since sample.pdf's
# chunks obviously don't contain NE555N's answers. This is also why
# v6's plan explicitly rejected a "one dataset per uploaded document"
# pattern (see claude/rag-v6-plan.md's Non-goals) - this dataset
# exists only because NE555N.pdf itself is a permanent, committed
# reference document, not a real user's transient upload.
NE555N_EVALUATION_DATASET = [
    {
        "id": "ne555_turn_off_time",
        "question": "What is the turn off time of the NE555?",
        "expected_keywords": ["toff", "0.5", "µs"],
        "description": (
            "Turn off time (toff) - the exact question that returned "
            "the no-answer fallback throughout RAG v5.5's investigation, "
            "until RAG v6.2's table-aware extraction fix"
        ),
        "style": "literal",
    },
    {
        "id": "ne555_operating_supply_voltage_range",
        "question": "What is the operating supply voltage range of the NE555?",
        "expected_keywords": ["4.5", "16"],
        "description": "Operating supply voltage range",
        "style": "literal",
    },
]

def evaluate_retrieval(dataset, chunks, top_k=3):
    """
    Run every question in the dataset through the lexical
    retriever and check whether all of its expected_keywords
    appear somewhere in the combined text of the retrieved
    chunks.

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
    matters for retrieval metrics, since Precision/Recall/MRR need
    a real, per-chunk ground-truth judgment - a "pass" built from
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


def calculate_retrieval_metrics(dataset, chunks, top_k=3, retrieve_fn=None):
    """
    Compute standard information-retrieval quality metrics for
    each question in the dataset, plus their aggregate averages.

    For each question:
        - total_relevant_in_corpus: how many chunks in the WHOLE
          document collection would, on their own, answer this
          question (the ground truth for Recall). This is a
          property of the CORPUS, not of any retrieval method, so
          it's computed the same way regardless of retrieve_fn.
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
        chunks: Full document chunk repository (used to establish
            ground truth, and also as the search space for the
            default lexical retrieve_fn).
        top_k: Number of chunks to retrieve per question.
        retrieve_fn: Optional callable `(question, top_k) -> list`
            of results shaped like retrieve_relevant_chunks's
            output (each result must have a "chunk" dict with a
            "text" key - that's all is_relevant_chunk needs; extra
            fields like "score"/"confidence" vs. hybrid_retrieve's
            "rrf_score"/"lexical_rank"/etc. are irrelevant here).
            Defaults to lexical retrieval (retrieve_relevant_chunks
            against `chunks`). This lets the exact same
            Precision/Recall/MRR arithmetic be reused to evaluate
            semantic search or hybrid retrieval too (see
            compare_retrieval_methods), instead of duplicating it
            per method.

    Returns:
        dict with "results" (per-question metrics), "total",
        and the dataset-wide averages "mean_precision_at_k",
        "mean_recall_at_k", "mrr".
    """

    if retrieve_fn is None:
        retrieve_fn = lambda question, k: retrieve_relevant_chunks(
            question, chunks, top_k=k
        )

    results = []

    for case in dataset:
        total_relevant = sum(
            1 for chunk in chunks if is_relevant_chunk(case, chunk["text"])
        )

        retrieved = retrieve_fn(case["question"], top_k)

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


def compare_retrieval_methods(dataset, chunks, collection=None, top_k=3):
    """
    Compute Precision@K / Recall@K / MRR for lexical, semantic,
    and hybrid retrieval on the SAME dataset and the same
    ground-truth corpus, so the three strategies can be compared
    on equal footing instead of eyeballed from separate manual
    runs (see claude/rag-v3-progress.md's V3.3 manual finding that
    prompted this - semantic winning on paraphrase, losing on
    short technical jargon).

    Args:
        dataset: List of evaluation cases (see EVALUATION_DATASET).
        chunks: Full document chunk repository - used both as
            ground truth (is_relevant_chunk) and as the lexical
            retriever's search space.
        collection: Optional vector store collection (for tests).
            Defaults to the real persistent collection. Passed
            through to both semantic and hybrid retrieval.
        top_k: Number of chunks retrieved per question, per method.

    Returns:
        dict with "lexical", "semantic", "hybrid" keys, each a
        full report as returned by calculate_retrieval_metrics.
    """
    return {
        "lexical": calculate_retrieval_metrics(
            dataset, chunks, top_k=top_k,
            retrieve_fn=lambda question, k: retrieve_relevant_chunks(
                question, chunks, top_k=k
            )
        ),
        "semantic": calculate_retrieval_metrics(
            dataset, chunks, top_k=top_k,
            retrieve_fn=lambda question, k: semantic_search(
                question, top_k=k, collection=collection
            )
        ),
        "hybrid": calculate_retrieval_metrics(
            dataset, chunks, top_k=top_k,
            retrieve_fn=lambda question, k: hybrid_retrieve(
                question, chunks, collection=collection, top_k=k
            )
        ),
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


def print_comparison_report(comparison):
    """
    Print a human-readable side-by-side summary of a comparison
    report produced by compare_retrieval_methods.

    Args:
        comparison: dict returned by compare_retrieval_methods
            ("lexical" / "semantic" / "hybrid" keys, each a
            calculate_retrieval_metrics report).
    """

    print("=" * 60)
    print("RETRIEVAL METHOD COMPARISON")
    print("=" * 60)

    header = f"{'Method':<10} {'Precision@K':>12} {'Recall@K':>10} {'MRR':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for method_name in ("lexical", "semantic", "hybrid"):
        report = comparison[method_name]
        print(
            f"{method_name:<10} "
            f"{report['mean_precision_at_k']:>12} "
            f"{report['mean_recall_at_k']:>10} "
            f"{report['mrr']:>6}"
        )

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
    