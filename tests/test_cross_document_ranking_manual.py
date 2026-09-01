"""
Manual measurement for the cross-document retrieval ranking gap
found during RAG v7.3.1's live validation: "What is the maximum
collector-emitter voltage?" appeared to return the no-answer
fallback even though sample.pdf contains the answer.

Uses an isolated in-memory ChromaDB collection (chromadb.EphemeralClient,
unique name per run) - never the real on-disk production store. An
earlier version of this script (and test_hybrid_weighting_manual.py)
called vector_store.get_collection() with no override, which
defaults to the REAL production collection - running it silently
reset and overwrote Daniel's real persisted vector store as a side
effect, desyncing it from data/chunk_store.db and making the live
bug seem to disappear on its own. That was never a real fix.

Measures two independent candidate fixes against the real combined
document corpus (sample.pdf + plantas.pdf + NE555N.pdf, freshly
built into the isolated collection above) with real embeddings, at
the current production RRF weights (lexical_weight=2.0,
semantic_weight=1.0 - see main.py):

    1. Raising TOP_K_RESULTS (how many chunks reach the LLM).
    2. Lowering LEXICAL_SAFETY_NET_THRESHOLD (hybrid_retrieval.py).

For each candidate, reports whether the collector_emitter_voltage
case specifically is found (and at what rank), plus Precision@K /
Recall@K / MRR across the full 18-question golden dataset, to catch
any regression the fix might cause elsewhere.

Run by hand:

    python -m tests.test_cross_document_ranking_manual
"""
import uuid

import chromadb

import src.hybrid_retrieval as hybrid_retrieval
from src.main import build_chunk_repository, PRODUCTION_LEXICAL_WEIGHT, PRODUCTION_SEMANTIC_WEIGHT
from src.embeddings import generate_embeddings_for_chunks
from src.vector_store import add_chunks_to_store, get_collection
from src.hybrid_retrieval import hybrid_retrieve
from src.evaluation import (
    EVALUATION_DATASET,
    NE555N_EVALUATION_DATASET,
    calculate_retrieval_metrics,
    is_relevant_chunk,
)


DATASET = EVALUATION_DATASET + NE555N_EVALUATION_DATASET
TARGET_CASE = next(
    case for case in EVALUATION_DATASET if case["id"] == "collector_emitter_voltage"
)

TOP_K_CANDIDATES = [3, 4, 5, 6]
THRESHOLD_CANDIDATES = [0.35, 0.30, 0.25, 0.20]


def _target_case_status(chunks, collection, top_k):
    results = hybrid_retrieve(
        TARGET_CASE["question"], chunks, collection=collection, top_k=top_k,
        lexical_weight=PRODUCTION_LEXICAL_WEIGHT, semantic_weight=PRODUCTION_SEMANTIC_WEIGHT,
    )
    for rank, result in enumerate(results, start=1):
        if is_relevant_chunk(TARGET_CASE, result["chunk"]["text"]):
            return f"FOUND at rank {rank}"
    return "NOT FOUND"


def _dataset_metrics(chunks, collection, top_k):
    return calculate_retrieval_metrics(
        DATASET, chunks, top_k=top_k,
        retrieve_fn=lambda question, k: hybrid_retrieve(
            question, chunks, collection=collection, top_k=k,
            lexical_weight=PRODUCTION_LEXICAL_WEIGHT, semantic_weight=PRODUCTION_SEMANTIC_WEIGHT,
        )
    )


print("=" * 80)
print("BUILDING CHUNK REPOSITORY + ISOLATED VECTOR STORE (real multi-document corpus)")
print("=" * 80)

chunks = build_chunk_repository()
embedded_chunks = generate_embeddings_for_chunks(chunks)

client = chromadb.EphemeralClient()
collection = get_collection(client=client, name=f"test_cross_document_{uuid.uuid4().hex}")
add_chunks_to_store(embedded_chunks, collection=collection)

print(f"Loaded {len(chunks)} chunks.")
print(f"Target case: {TARGET_CASE['id']} - \"{TARGET_CASE['question']}\"")

print("\n" + "#" * 80)
print("# CANDIDATE 1: raising TOP_K_RESULTS (LEXICAL_SAFETY_NET_THRESHOLD held at 0.35)")
print("#" * 80)

header = f"{'top_k':>6} {'target case':>16} {'Precision@K':>12} {'Recall@K':>10} {'MRR':>6}"
print(f"\n{header}")
print("-" * len(header))

for top_k in TOP_K_CANDIDATES:
    status = _target_case_status(chunks, collection, top_k)
    metrics = _dataset_metrics(chunks, collection, top_k)
    print(
        f"{top_k:>6} {status:>16} "
        f"{metrics['mean_precision_at_k']:>12} "
        f"{metrics['mean_recall_at_k']:>10} "
        f"{metrics['mrr']:>6}"
    )

print("\n" + "#" * 80)
print("# CANDIDATE 2: lowering LEXICAL_SAFETY_NET_THRESHOLD (TOP_K_RESULTS held at 3)")
print("#" * 80)

header = f"{'threshold':>10} {'target case':>16} {'Precision@K':>12} {'Recall@K':>10} {'MRR':>6}"
print(f"\n{header}")
print("-" * len(header))

original_threshold = hybrid_retrieval.LEXICAL_SAFETY_NET_THRESHOLD
try:
    for threshold in THRESHOLD_CANDIDATES:
        hybrid_retrieval.LEXICAL_SAFETY_NET_THRESHOLD = threshold
        status = _target_case_status(chunks, collection, top_k=3)
        metrics = _dataset_metrics(chunks, collection, top_k=3)
        print(
            f"{threshold:>10} {status:>16} "
            f"{metrics['mean_precision_at_k']:>12} "
            f"{metrics['mean_recall_at_k']:>10} "
            f"{metrics['mrr']:>6}"
        )
finally:
    hybrid_retrieval.LEXICAL_SAFETY_NET_THRESHOLD = original_threshold
    