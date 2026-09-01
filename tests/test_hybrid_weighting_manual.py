"""
Manual weight-comparison for RAG v7.2: Precision@K, Recall@K, and
MRR for several (lexical_weight, semantic_weight) pairs passed to
hybrid_retrieve, computed on the combined golden dataset (sample.pdf
+ NE555N.pdf questions) and the full real document set (sample.pdf +
plantas.pdf + NE555N.pdf), side by side with the unweighted
(1.0, 1.0) baseline.

hybrid_retrieve defaults to lexical_weight=1.0/semantic_weight=1.0 -
the exact formula RAG v3.4-v6 already used. This script exists to
answer, with real numbers instead of a guess, whether some other
weighting measurably improves retrieval quality before changing
that default. If no candidate clearly beats the baseline, the
default stays 1.0/1.0 and that's a real finding, not a failure to
find something.

Breaks the comparison down by question "style" (literal vs.
paraphrase), same as test_retrieval_comparison_manual.py, since
that's the terrain where lexical and semantic are known to trade
off (see rag-v3-progress.md).

Uses an isolated in-memory ChromaDB collection (chromadb.EphemeralClient,
unique name per run), matching the isolation convention established
in V4 for every other test that touches the vector store. Bug found
during the RAG v7.4 investigation (2026-09-01): this script used to
call vector_store.get_collection() with no override, which defaults
to the REAL on-disk production collection - running this script
silently reset and overwrote Daniel's real persisted vector store
with a fresh rebuild as a side effect, desyncing it from
data/chunk_store.db. Never write to the real store from a measurement
script again.

Run by hand:

    python -m tests.test_hybrid_weighting_manual
"""
import uuid

import chromadb

from src.main import build_chunk_repository
from src.embeddings import generate_embeddings_for_chunks
from src.vector_store import add_chunks_to_store, get_collection
from src.hybrid_retrieval import hybrid_retrieve
from src.evaluation import (
    EVALUATION_DATASET,
    NE555N_EVALUATION_DATASET,
    calculate_retrieval_metrics,
)


# Candidate (lexical_weight, semantic_weight) pairs to measure
# against the baseline. Kept to a small, deliberate grid - not an
# exhaustive search - since the goal is "does leaning either
# direction help at all", not hyperparameter tuning for its own
# sake (see AI_PROJECT_INSTRUCTIONS.md's Performance Philosophy:
# don't optimize without a measured bottleneck).
CANDIDATE_WEIGHTS = [
    (1.0, 1.0),   # baseline - current default, must always be included
    (2.0, 1.0),   # lean lexical
    (1.0, 2.0),   # lean semantic
    (3.0, 1.0),   # lean lexical, harder
    (1.0, 3.0),   # lean semantic, harder
]

DATASET = EVALUATION_DATASET + NE555N_EVALUATION_DATASET


def print_weight_comparison(dataset, chunks, collection, label):
    print("\n" + "#" * 80)
    print(f"# {label}")
    print("#" * 80)

    header = f"{'lexical_w':>10} {'semantic_w':>11} {'Precision@K':>12} {'Recall@K':>10} {'MRR':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for lexical_weight, semantic_weight in CANDIDATE_WEIGHTS:
        # lw/sw default args capture each loop value by VALUE, not
        # by reference - without them, every lambda would share the
        # same closed-over lexical_weight/semantic_weight variables
        # and all end up using the LAST pair in CANDIDATE_WEIGHTS.
        report = calculate_retrieval_metrics(
            dataset, chunks, top_k=3,
            retrieve_fn=lambda question, k, lw=lexical_weight, sw=semantic_weight: hybrid_retrieve(
                question, chunks, collection=collection, top_k=k,
                lexical_weight=lw, semantic_weight=sw,
            )
        )
        print(
            f"{lexical_weight:>10} {semantic_weight:>11} "
            f"{report['mean_precision_at_k']:>12} "
            f"{report['mean_recall_at_k']:>10} "
            f"{report['mrr']:>6}"
        )


print("=" * 80)
print("BUILDING CHUNK REPOSITORY + ISOLATED VECTOR STORE")
print("=" * 80)

chunks = build_chunk_repository()
embedded_chunks = generate_embeddings_for_chunks(chunks)

client = chromadb.EphemeralClient()
collection = get_collection(client=client, name=f"test_hybrid_weighting_{uuid.uuid4().hex}")
add_chunks_to_store(embedded_chunks, collection=collection)

print(f"Loaded {len(chunks)} chunks.")

literal_cases = [case for case in DATASET if case.get("style") == "literal"]
paraphrase_cases = [case for case in DATASET if case.get("style") == "paraphrase"]

print(
    f"Dataset: {len(DATASET)} total "
    f"({len(literal_cases)} literal, {len(paraphrase_cases)} paraphrased)"
)

print_weight_comparison(DATASET, chunks, collection, "ALL QUESTIONS")
print_weight_comparison(literal_cases, chunks, collection, "LITERAL ONLY (matches document wording)")
print_weight_comparison(paraphrase_cases, chunks, collection, "PARAPHRASED ONLY (natural language, different wording)")
