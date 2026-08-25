"""
Manual comparative evaluation: Precision@K, Recall@K, and MRR for
lexical, semantic, and hybrid retrieval, computed on the SAME
golden dataset and the SAME real document set (sample.pdf +
plantas.pdf), side by side.

Also breaks the comparison down by question "style" ("literal"
questions using the datasheet's own vocabulary vs. "paraphrase"
questions using different natural-language wording for the same
underlying facts) - this is the terrain where semantic search is
supposed to have an edge over lexical, so seeing the numbers
separately (not just one blended average) is what actually informs
whether/how to fix hybrid's current underperformance vs. lexical.

Run by hand:

    python -m tests.test_retrieval_comparison_manual
"""
from src.main import build_chunk_repository
from src.embeddings import generate_embeddings_for_chunks
from src.vector_store import add_chunks_to_store, reset_store, get_collection
from src.evaluation import (
    EVALUATION_DATASET,
    compare_retrieval_methods,
    print_comparison_report,
)


print("=" * 80)
print("BUILDING CHUNK REPOSITORY + VECTOR STORE")
print("=" * 80)

chunks = build_chunk_repository()
embedded_chunks = generate_embeddings_for_chunks(chunks)

collection = get_collection()
reset_store(collection=collection)
add_chunks_to_store(embedded_chunks, collection=collection)

print(f"Loaded {len(chunks)} chunks.")

literal_cases = [case for case in EVALUATION_DATASET if case.get("style") == "literal"]
paraphrase_cases = [case for case in EVALUATION_DATASET if case.get("style") == "paraphrase"]

print(
    f"Dataset: {len(EVALUATION_DATASET)} total "
    f"({len(literal_cases)} literal, {len(paraphrase_cases)} paraphrased)\n"
)

print("\n" + "#" * 80)
print("# ALL QUESTIONS")
print("#" * 80)
print_comparison_report(
    compare_retrieval_methods(EVALUATION_DATASET, chunks, collection=collection, top_k=3)
)

print("\n" + "#" * 80)
print("# LITERAL ONLY (matches datasheet wording)")
print("#" * 80)
print_comparison_report(
    compare_retrieval_methods(literal_cases, chunks, collection=collection, top_k=3)
)

print("\n" + "#" * 80)
print("# PARAPHRASED ONLY (natural language, different wording)")
print("#" * 80)
print_comparison_report(
    compare_retrieval_methods(paraphrase_cases, chunks, collection=collection, top_k=3)
)
