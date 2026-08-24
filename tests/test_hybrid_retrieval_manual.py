"""
Manual side-by-side comparison: lexical, semantic, and hybrid
(RRF-fused) retrieval on the real document set. Reuses the same
questions as test_semantic_search_manual.py so the improvement
(or lack of it) is directly visible against the earlier baseline.

Run by hand:

    python -m tests.test_hybrid_retrieval_manual
"""
from src.main import build_chunk_repository
from src.embeddings import generate_embeddings_for_chunks
from src.vector_store import add_chunks_to_store, reset_store, get_collection
from src.retriever import retrieve_relevant_chunks
from src.semantic_search import semantic_search
from src.hybrid_retrieval import hybrid_retrieve


QUESTIONS = [
    "What is the maximum collector-emitter voltage?",
    "How much voltage can the device withstand between collector and emitter?",
    "What is the DC current gain?",
]


print("=" * 80)
print("BUILDING CHUNK REPOSITORY + VECTOR STORE")
print("=" * 80)

chunks = build_chunk_repository()
embedded_chunks = generate_embeddings_for_chunks(chunks)

collection = get_collection()
reset_store(collection=collection)
add_chunks_to_store(embedded_chunks, collection=collection)

print(f"Loaded {len(chunks)} chunks.\n")


for question in QUESTIONS:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    print("\n--- LEXICAL ---")
    for r in retrieve_relevant_chunks(question, chunks, top_k=2):
        print(f"chunk_id={r['chunk']['chunk_id']} score={r['score']}")

    print("\n--- SEMANTIC ---")
    for r in semantic_search(question, top_k=2, collection=collection):
        print(f"chunk_id={r['chunk']['chunk_id']} score={r['score']:.4f}")

    print("\n--- HYBRID (RRF) ---")
    for r in hybrid_retrieve(question, chunks, collection=collection, top_k=2):
        print(
            f"chunk_id={r['chunk']['chunk_id']} rrf_score={r['rrf_score']:.5f} "
            f"lexical_rank={r['lexical_rank']} semantic_rank={r['semantic_rank']}"
        )
        