"""
Manual side-by-side comparison: lexical retrieval (retriever.py)
vs. semantic search (semantic_search.py) on the real document set.

Ingests the real PDFs, generates real embeddings, stores them in
the real persistent vector store, then runs the same questions
through both retrieval methods so the difference is visible
directly - not asserted, just shown (matches test_rag_manual.py's
style).

Run by hand:

    python -m tests.test_semantic_search_manual
"""
from src.main import build_chunk_repository
from src.embeddings import generate_embeddings_for_chunks
from src.vector_store import add_chunks_to_store, reset_store, get_collection
from src.retriever import retrieve_relevant_chunks
from src.semantic_search import semantic_search


QUESTIONS = [
    "What is the maximum collector-emitter voltage?",
    "How much voltage can the device withstand between collector and emitter?",
    "What is the DC current gain?",
]


print("=" * 80)
print("INGESTING DOCUMENTS AND BUILDING THE VECTOR STORE")
print("=" * 80)

chunks = build_chunk_repository()
print(f"Loaded {len(chunks)} chunks.")

print("Generating real embeddings (this uses the downloaded model)...")
embedded_chunks = generate_embeddings_for_chunks(chunks)

collection = get_collection()
reset_store(collection=collection)  # clean rebuild, avoid stale chunks
stored = add_chunks_to_store(embedded_chunks, collection=collection)
print(f"Stored {stored} chunks in the vector store.")


for question in QUESTIONS:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    print("\n--- LEXICAL (retriever.py) ---")
    lexical_results = retrieve_relevant_chunks(question, chunks, top_k=2)
    for result in lexical_results:
        print(f"score={result['score']} confidence={result['confidence']} "
              f"chunk_id={result['chunk']['chunk_id']}")
        print(f"  {result['chunk']['text'][:100]}...")

    print("\n--- SEMANTIC (semantic_search.py) ---")
    semantic_results = semantic_search(question, top_k=2, collection=collection)
    for result in semantic_results:
        print(f"score={result['score']:.4f} confidence={result['confidence']} "
              f"chunk_id={result['chunk']['chunk_id']}")
        print(f"  {result['chunk']['text'][:100]}...")
        