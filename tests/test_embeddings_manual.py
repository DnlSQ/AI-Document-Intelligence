"""
Manual check for the REAL embedding model (not mocked).

This is not part of the automated pytest suite (see
test_embeddings.py for that) - it downloads and runs the actual
sentence-transformers model, the same way test_rag_manual.py runs
the real retriever + real Ollama model rather than mocks. Run it
by hand, once, after `pip install -r requirements.txt`:

    python -m tests.test_embeddings_manual

What it demonstrates: the actual reason RAG v3 exists. The
current lexical retriever (retriever.py) scores chunks by exact
word overlap, so a query and a passage that mean the same thing
but use different words score poorly together. Embeddings should
place them close in vector space anyway.
"""
from src.embeddings import generate_embedding


def cosine_similarity(vector_a, vector_b):
    """
    Plain dot product is enough here because generate_embedding
    returns L2-normalized vectors (see embeddings.py).
    """
    return sum(a * b for a, b in zip(vector_a, vector_b))


QUERY = "What is the maximum collector-emitter voltage?"

# Same meaning as QUERY, almost no words in common with it -
# a case where the current lexical retriever would score this
# passage weakly (little/no direct term overlap), but a decent
# embedding model should still place it close to QUERY.
SEMANTICALLY_RELATED = (
    "VCEO rating: -50 V. This is the highest voltage the device "
    "can withstand between its collector and emitter terminals."
)

# Genuinely unrelated content (from the plant biology PDF used
# in the multi-document tests), included as a sanity floor.
UNRELATED = (
    "Plants are classified based on characteristics such as "
    "the presence of vascular tissue, seeds, and flowers."
)


print("=" * 80)
print("EMBEDDING MODEL MANUAL CHECK")
print("=" * 80)

print(f"\nQuery: {QUERY}")

query_vector = generate_embedding(QUERY)
related_vector = generate_embedding(SEMANTICALLY_RELATED)
unrelated_vector = generate_embedding(UNRELATED)

similarity_related = cosine_similarity(query_vector, related_vector)
similarity_unrelated = cosine_similarity(query_vector, unrelated_vector)

print(f"\nEmbedding dimensions: {len(query_vector)}")

print("\n--- Semantically related, few shared words ---")
print(SEMANTICALLY_RELATED)
print(f"Cosine similarity to query: {similarity_related:.4f}")

print("\n--- Unrelated (different document/domain) ---")
print(UNRELATED)
print(f"Cosine similarity to query: {similarity_unrelated:.4f}")

print("\n" + "-" * 80)
if similarity_related > similarity_unrelated:
    print(
        "PASS: the related passage scored higher than the "
        "unrelated one, despite sharing few exact words with "
        "the query - this is what the lexical retriever alone "
        "cannot do."
    )
else:
    print(
        "UNEXPECTED: the unrelated passage scored as high as or "
        "higher than the related one. Worth a closer look before "
        "relying on this model for retrieval."
    )
    