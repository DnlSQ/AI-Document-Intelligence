"""
V3.3 Semantic Search.

Responsible only for:

    - Embedding a user question.
    - Querying the vector store for the most similar chunks.
    - Shaping results to match the existing retrieval result
      format (chunk / score / confidence), so this can be
      compared against - and later combined with - the existing
      lexical retriever (retriever.py) in V3.4 Hybrid Retrieval.

Must not contain:

    - Embedding generation logic (see embeddings.py).
    - Vector storage logic (see vector_store.py).
    - LLM logic or prompt construction (see generator.py, llm.py).
"""
from src.embeddings import generate_embedding
from src.vector_store import get_collection


def semantic_search(question, top_k=3, collection=None):
    """
    Retrieve the most semantically similar chunks to a question.

    Args:
        question: User question.
        top_k: Number of chunks to retrieve.
        collection: Optional vector store collection (for tests).
            Defaults to the real persistent collection.

    Returns:
        List of results, ranked most similar first, each:
            {
                "chunk": {
                    "chunk_id": int,
                    "page": int,
                    "text": str,
                    "source": str
                },
                "score": float,       # cosine similarity, -1..1
                "confidence": float   # score clamped to 0..1
            }
        Empty list if the store has no chunks at all.
    """
    if collection is None:
        collection = get_collection()

    if collection.count() == 0:
        return []

    query_vector = generate_embedding(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    scored_chunks = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        # The collection is created with hnsw:space="cosine"
        # (see vector_store.get_collection), so Chroma's
        # "distance" here is cosine distance: 1 - cosine
        # similarity. Converting back gives a similarity score
        # directly, without needing the raw vectors again.
        similarity = 1 - distance

        scored_chunks.append({
            "chunk": {
                "chunk_id": metadata["chunk_id"],
                "page": metadata["page"],
                "text": text,
                "source": metadata["source"]
            },
            "score": similarity,
            "confidence": round(max(0.0, min(1.0, similarity)), 2)
        })

    return scored_chunks
