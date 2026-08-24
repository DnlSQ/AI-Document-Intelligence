"""
V3.2 Vector Store.

Responsible only for:

    - Persisting chunk embeddings (and their metadata) to a
      local vector store.
    - Basic storage operations: add, count, reset.

Must not contain:

    - Embedding generation (see embeddings.py - this module only
      stores vectors it's given, it never computes them).
    - Similarity search / query logic (planned for the V3.3
      Semantic Search module).
"""
import chromadb

# Local, on-disk ChromaDB store. No server process, no network,
# no paid service - PersistentClient just writes to a folder on
# disk, the same "runs completely offline" model as everything
# else in this project.
VECTOR_STORE_PATH = "data/vector_store"

COLLECTION_NAME = "document_chunks"


def get_collection(client=None, name=COLLECTION_NAME):
    """
    Get (creating if it doesn't exist yet) the persistent chunk
    collection.

    Args:
        client: Optional chromadb client to use instead of the
            default on-disk one. Tests inject an in-memory
            (Ephemeral) client here so the suite never touches
            disk or depends on a store left over from a previous
            run.
        name: Collection name. Defaults to the shared production
            collection; tests override this with a unique name
            per test, since chromadb can otherwise reuse the same
            underlying in-memory storage for same-named
            collections across different client instances in one
            process - a unique name guarantees real isolation
            between tests regardless of that internal behavior.

    Returns:
        A chromadb Collection.

    Note: embedding_function=None means this collection NEVER
    computes its own embeddings - every add/query call must
    supply vectors explicitly. embeddings.py is the single
    source of truth for how text becomes a vector, and this
    avoids Chroma silently downloading its own default model
    whenever a vector wasn't supplied.
    """
    if client is None:
        client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    return client.get_or_create_collection(
        name=name,
        embedding_function=None
    )


def add_chunks_to_store(embedded_chunks, collection=None):
    """
    Store embedded chunks (as produced by
    embeddings.generate_embeddings_for_chunks) in the vector
    store.

    Uses upsert rather than add: re-ingesting the same document
    (e.g. restarting the app, or re-running ingestion after a
    document edit) must overwrite the existing entry for that
    chunk_id instead of raising a duplicate-ID error. chunk_id is
    already guaranteed globally unique across the whole chunk
    repository (see main.build_chunk_repository), so it doubles
    safely as the vector store's ID.

    Args:
        embedded_chunks: List of chunk dicts, each containing at
            least chunk_id, page, text, source, and embedding.
        collection: Optional collection to write to (for tests).
            Defaults to the real persistent collection.

    Returns:
        Number of chunks stored.
    """
    if not embedded_chunks:
        return 0

    if collection is None:
        collection = get_collection()

    collection.upsert(
        ids=[str(chunk["chunk_id"]) for chunk in embedded_chunks],
        embeddings=[chunk["embedding"] for chunk in embedded_chunks],
        documents=[chunk["text"] for chunk in embedded_chunks],
        metadatas=[
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "source": chunk["source"],
            }
            for chunk in embedded_chunks
        ]
    )

    return len(embedded_chunks)


def get_chunk_count(collection=None):
    """
    Return how many chunks are currently stored.
    """
    if collection is None:
        collection = get_collection()

    return collection.count()


def reset_store(collection=None):
    """
    Delete every chunk currently in the store.

    Needed for a true clean rebuild (e.g. a document was removed
    from DOCUMENT_PATHS): upsert alone would leave that
    document's old chunks behind forever, since nothing would
    ever overwrite or remove them.
    """
    if collection is None:
        collection = get_collection()

    existing_ids = collection.get()["ids"]

    if existing_ids:
        collection.delete(ids=existing_ids)
        