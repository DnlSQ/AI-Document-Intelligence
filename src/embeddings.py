"""
V3.1 Embedding Generation.

Responsible only for:

    - Loading a local sentence-embedding model.
    - Converting text into semantic embedding vectors.

Must not contain:

    - Retrieval, ranking, or similarity search logic (see
      retriever.py, and the future vector store / semantic
      search modules planned for V3.2 / V3.3).
    - LLM logic or prompt construction (see generator.py, llm.py).

This module has no knowledge of the rest of the RAG pipeline: it
takes text in and returns vectors out. Nothing outside this file
should need to know which embedding model produced the vectors.
"""

# Local, open-source sentence-embedding model (Apache 2.0 license,
# via sentence-transformers). Downloaded once from Hugging Face and
# cached on disk; every call after that runs fully offline - the
# same "pull once, run locally" model already used for the Ollama
# LLM (see llm.py / config.MODEL_NAME).
#
# all-MiniLM-L6-v2 is a widely used baseline: 384-dimensional
# vectors, small enough to run fast on CPU (no GPU required), with
# good general-purpose semantic quality. This is the RAG v3
# default. BAAI/bge-small-en-v1.5 (also listed as a preferred
# technology in AI_PROJECT_INSTRUCTIONS.md) is a stronger but
# heavier alternative that could replace this constant later
# without touching any other module.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lazily loaded singleton. Loading a sentence-transformers model
# has a real disk and memory cost, so it must happen at most once
# per process, and only when an embedding is actually requested -
# not at import time (e.g. a script that only imports this module
# to reuse a constant shouldn't pay to load the model).
_model = None


def _load_model():
    """
    Load the configured sentence-embedding model from disk
    (downloading it first if this is the first run).

    Imported lazily, inside this function rather than at module
    level, so that importing embeddings.py never requires
    sentence-transformers to be installed unless an embedding is
    actually generated.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _get_model():
    """
    Return the shared embedding model instance, loading it on
    first use.
    """
    global _model

    if _model is None:
        _model = _load_model()

    return _model


def generate_embedding(text):
    """
    Generate a semantic embedding vector for a single piece of
    text (e.g. a user question).

    Args:
        text: Input text.

    Returns:
        List[float]: the embedding vector, L2-normalized so that
        cosine similarity between two vectors reduces to a plain
        dot product - this is what the planned V3.2 vector store
        will rely on.

    Raises:
        ValueError: if text is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)

    return vector.tolist()


def generate_embeddings_for_chunks(chunks):
    """
    Generate a semantic embedding for each chunk's text, encoded
    together in a single batch for efficiency.

    Args:
        chunks: List of chunk dicts as produced by
            chunker.create_document_chunks (each containing at
            least a "text" key). Not mutated.

    Returns:
        List of new chunk dicts, each identical to the input
        chunk plus an added "embedding" key (List[float]).
        Empty list if chunks is empty.
    """
    if not chunks:
        return []

    model = _get_model()
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, normalize_embeddings=True)

    embedded_chunks = []

    for chunk, vector in zip(chunks, vectors):
        embedded_chunk = dict(chunk)
        embedded_chunk["embedding"] = vector.tolist()
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks
