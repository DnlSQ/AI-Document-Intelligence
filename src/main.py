"""
RAG pipeline orchestration.

Wires together the full system:

    Ingestion:  PDF -> Document Loader -> Text Cleaner -> Chunker
                     -> Embeddings -> Vector Store

    Query:      Question -> Hybrid Retrieval (lexical + semantic,
                     fused with RRF) -> Generator -> Grounded Answer

    Startup (RAG v4): initialize_system reuses whatever has
    already been persisted (chunk_store.py + vector_store.py)
    instead of always reprocessing every PDF from scratch - see
    initialize_system's docstring.
"""
from src.config import DOCUMENT_PATHS, TOP_K_RESULTS, MIN_CONFIDENCE_THRESHOLD, CHUNK_DB_PATH
from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.embeddings import generate_embeddings_for_chunks
from src.vector_store import get_collection, reset_store, add_chunks_to_store
from src.hybrid_retrieval import hybrid_retrieve
from src.generator import generate_answer
from src.chunk_store import load_all_chunks
from src.ingestion import add_or_replace_document, replace_document_vectors


NO_CONTEXT_ANSWER = "I don't have enough information in the provided document."


def build_chunk_repository(pdf_paths=DOCUMENT_PATHS):
    """
    Load one or more PDFs, clean their text and split them into
    indexed chunks, combined into a single searchable repository.

    This is the full ingestion pipeline, run once per document:

        PDF -> Document Loader -> Text Cleaner -> Chunker

    The retriever doesn't need to know which document a chunk
    came from to score it correctly: an unrelated document's
    chunks simply score 0 (or very low) against an off-topic
    query and get filtered out naturally. This is what enables
    searching across multiple, unrelated documents at once
    without picking one manually.

    Chunk IDs are re-numbered sequentially across the COMBINED
    repository (not per document), so they stay globally unique.
    This matters because retriever.retrieve_relevant_chunks uses
    chunk_id as its final, deterministic tie-break signal - if
    two documents each produced their own "chunk_id: 1", that
    guarantee would silently break.

    NOTE (RAG v4): this full-rebuild path is kept for the manual
    test scripts and for test_main.py's own coverage, but the
    running application no longer calls this directly - see
    initialize_system, which only falls back to a per-document
    version of this same pipeline (via ingestion.py) when nothing
    has been persisted yet.

    Args:
        pdf_paths: List of paths to source PDF documents.

    Returns:
        List of document chunks from all documents combined,
        ready for retrieval.
    """

    all_chunks = []

    for pdf_path in pdf_paths:
        pages = extract_text_from_pdf(pdf_path)

        for page in pages:
            page["text"] = clean_text(page["text"])

        document_chunks = create_document_chunks(
            pages,
            source=pdf_path
        )

        all_chunks.extend(document_chunks)

    for index, chunk in enumerate(all_chunks, start=1):
        chunk["chunk_id"] = index

    return all_chunks


def build_vector_store(chunks, collection=None):
    """
    Generate embeddings for every chunk and (re)populate the
    vector store with them - the V3.1/V3.2 ingestion step
    semantic and hybrid retrieval both depend on.

    Always resets the store before writing, rather than only
    upserting: this keeps the store in sync with `chunks` even
    when a document was removed from DOCUMENT_PATHS - upsert
    alone would leave that document's old chunks behind forever
    (see vector_store.reset_store's docstring).

    NOTE (RAG v4): kept for the same reason as
    build_chunk_repository above - the running application uses
    initialize_system / replace_document_vectors instead, which
    update the store incrementally per document rather than
    resetting everything.

    Args:
        chunks: Document chunk repository (from
            build_chunk_repository).
        collection: Optional vector store collection (for tests).
            Defaults to the real persistent collection.

    Returns:
        The (now populated) collection.
    """
    if collection is None:
        collection = get_collection()

    embedded_chunks = generate_embeddings_for_chunks(chunks)

    reset_store(collection=collection)
    add_chunks_to_store(embedded_chunks, collection=collection)

    return collection


def initialize_system(document_paths=DOCUMENT_PATHS, db_path=CHUNK_DB_PATH, collection=None):
    """
    Start the system from whatever has already been persisted,
    instead of always reprocessing every PDF from scratch.

    If the chunk store already has data (a previous run already
    ingested it), that data is loaded directly and the existing
    vector store collection is reused as-is - the whole point of
    RAG v4's persistence layer (chunk_store.py + vector_store.py)
    is that this is the common case after the very first run.

    If the chunk store is empty (first run ever, or a fresh
    install with no data yet), each configured document is run
    through the same add_or_replace_document +
    replace_document_vectors pipeline a document upload will use
    later (RAG v5) - so the first-run path and the upload path
    share the exact same ingestion logic, rather than duplicating
    it here as a separate "bulk" version.

    Args:
        document_paths: PDF paths to ingest on a first run. Only
            used when the chunk store is empty.
        db_path: Path to the SQLite chunk database.
        collection: Optional vector store collection (for tests).
            Defaults to the real persistent collection.

    Returns:
        (chunks, collection): the full chunk list (loaded or
        freshly ingested) and the vector store collection, ready
        to pass into answer_question.
    """
    if collection is None:
        collection = get_collection()

    chunks = load_all_chunks(db_path)

    if chunks:
        return chunks, collection

    all_chunks = []

    for pdf_path in document_paths:
        document_chunks = add_or_replace_document(pdf_path, db_path)
        replace_document_vectors(document_chunks, pdf_path, collection=collection)
        all_chunks.extend(document_chunks)

    return all_chunks, collection


def answer_question(question, chunks, collection=None, top_k=TOP_K_RESULTS):
    """
    Retrieve relevant chunks for a question (via hybrid retrieval)
    and generate a grounded answer.

    This is the full query pipeline:

        Question -> Hybrid Retrieval -> Generator -> Grounded Answer

    Hybrid retrieval (V3.4) fuses lexical retrieval and semantic
    search via Reciprocal Rank Fusion - see hybrid_retrieval.py.

    The LLM is not called (the grounding fallback is returned
    directly instead) in two cases:

    1. Retrieval found no matching chunks at all.
    2. Retrieval found a match, but neither method was
       individually confident in it: the gate uses
       max(lexical_confidence, semantic_confidence) from the top
       result, not the RRF score itself - RRF's own scale is
       compressed by its damping constant and isn't meaningful as
       an absolute accept/reject threshold, only for ordering. See
       rag-v3-progress.md for the full reasoning.

    Args:
        question: User question.
        chunks: Document chunk repository (for lexical retrieval).
        collection: Optional vector store collection (for tests).
            Defaults to the real persistent collection.
        top_k: Number of chunks to retrieve.

    Returns:
        Grounded answer string.
    """

    retrieved_chunks = hybrid_retrieve(
        question,
        chunks,
        collection=collection,
        top_k=top_k
    )

    if not retrieved_chunks:
        return NO_CONTEXT_ANSWER

    top_result = retrieved_chunks[0]
    top_confidence = max(
        top_result["lexical_confidence"],
        top_result["semantic_confidence"]
    )

    if top_confidence < MIN_CONFIDENCE_THRESHOLD:
        return NO_CONTEXT_ANSWER

    return generate_answer(question, retrieved_chunks)


def run_cli():
    """
    Run the interactive AI Document Intelligence CLI.

    Reuses whatever has already been persisted (RAG v4:
    initialize_system) instead of always reprocessing every PDF
    from scratch - only the very first run ever performs the full
    ingestion pass.
    """

    print("====================================")
    print("      AI Document Intelligence")
    print("====================================")

    chunks, collection = initialize_system()

    print(f"Loaded {len(chunks)} chunk(s) from the persistent store.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer = answer_question(question, chunks, collection=collection)

        print(f"\nAI: {answer}\n")


if __name__ == "__main__":
    run_cli()
    