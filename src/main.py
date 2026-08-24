"""
RAG pipeline orchestration.

Wires together the full system:

    Ingestion:  PDF -> Document Loader -> Text Cleaner -> Chunker
                     -> Embeddings -> Vector Store

    Query:      Question -> Hybrid Retrieval (lexical + semantic,
                     fused with RRF) -> Generator -> Grounded Answer
"""
from src.config import DOCUMENT_PATHS, TOP_K_RESULTS, MIN_CONFIDENCE_THRESHOLD
from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.embeddings import generate_embeddings_for_chunks
from src.vector_store import get_collection, reset_store, add_chunks_to_store
from src.hybrid_retrieval import hybrid_retrieve
from src.generator import generate_answer


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

    Loads the configured document(s) once at startup, builds the
    semantic index, then answers questions using the full hybrid
    RAG pipeline until the user exits.
    """

    print("====================================")
    print("      AI Document Intelligence")
    print("====================================")
    print("Loading documents:")
    for path in DOCUMENT_PATHS:
        print(f"  - {path}")

    chunks = build_chunk_repository()

    print(f"\nLoaded {len(chunks)} chunk(s) from {len(DOCUMENT_PATHS)} document(s).")

    print("Building semantic index (first run downloads the embedding model)...")
    collection = build_vector_store(chunks)

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
    