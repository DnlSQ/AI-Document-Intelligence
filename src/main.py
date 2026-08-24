from src.config import DOCUMENT_PATHS, TOP_K_RESULTS, MIN_CONFIDENCE_THRESHOLD
from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.retriever import retrieve_relevant_chunks
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


def answer_question(question, chunks, top_k=TOP_K_RESULTS):
    """
    Retrieve relevant chunks for a question and generate
    a grounded answer.

    This is the full query pipeline:

        Question -> Retriever -> Generator -> Grounded Answer

    The LLM is not called (the grounding fallback is returned
    directly instead) in two cases:

    1. Retrieval found no matching chunks at all.
    2. Retrieval found a match, but its confidence (see
       retriever.calculate_confidence) is below
       MIN_CONFIDENCE_THRESHOLD - the match is too weak to
       trust as real grounding.

    Note: confidence is a lexical signal and has known limits
    (e.g. it cannot perfectly tell a real generic-word match
    apart from a coincidental stopword overlap). It catches the
    common, clear-cut cases; it is not a guarantee against every
    possible false positive. Semantic retrieval (RAG v3) is the
    planned fix for that deeper limitation.

    Args:
        question: User question.
        chunks: Document chunk repository.
        top_k: Number of chunks to retrieve.

    Returns:
        Grounded answer string.
    """

    retrieved_chunks = retrieve_relevant_chunks(
        question,
        chunks,
        top_k=top_k
    )

    if not retrieved_chunks:
        return NO_CONTEXT_ANSWER

    top_confidence = retrieved_chunks[0]["confidence"]

    if top_confidence < MIN_CONFIDENCE_THRESHOLD:
        return NO_CONTEXT_ANSWER

    return generate_answer(question, retrieved_chunks)


def run_cli():
    """
    Run the interactive AI Document Intelligence CLI.

    Loads the configured document(s) once at startup, then
    answers questions using the full RAG pipeline until
    the user exits.
    """

    print("====================================")
    print("      AI Document Intelligence")
    print("====================================")
    print("Loading documents:")
    for path in DOCUMENT_PATHS:
        print(f"  - {path}")

    chunks = build_chunk_repository()

    print(f"\nLoaded {len(chunks)} chunk(s) from {len(DOCUMENT_PATHS)} document(s).")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer = answer_question(question, chunks)

        print(f"\nAI: {answer}\n")


if __name__ == "__main__":
    run_cli()
    