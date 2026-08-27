"""
Document-level ingestion: adding a new document to the persistent
chunk repository, or replacing an existing one under the same name.

Responsible only for:
    - Orchestrating extraction -> cleaning -> chunking for ONE
      document
    - Assigning chunk_ids that continue from whatever is already
      persisted, so multiple documents never collide
    - Removing a document's outdated chunks when it's replaced

Must not contain:
    - SQL/persistence details (delegated to chunk_store.py)
    - Embedding or vector store logic (that's V4.3's responsibility)
    - Retrieval or LLM logic
"""
from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.chunk_store import (
    delete_chunks_by_source,
    get_next_chunk_id,
    save_chunks,
)
from src.config import CHUNK_DB_PATH


def add_or_replace_document(pdf_path, db_path=CHUNK_DB_PATH):
    """
    Add a document to the persistent chunk repository, replacing
    any previously stored chunks for the same source.

    Deleting first and always (rather than checking existence
    first) keeps "add" and "replace" a single code path: deleting
    a source with no existing chunks is a safe no-op (see
    chunk_store.delete_chunks_by_source), so there's nothing to
    branch on.

    Chunk IDs are assigned from the persisted database's running
    sequence (chunk_store.get_next_chunk_id), NOT restarted at 1 -
    that's what keeps this document's chunk_ids from ever colliding
    with, or renumbering, any other document already stored.

    Args:
        pdf_path: Path to the PDF file being added or updated. Used
            as the chunk "source" identity - re-uploading a file
            under this same path/name is what triggers a replace.
        db_path: Path to the SQLite chunk database (overridable for
            tests).

    Returns:
        The list of newly created chunks for this document (already
        persisted).
    """
    source = pdf_path

    delete_chunks_by_source(source, db_path)

    pages = extract_text_from_pdf(pdf_path)

    for page in pages:
        page["text"] = clean_text(page["text"])

    chunks = create_document_chunks(pages, source=source)

    starting_id = get_next_chunk_id(db_path)
    for offset, chunk in enumerate(chunks):
        chunk["chunk_id"] = starting_id + offset

    save_chunks(chunks, db_path)

    return chunks
