"""
Document-level ingestion: adding a new document to the persistent
chunk repository and vector store, or replacing an existing one
under the same name.

Responsible only for:
    - Orchestrating extraction -> cleaning -> chunking for ONE
      document
    - Assigning chunk_ids that continue from whatever is already
      persisted, so multiple documents never collide
    - Removing a document's outdated chunks/vectors when it's
      replaced
    - Keeping the chunk repository (chunk_store.py) and the vector
      store (vector_store.py) in sync for that one document

Must not contain:
    - SQL/persistence details (delegated to chunk_store.py)
    - Embedding computation itself (delegated to embeddings.py)
      or low-level vector storage details (delegated to
      vector_store.py) - this module only calls them in the right
      order
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
from src.embeddings import generate_embeddings_for_chunks
from src.vector_store import (
    get_collection,
    delete_chunks_by_source as delete_vectors_by_source,
    add_chunks_to_store,
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
        persisted in the chunk repository - NOT yet embedded or
        stored in the vector store, see replace_document_vectors).
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


def replace_document_vectors(chunks, source, collection=None):
    """
    Keep the vector store in sync with a document that was just
    added or replaced in the chunk repository.

    Deletes any previously stored vectors for this source (their
    chunk_ids may no longer match anything in `chunks` - a replace
    assigns fresh chunk_ids, see add_or_replace_document - so a
    plain upsert alone would leave the old vectors behind forever),
    then embeds and stores only the given chunks. Vectors for every
    OTHER document already in the store are left untouched - this
    is what avoids re-embedding the whole library on every upload.

    Args:
        chunks: The chunks for THIS document only (as returned by
            add_or_replace_document) - not the whole repository.
        source: The document identity whose old vectors should be
            removed before the new ones are added.
        collection: Optional vector store collection (for tests).
            Defaults to the real persistent collection.

    Returns:
        Number of chunks stored (see vector_store.add_chunks_to_store).
    """
    if collection is None:
        collection = get_collection()

    delete_vectors_by_source(source, collection=collection)

    embedded_chunks = generate_embeddings_for_chunks(chunks)

    return add_chunks_to_store(embedded_chunks, collection=collection)
