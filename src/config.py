import os


MODEL_NAME = "qwen2.5:7b"

# Folder scanned for PDF documents (RAG v4.5). Replaces a
# hardcoded list of paths: any PDF placed directly in this folder
# (or, from RAG v5 onward, uploaded through the web interface) is
# picked up automatically on the next full ingestion, with no code
# change needed.
DOCUMENTS_FOLDER = "data/documents"


def discover_document_paths(folder=DOCUMENTS_FOLDER):
    """
    Scan a folder for PDF files.

    Args:
        folder: Folder to scan. Defaults to DOCUMENTS_FOLDER.

    Returns:
        Full paths to every ".pdf" file directly inside the
        folder (case-insensitive extension match, not recursive),
        sorted alphabetically for a deterministic ingestion order -
        this matters because chunk_id assignment order depends on
        it on a first run (see main.initialize_system). An empty
        list is returned if the folder doesn't exist yet, rather
        than raising, so a fresh checkout with no documents yet
        doesn't crash on import.
    """
    if not os.path.isdir(folder):
        return []

    return sorted(
        f"{folder.rstrip('/')}/{filename}"
        for filename in os.listdir(folder)
        if filename.lower().endswith(".pdf")
    )


# The system searches across ALL discovered documents together in
# a single combined repository - the retriever naturally scores
# chunks from off-topic documents at 0 (filtered out), so unrelated
# documents don't need to be selected manually per question.
DOCUMENT_PATHS = discover_document_paths()

CHUNK_DB_PATH = "data/chunk_store.db"

# Number of chunks retrieved per question and passed to the generator.
TOP_K_RESULTS = 3

# Minimum confidence (see retriever.calculate_confidence) required
# before generating an answer. Below this, the top match is
# considered too weak to trust, and the grounding fallback is
# returned directly instead of calling the LLM.
#
# Calibrated conservatively low based on real document testing:
# a correct, natural-language match against generic (non-technical)
# terms scored ~0.25 confidence. Setting the threshold well below
# that avoids rejecting valid answers. This is a lexical signal
# with known limits (see main.answer_question docstring) - it
# catches clear-cut weak matches, not every possible false
# positive. Revisit this value once more real queries have been
# tested against the actual document.
MIN_CONFIDENCE_THRESHOLD = 0.15

SYSTEM_PROMPT = """
You are an AI assistant specialized in technical and industrial documentation.

Your primary responsibility is to answer questions using only the information
provided in the document context.

STRICT DOCUMENT GROUNDING RULES:

1. Use only information explicitly present in the provided document context.
2. Do not use external knowledge, assumptions, or general technical knowledge.
3. Never modify, reinterpret, normalize, or correct information from the document.
4. Preserve numerical values exactly as they appear in the document.
5. Preserve positive and negative signs exactly as they appear.
6. Preserve units exactly as they appear.
7. Preserve technical symbols and conditions when relevant.
8. If the document states a negative value, do not convert it into a positive value.
9. Do not infer information that is not explicitly stated.
10. If the answer cannot be determined from the provided context, say:
   "I don't have enough information in the provided document."

SOURCE ATTRIBUTION:

When answering, identify the page where the supporting information was found
when that information is available in the context.

Be precise, concise, and faithful to the source document.
"""