import re


# Default chunk configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MIN_CHUNK_SIZE = 100


def split_text(
    text,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    min_chunk_size=DEFAULT_MIN_CHUNK_SIZE
):
    """
    Split text into chunks while trying to preserve word boundaries.

    Args:
        text: Text to split.
        chunk_size: Maximum approximate size of each chunk.
        chunk_overlap: Number of characters shared between chunks.
        min_chunk_size: Minimum preferred size for a chunk.

    Returns:
        List of text chunks.
    """

    if not text or not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    if min_chunk_size <= 0:
        raise ValueError(
            "min_chunk_size must be greater than 0"
        )

    text = text.strip()

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # If this is not the last chunk,
        # try to end at a natural boundary.
        if end < text_length:

            boundary = text.rfind("\n", start, end)

            if boundary == -1:
                boundary = text.rfind(" ", start, end)

            # Only use the boundary if the resulting
            # chunk is not too small.
            if boundary > start:
                candidate_size = boundary - start

                if candidate_size >= min_chunk_size:
                    end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - chunk_overlap

        # Prevent infinite loops.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def create_document_chunks(
    pages,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    min_chunk_size=DEFAULT_MIN_CHUNK_SIZE,
    source=None
):
    """
    Create chunks from document pages while preserving metadata.

    Args:
        pages: List of dictionaries containing:

            {
                "page": int,
                "text": str
            }

        chunk_size: Maximum approximate size of each chunk.
        chunk_overlap: Number of characters shared between chunks.
        min_chunk_size: Minimum preferred size for a chunk.
        source: Source document name or path.

    Returns:
        List of dictionaries containing:

            {
                "chunk_id": int,
                "page": int,
                "text": str,
                "source": str
            }
    """

    document_chunks = []

    chunk_id = 1

    for page in pages:

        page_number = page["page"]
        page_text = page["text"]

        chunks = split_text(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )

        for chunk in chunks:

            document_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk,
                    "source": source
                }
            )

            chunk_id += 1

    return document_chunks
