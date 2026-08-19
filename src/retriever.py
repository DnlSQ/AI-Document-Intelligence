import re


def tokenize(text):
    """
    Convert text into normalized tokens.

    Args:
        text: Input text.

    Returns:
        List of lowercase words.
    """

    return re.findall(r"\b\w+\b", text.lower())


def normalize_text(text):
    """
    Normalize text for phrase matching.

    The normalization:
    - Converts text to lowercase.
    - Removes extra whitespace.
    - Preserves technical characters such as hyphens.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """

    return re.sub(r"\s+", " ", text.lower()).strip()


def calculate_relevance_score(query, text):
    """
    Calculate a lexical relevance score.

    The score is based on:
    1. Number of unique query terms found in the text.
    2. Bonus for an exact phrase match.

    Args:
        query: User question.
        text: Chunk text.

    Returns:
        Integer relevance score.
    """

    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))

    # Base score: number of unique query terms
    # present in the document chunk.
    term_score = len(
        query_tokens.intersection(text_tokens)
    )

    # Normalize both query and text before checking
    # whether the complete query appears as an exact phrase.
    normalized_query = normalize_text(query)
    normalized_text = normalize_text(text)

    # Phrase bonus.
    #
    # Example:
    #
    # Query:
    # "collector-emitter voltage"
    #
    # Text:
    # "VCEO collector-emitter voltage open base -50 V"
    #
    # The complete phrase appears in the text, so
    # the chunk receives an additional relevance bonus.
    phrase_bonus = 0

    if (
        normalized_query
        and normalized_query in normalized_text
    ):
        phrase_bonus = 2

    return term_score + phrase_bonus


def retrieve_relevant_chunks(
    query,
    chunks,
    top_k=3
):
    """
    Retrieve the most relevant document chunks.

    Args:
        query: User question.
        chunks: List of document chunks.
        top_k: Maximum number of chunks to return.

    Returns:
        List of chunks ordered by relevance.
    """

    scored_chunks = []

    for chunk in chunks:

        score = calculate_relevance_score(
            query,
            chunk["text"]
        )

        if score > 0:

            scored_chunks.append(
                {
                    "chunk": chunk,
                    "score": score
                }
            )

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_chunks[:top_k]
