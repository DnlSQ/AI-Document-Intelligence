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


def calculate_relevance_score(query, text):
    """
    Calculate a simple lexical relevance score.

    The score represents how many query terms
    are present in the chunk text.

    Args:
        query: User question.
        text: Chunk text.

    Returns:
        Integer relevance score.
    """

    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))

    return len(query_tokens.intersection(text_tokens))


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
