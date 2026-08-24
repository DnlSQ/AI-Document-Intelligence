import re

# Additional relevance weight applied to technical terms.
TECHNICAL_TERM_WEIGHT = 3

# Bonus applied when the complete normalized query phrase
# appears in the document text.
EXACT_PHRASE_BONUS = 2

# Extra bonus if a chunk contains at least one technical term hit
TECHNICAL_CHUNK_BONUS = 1

# Maximum length for a bare all-uppercase token to be considered a
# technical abbreviation (e.g. VCEO, ICBO, hFE-style identifiers are
# short by nature). This prevents ordinary words from being
# misclassified as technical when a document section happens to be
# formatted in all caps (e.g. headers, warnings, datasheet titles).
MAX_TECHNICAL_ABBREVIATION_LENGTH = 6


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

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_technical_term(token):
    """
    Determine whether a token should be considered
    a technical term.

    Args:
        token: Token to evaluate.

    Returns:
        True if the token is considered technical.
    """
    if not token:
        return False

    # Letter + number combinations
    if re.search(r"[a-zA-Z]", token) and re.search(r"\d", token):
        return True

    # Multi-letter technical abbreviations
    if re.fullmatch(r"[A-Za-z]{2,}", token):
        # Fully uppercase abbreviation. Only short tokens qualify,
        # since real identifiers (VCEO, ICBO) are short; longer
        # all-caps words are more likely to be ordinary words in
        # an all-caps document section rather than identifiers.
        if token.isupper() and len(token) <= MAX_TECHNICAL_ABBREVIATION_LENGTH:
            return True
        # Mixed-case identifier such as hFE or VCEsat. This must
        # exclude fully-uppercase tokens, since token[1:] of an
        # all-caps word is also all-caps and would otherwise
        # match here regardless of the length gate above.
        if not token.isupper() and any(char.isupper() for char in token[1:]):
            return True

    return False


def is_technical_match_in_text(normalized_token, text):
    """
    Determine whether a query token corresponds to a technical
    term as it actually appears in the document text.

    The document is treated as the source of truth for casing:
    a query written in lowercase (e.g. "vceo") must still be
    recognized as technical if the document contains it as an
    identifier (e.g. "VCEO").

    Args:
        normalized_token: Lowercase token from the query.
        text: Original (non-lowercased) document text.

    Returns:
        True if any occurrence of the token in the text is
        considered a technical term.
    """
    occurrences = re.findall(
        r"\b" + re.escape(normalized_token) + r"\b",
        text,
        flags=re.IGNORECASE
    )

    return any(is_technical_term(occurrence) for occurrence in occurrences)


def calculate_relevance_score(query, text):
    """
    Calculate a relevance score using:

    1. Lexical term matching.
    2. Technical term weighting.
    3. Exact phrase matching.
    """
    query_tokens = tokenize(query)
    text_tokens = set(tokenize(text))

    score = 0
    technical_hit = False

    for normalized_token in query_tokens:
        if normalized_token not in text_tokens:
            continue

        if is_technical_match_in_text(normalized_token, text):
            score += TECHNICAL_TERM_WEIGHT
            technical_hit = True
        else:
            score += 1

    # Exact phrase bonus
    normalized_query = normalize_text(query)
    normalized_text = normalize_text(text)
    if normalized_query and normalized_query in normalized_text:
        score += EXACT_PHRASE_BONUS

    # Bonus if chunk contains at least one technical term hit
    if technical_hit:
        score += TECHNICAL_CHUNK_BONUS

    return score


def count_technical_term_matches(query, text):
    """
    Count how many query terms found in the text are technical
    terms as they appear in the document.

    Used as a ranking tie-breaker so chunks are preferred when
    the QUERY's technical terms specifically appear in them -
    not simply because the chunk happens to contain some
    unrelated technical-looking word elsewhere in its text.

    Args:
        query: User question.
        text: Chunk text.

    Returns:
        Integer count of query-relevant technical term matches.
    """
    query_tokens = tokenize(query)
    text_tokens = set(tokenize(text))

    count = 0

    for normalized_token in query_tokens:
        if normalized_token not in text_tokens:
            continue

        if is_technical_match_in_text(normalized_token, text):
            count += 1

    return count


def calculate_max_possible_score(query, text):
    """
    Calculate the maximum relevance score achievable for a
    specific (query, chunk) pair, used as the denominator when
    normalizing scores into a confidence value.

    Only query tokens that actually appear in the chunk's text
    are counted, each at the best-case (technical) weight, plus
    the exact phrase and technical chunk bonuses (each achievable
    once). This avoids penalizing confidence for query tokens
    (e.g. stopwords like "what"/"is"/"the" in a natural-language
    question) that could never have matched this chunk in the
    first place, no matter how relevant the chunk actually is.

    Args:
        query: User question.
        text: Chunk text being scored against.

    Returns:
        Integer upper bound on the relevance score for this
        (query, chunk) pair. Zero if no query token matches.
    """
    query_tokens = tokenize(query)
    text_tokens = set(tokenize(text))

    matched_tokens = [
        token for token in query_tokens if token in text_tokens
    ]

    if not matched_tokens:
        return 0

    max_score = len(matched_tokens) * TECHNICAL_TERM_WEIGHT
    max_score += EXACT_PHRASE_BONUS
    max_score += TECHNICAL_CHUNK_BONUS

    return max_score


def calculate_confidence(score, query, text):
    """
    Convert a raw relevance score into a normalized confidence
    value in the range [0.0, 1.0] for a specific (query, chunk)
    pair.

    Args:
        score: Relevance score produced by calculate_relevance_score.
        query: The user question the score was computed for.
        text: Chunk text the score was computed against.

    Returns:
        Float confidence value, rounded to 2 decimal places.
        0.0 if no query token matched this chunk.
    """
    max_possible_score = calculate_max_possible_score(query, text)

    if max_possible_score == 0:
        return 0.0

    confidence = score / max_possible_score

    return round(min(confidence, 1.0), 2)


def retrieve_relevant_chunks(query, chunks, top_k=3):
    """
    Retrieve the most relevant document chunks.

    Chunks are ranked by:

    1. Relevance score (primary signal).
    2. Number of query-relevant technical term matches
       (tie-break; a chunk only benefits from a technical term
       if that term was actually part of the query).
    3. Chunk ID, ascending (final tie-break; guarantees a fully
       deterministic order regardless of input order).

    Each result also includes a normalized "confidence" value
    (see calculate_confidence) alongside the raw "score".
    """
    scored_chunks = []

    for chunk in chunks:
        score = calculate_relevance_score(query, chunk["text"])
        if score > 0:
            scored_chunks.append({
                "chunk": chunk,
                "score": score,
                "confidence": calculate_confidence(score, query, chunk["text"])
            })

    scored_chunks.sort(
        key=lambda item: (
            item["score"],
            count_technical_term_matches(query, item["chunk"]["text"]),
            -item["chunk"]["chunk_id"]
        ),
        reverse=True
    )

    return scored_chunks[:top_k]
