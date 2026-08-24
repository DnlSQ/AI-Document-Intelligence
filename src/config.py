MODEL_NAME = "qwen2.5:7b"

# Documents loaded into the knowledge base at startup. The system
# searches across ALL of these together in a single combined
# repository - the retriever naturally scores chunks from
# off-topic documents at 0 (filtered out), so unrelated documents
# don't need to be selected manually per question.
DOCUMENT_PATHS = [
    "data/documents/sample.pdf",
    "data/documents/plantas.pdf",
]

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
