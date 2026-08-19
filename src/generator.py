from src.config import SYSTEM_PROMPT
from src.llm import ask_llm


def generate_answer(question, retrieved_chunks):
    """
    Generate a grounded answer using the user's question
    and the chunks retrieved from the document.

    Args:
        question: User question.
        retrieved_chunks: List of retrieval results.
            Each result must contain:
                {
                    "chunk": {...},
                    "score": int
                }

    Returns:
        String containing the LLM-generated answer.
    """

    context_parts = []

    for result in retrieved_chunks:
        chunk = result["chunk"]

        context_parts.append(
            f"Page: {chunk['page']}\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Source: {chunk['source']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
DOCUMENT CONTEXT
================

{context}

================

USER QUESTION
=============

{question}

================

INSTRUCTIONS
============

Answer the question using only the document context above.

Important:
- Preserve all numerical values exactly.
- Preserve positive and negative signs exactly.
- Preserve units exactly.
- Do not convert, reinterpret, or "correct" technical values.
- Do not use information that is not present in the document.
- If the answer is not explicitly supported by the context, say:
  "I don't have enough information in the provided document."

Provide the answer first.

Then provide the source using this format:

Source: [document name], page [number]
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    return ask_llm(messages)
