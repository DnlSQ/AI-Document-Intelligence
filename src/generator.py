from src.llm import ask_llm


def generate_answer(question, retrieved_chunks):
    """
    Generate an answer using the user's question and
    the chunks retrieved from the document.

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
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the information provided
in the context below.

Do not use external knowledge.
Do not invent information.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided document."

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    return ask_llm(messages)
