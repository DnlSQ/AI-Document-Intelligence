from ollama import chat
from config import MODEL_NAME


def ask_llm(messages):
    response = chat(
        model=MODEL_NAME,
        messages=messages
    )

    return response.message.content