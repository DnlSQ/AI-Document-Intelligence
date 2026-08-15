from config import SYSTEM_PROMPT
from llm import ask_llm


messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


print("====================================")
print("      AI Document Intelligence")
print("====================================")
print("Type 'exit' to quit.\n")


while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    answer = ask_llm(messages)

    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    print(f"\nAI: {answer}\n")