from ollama import chat


def ask_llm(messages):
    response = chat(
        model="qwen2.5:7b",
        messages=messages
    )

    return response.message.content


messages = []

print("====================================")
print("      Local AI Assistant")
print("      Model: Qwen 2.5 7B")
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