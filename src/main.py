from ollama import chat


def ask_llm(question):
    response = chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.message.content


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

    answer = ask_llm(question)

    print(f"\nAI: {answer}\n")