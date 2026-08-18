from src.generator import generate_answer


def test_generate_answer_uses_retrieved_context(monkeypatch):
    captured_messages = []

    def fake_ask_llm(messages):
        captured_messages.extend(messages)
        return "The maximum collector-emitter voltage is -50 V."

    monkeypatch.setattr(
        "src.generator.ask_llm",
        fake_ask_llm
    )

    retrieved_chunks = [
        {
            "chunk": {
                "chunk_id": 2,
                "page": 2,
                "text": (
                    "VCEO collector-emitter voltage "
                    "open base -50 V"
                )
            },
            "score": 5
        }
    ]

    question = "What is the maximum collector-emitter voltage?"

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    assert answer == (
        "The maximum collector-emitter voltage is -50 V."
    )

    assert len(captured_messages) == 1

    prompt = captured_messages[0]["content"]

    assert question in prompt
    assert "VCEO" in prompt
    assert "-50 V" in prompt
    assert "Page: 2" in prompt
    assert "Chunk ID: 2" in prompt


def test_generate_answer_with_multiple_chunks(monkeypatch):
    captured_messages = []

    def fake_ask_llm(messages):
        captured_messages.extend(messages)
        return "The answer is based on the provided document."

    monkeypatch.setattr(
        "src.generator.ask_llm",
        fake_ask_llm
    )

    retrieved_chunks = [
        {
            "chunk": {
                "chunk_id": 2,
                "page": 2,
                "text": "VCEO collector-emitter voltage -50 V"
            },
            "score": 5
        },
        {
            "chunk": {
                "chunk_id": 3,
                "page": 2,
                "text": "IO output current -500 mA"
            },
            "score": 4
        }
    ]

    question = "What are the maximum voltage and output current?"

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    assert answer == (
        "The answer is based on the provided document."
    )

    prompt = captured_messages[0]["content"]

    assert "Chunk ID: 2" in prompt
    assert "Chunk ID: 3" in prompt
    assert "Page: 2" in prompt
    assert "VCEO" in prompt
    assert "-50 V" in prompt
    assert "IO" in prompt
    assert "-500 mA" in prompt
    assert question in prompt
    