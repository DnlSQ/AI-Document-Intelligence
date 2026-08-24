from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.retriever import retrieve_relevant_chunks
from src.generator import generate_answer


PDF_PATH = "data/documents/sample.pdf"


QUESTIONS = [
    {
        "id": 2,
        "question": "What is the maximum collector-emitter voltage?"
    },
    {
        "id": 3,
        "question": "What is the maximum input voltage?"
    },
    {
        "id": 4,
        "question": "What is the DC current gain?"
    }
]


# ============================================================
# LOAD DOCUMENT
# ============================================================

pages = extract_text_from_pdf(PDF_PATH)

for page in pages:
    page["text"] = clean_text(page["text"])


# ============================================================
# CREATE CHUNKS
# ============================================================

chunks = create_document_chunks(
    pages,
    chunk_size=1000,
    chunk_overlap=150,
    source=PDF_PATH
)


# ============================================================
# TEST QUESTIONS
# ============================================================

for item in QUESTIONS:

    question_id = item["id"]
    question = item["question"]

    print("\n")
    print("=" * 80)
    print(f"QUESTION {question_id}")
    print("=" * 80)

    print(question)

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    retrieved_chunks = retrieve_relevant_chunks(
        question,
        chunks,
        top_k=3
    )

    print("\n" + "-" * 80)
    print("RETRIEVED CHUNKS")
    print("-" * 80)

    for index, result in enumerate(retrieved_chunks, start=1):

        chunk = result["chunk"]

        print(f"\nRESULT {index}")
        print(f"Score: {result['score']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page']}")
        print(f"Source: {chunk['source']}")

        print("\nChunk text:")
        print(chunk["text"])

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("GENERATING ANSWER WITH QWEN...")
    print("-" * 80)

    try:

        answer = generate_answer(
            question,
            retrieved_chunks
        )

        print("\nFINAL ANSWER")
        print("-" * 80)
        print(answer)

    except Exception as error:

        print("\nERROR")
        print("-" * 80)
        print(type(error).__name__)
        print(str(error))
        