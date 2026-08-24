from src.document_loader import extract_text_from_pdf
from src.text_cleaner import clean_text
from src.chunker import create_document_chunks
from src.retriever import retrieve_relevant_chunks


PDF_PATH = "data/documents/sample.pdf"


def load_chunks():
    pages = extract_text_from_pdf(PDF_PATH)

    for page in pages:
        page["text"] = clean_text(page["text"])

    return create_document_chunks(
        pages,
        chunk_size=1000,
        chunk_overlap=150,
        source=PDF_PATH
    )


questions = [
    "What is the maximum output current of the PDTB113ZT?",
    "What is the maximum collector-emitter voltage?",
    "What is the maximum input voltage?",
    "What is the DC current gain?"
]


chunks = load_chunks()


for number, question in enumerate(questions, start=1):

    print("\n" + "=" * 80)
    print(f"QUESTION {number}")
    print("=" * 80)
    print(question)

    results = retrieve_relevant_chunks(
        question,
        chunks,
        top_k=3
    )

    for rank, result in enumerate(results, start=1):

        chunk = result["chunk"]
        score = result["score"]

        print("\n" + "-" * 80)
        print(f"RESULT {rank}")
        print(f"Score: {score}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page']}")
        print(f"Source: {chunk['source']}")
        print("-" * 80)

        print(chunk["text"])
        