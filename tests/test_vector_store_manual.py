"""
Manual check that the REAL persistent vector store (ChromaDB on
disk, not the in-memory EphemeralClient used in test_vector_store.py)
actually survives being reopened - the whole point of a vector
store is that it doesn't need to be rebuilt every time the app
starts.

Run by hand:

    python -m tests.test_vector_store_manual
"""
from src.embeddings import generate_embeddings_for_chunks
from src.vector_store import get_collection, add_chunks_to_store, get_chunk_count


sample_chunks = [
    {
        "chunk_id": 1,
        "page": 2,
        "text": "VCEO collector-emitter voltage -50 V",
        "source": "data/documents/sample.pdf",
    },
    {
        "chunk_id": 2,
        "page": 3,
        "text": "IO output current -500 mA",
        "source": "data/documents/sample.pdf",
    },
]

print("=" * 80)
print("VECTOR STORE MANUAL CHECK")
print("=" * 80)

print("\nGenerating real embeddings for 2 sample chunks...")
embedded_chunks = generate_embeddings_for_chunks(sample_chunks)

print("Writing to the real persistent store (data/vector_store)...")
add_chunks_to_store(embedded_chunks)

print("\nSimulating an app restart: asking for a FRESH collection handle...")
reopened_collection = get_collection()
count = get_chunk_count(collection=reopened_collection)

print(f"Chunks found after reopening: {count}")

if count >= 2:
    print("\nPASS: data survived reopening the store - real disk persistence confirmed.")
else:
    print("\nUNEXPECTED: expected at least 2 chunks, found fewer.")
    