"""
Manual check: how much real risk does the measured hybrid
ranking weakness (see rag-v3-progress.md's Comparative Retrieval
Evaluation) actually pose to main.answer_question's no-answer
confidence gate?

The gate only inspects retrieved_chunks[0] (the RANK-1 hybrid
result) - see main.answer_question:

    top_result = retrieved_chunks[0]
    top_confidence = max(top_result["lexical_confidence"], top_result["semantic_confidence"])
    if top_confidence < MIN_CONFIDENCE_THRESHOLD:
        return NO_CONTEXT_ANSWER

If the genuinely relevant chunk is ranked #2 or #3 instead of #1
(which we already measured happens - see the paraphrase MRR
numbers), this ONLY causes a real problem if the chunk that DID
land at rank #1 has low confidence in BOTH methods - because then
the gate rejects a question that actually had a good answer
available in the retrieved context (generate_answer receives ALL
top_k chunks, not just rank 1, so as long as the relevant one is
retrieved at all, the LLM has it - UNLESS the gate rejects first).

This script checks, for every EVALUATION_DATASET question, under
real hybrid retrieval:

    1. Is the "ground truth" relevant chunk actually retrieved at
       all (in the top_k)?
    2. At what rank?
    3. What is the rank-1 chunk's gate confidence
       (max(lexical_confidence, semantic_confidence))?
    4. Would the gate have rejected this question
       (gate confidence < MIN_CONFIDENCE_THRESHOLD)?

A question is a REAL problem case only if: the relevant chunk WAS
retrieved (at any rank) AND the gate confidence is below
threshold - meaning a real, answerable question got rejected.

Run by hand:

    python -m tests.test_confidence_gate_risk_manual
"""
from src.main import build_chunk_repository
from src.embeddings import generate_embeddings_for_chunks
from src.vector_store import add_chunks_to_store, reset_store, get_collection
from src.hybrid_retrieval import hybrid_retrieve
from src.evaluation import EVALUATION_DATASET, is_relevant_chunk
from src.config import MIN_CONFIDENCE_THRESHOLD


print("=" * 80)
print("BUILDING CHUNK REPOSITORY + VECTOR STORE")
print("=" * 80)

chunks = build_chunk_repository()
embedded_chunks = generate_embeddings_for_chunks(chunks)

collection = get_collection()
reset_store(collection=collection)
add_chunks_to_store(embedded_chunks, collection=collection)

print(f"Loaded {len(chunks)} chunks.\n")
print(f"MIN_CONFIDENCE_THRESHOLD = {MIN_CONFIDENCE_THRESHOLD}\n")

real_problem_cases = []

for case in EVALUATION_DATASET:
    results = hybrid_retrieve(case["question"], chunks, collection=collection, top_k=3)

    relevant_rank = None
    for rank, result in enumerate(results, start=1):
        if is_relevant_chunk(case, result["chunk"]["text"]):
            relevant_rank = rank
            break

    if results:
        top_result = results[0]
        gate_confidence = max(
            top_result["lexical_confidence"],
            top_result["semantic_confidence"]
        )
        would_reject = gate_confidence < MIN_CONFIDENCE_THRESHOLD
    else:
        gate_confidence = 0.0
        would_reject = True

    is_real_problem = (relevant_rank is not None) and would_reject

    if is_real_problem:
        real_problem_cases.append(case["id"])

    style = case.get("style", "?")
    print(
        f"[{style:<10}] {case['id']:<35} "
        f"relevant_rank={relevant_rank} "
        f"gate_confidence={gate_confidence:.2f} "
        f"would_reject={would_reject} "
        f"{'<-- REAL PROBLEM' if is_real_problem else ''}"
    )

print("\n" + "=" * 80)
if real_problem_cases:
    print(f"FOUND {len(real_problem_cases)} REAL PROBLEM CASE(S): {real_problem_cases}")
    print("These are answerable questions the gate would incorrectly reject.")
else:
    print("NO real problem cases found: whenever the gate's rank-1 confidence")
    print("was too low to answer, the relevant chunk wasn't retrieved at all")
    print("either - so the gate isn't causing any additional false rejections")
    print("beyond what a 'no answer available' case already deserves.")
print("=" * 80)
