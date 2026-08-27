## Current Status (Updated)

**RAG v1: Complete.**
**RAG v2: Complete.**
**RAG v3: Complete (V3.1-V3.4), wired end-to-end and validated with real Ollama runs.**
**RAG v4: Complete (V4.1-V4.5), persistent incremental storage layer.**

| Phase | Status |
|---|---|
| V2.1.1 Exact Phrase Matching | Done |
| V2.1.2 Technical Term Weighting | Done |
| V2.1.3 Ranking Improvements | Done (Part A: query-aware, deterministic tie-break; Part B length-normalization deferred, not needed yet) |
| V2.1.4 Confidence Scoring | Done (recalibrated: normalized per query/chunk pair, not per raw query, to avoid penalizing natural-language questions) |
| V2.1.5 No-Answer Detection | Done (confidence-threshold gate in main.py) |
| V2.2 Evaluation Framework | Done (golden dataset, expanded 4->16 questions on 2026-08-25 - see below) |
| V2.3 Retrieval Metrics | Done (Precision@K, Recall@K, MRR; generalized on 2026-08-25 to compare lexical/semantic/hybrid) |
| Stopword-Aware Scoring | Done (2026-08-25 - closes the item deferred since V2, triggered by a real observed false positive in the V3.4 end-to-end test) |
| Multi-document support | Done (outside formal roadmap; validated with 2 unrelated real documents) |
| main.py RAG pipeline wiring | Done (retrieval -> generation, fully wired) |
| V3.1 Embedding Generation | Done (`src/embeddings.py`, sentence-transformers/all-MiniLM-L6-v2) |
| V3.2 Vector Store | Done (`src/vector_store.py`, local ChromaDB, `PersistentClient`) |
| V3.3 Semantic Search | Done (`src/semantic_search.py`, cosine similarity) |
| V3.4 Hybrid Retrieval | Done (`src/hybrid_retrieval.py`, Reciprocal Rank Fusion; wired into `main.py` end-to-end) |
| Comparative Retrieval Evaluation | Done (2026-08-25 - `compare_retrieval_methods` measures lexical/semantic/hybrid side by side on the same dataset) |
| Confidence Gate Risk Check | Done (2026-08-25 - confirmed the measured hybrid ranking weakness never causes a false no-answer rejection) |
| V4.1 SQLite Chunk Store | Done (2026-08-27 - `src/chunk_store.py`, persists chunk metadata so it survives restarts) |
| V4.2 Replace-by-Source Ingestion | Done (2026-08-27 - `src/ingestion.py` `add_or_replace_document`; re-uploading a filename replaces its outdated chunks, chunk_ids never reused) |
| V4.3 Incremental Vector Store Updates | Done (2026-08-27 - `vector_store.delete_chunks_by_source` + `ingestion.replace_document_vectors`; only the changed document is re-embedded, not the whole library) |
| V4.4 Startup / Initialization Logic | Done (2026-08-27 - `main.initialize_system` reuses persisted data on a warm start instead of re-ingesting everything; confirmed with a real run: the embedding model isn't even loaded when data is already persisted) |
| V4.5 Dynamic Document Discovery | Done (2026-08-27 - `config.discover_document_paths` scans the documents folder instead of a hardcoded list) |

164 automated tests passing, zero known regressions.

**Known accepted trade-off (measured, not a bug):** unweighted Reciprocal Rank Fusion in hybrid retrieval improves ranking on paraphrased questions (MRR 0.46 vs. 0.42 for lexical alone) but costs ranking quality on literal, datasheet-vocabulary questions (MRR 0.79 vs. a perfect 1.0 for lexical alone) - which are the majority of realistic queries. A dedicated risk check confirmed this never causes the no-answer gate to reject an answerable question. Deliberately left as-is; see README.md's Known Limitations for the full writeup and candidate fixes (weighted RRF, confidence-gated fallback) if this needs revisiting later. Re-run after RAG v4's storage refactor and reproduced the exact same MRR numbers across all 16 questions - confirming v4 changed nothing about retrieval behavior.

**Known accepted risk (RAG v4, documented, not solved):** chunk metadata (SQLite) and the vector store (ChromaDB) are two separate stores updated in sequence, not inside one transaction. A failure between the two writes could leave a document's metadata and vectors out of sync. Accepted for now given a single local user with no concurrent writers.

**RAG v4: complete.** Next priority: **RAG v5**, a technician-facing web interface for uploading PDFs into the persistent library (replacing an outdated document by uploading a new one under the same name) and asking questions through a browser instead of the CLI. Other open, non-blocking follow-ups: growing the evaluation corpus with additional real documents, and revisiting hybrid retrieval's fusion weighting if a larger or more paraphrase-heavy corpus changes the current trade-off.

### Testing convention established during V2

For any test that depends on the real PDF and its full
ingestion pipeline (document_loader -> text_cleaner ->
chunker), pair it with a second, synthetic-data test file that
can run without the real PDF or its external dependencies
(pymupdf, Ollama). Naming convention used: `test_X.py` (real
document) alongside `test_X_framework.py` or a synthetic
sibling file. This lets logic be validated independently of
environment/document availability.

### Testing convention established during V4

Persistence-layer tests (`chunk_store.py`, `ingestion.py`,
`vector_store.py`'s incremental functions) use pytest's
`tmp_path` fixture (SQLite) or a fresh in-memory ChromaDB
`EphemeralClient` with a unique collection name (vector store)
for isolation, instead of touching the real `data/` files.
External pipeline steps (PDF extraction, cleaning, chunking,
embedding generation) are mocked via `monkeypatch` at the
point of use so these tests exercise only the orchestration
logic being added, not steps already covered by their own
modules' test suites.

---
# AI Document Intelligence - Development Instructions

## Purpose

These instructions define how any AI assistant must contribute to the development of the AI Document Intelligence project.

The AI must preserve project consistency, architecture quality, testing discipline, and local-first design.

## Core Mission

Build a professional-grade Retrieval-Augmented Generation (RAG) system completely from scratch.

The project must demonstrate:

- AI Engineering
- RAG Architecture
- LLM Integration
- Software Engineering
- Testing
- Document Intelligence
- Production-oriented design

The project is intended as a portfolio project and learning platform.

## Non-Negotiable Constraints

### No Paid APIs

Never introduce:

- OpenAI API
- Anthropic API
- Gemini API
- Azure OpenAI
- Cohere API
- Pinecone Cloud
- Weaviate Cloud
- Any paid SaaS service

If a proposed solution requires payment:

**REJECT THE SOLUTION**

and propose a local alternative.

### Local First

All capabilities must run locally.

Preferred stack:

- Python
- Ollama
- Qwen 2.5 7B
- FAISS
- ChromaDB Local
- SQLite
- Sentence Transformers

The system must remain usable without internet access.

### Open Source Only

Prefer:

- MIT
- Apache 2.0
- BSD
- Open-source projects

Avoid vendor lock-in.

## Development Methodology

Every new feature must follow:
Design
↓
Tests
↓
Implementation
↓
Validation
↓
Refactor
↓
Commit

Never skip testing.

### Test-First Development

Before implementing new functionality:

1. Define expected behavior.
2. Create tests.
3. Run tests.
4. Observe failures.
5. Implement solution.
6. Verify tests pass.

Preferred philosophy:
Failing Tests
↓
Implementation
↓
Passing Tests

### Never Break Existing Features

Before modifying any module: pytest

After modification: pytest

The goal is:

Existing tests remain green.

New functionality must not break existing functionality.

## Retrieval Quality Before Complexity

The project must evolve in phases.

Correct order:
RAG v1
↓
RAG v2
↓
RAG v3
↓
RAG v4

Do NOT jump directly to embeddings, vector databases, or a
user-facing interface.

### RAG v1 Rules

Completed stage.

Includes:

- PDF ingestion
- Cleaning
- Chunking
- Lexical retrieval
- Generation
- Grounding
- Source attribution
- Testing

RAG v1 must remain stable.

### RAG v2 Rules

Completed stage (see Current Status above).

Focus:

- Retrieval quality
- Ranking quality
- Relevance scoring
- Evaluation
- Confidence estimation

Before introducing semantic retrieval.

### RAG v2 Development Order

Implement in this sequence:
V2.1 Better Retrieval
│
├── Exact Phrase Matching
├── Technical Term Weighting
├── Ranking Improvements
├── Confidence Scoring
└── No-Answer Handling

Then: V2.2 Evaluation Framework

Then: V2.3 Retrieval Metrics

### RAG v3 Rules

Only start after V2 is stable (it now is - see Current Status above).

Includes:

- Embeddings
- Vector Store
- Semantic Search
- Hybrid Retrieval

Must remain:

- 100% local
- 100% free

### RAG v4 Rules

Only start after V3 is stable (it now is - see Current Status above).

Focus: persistence and incremental updates, so the system can
hold a growing document library without paying the full
ingestion + embedding cost again on every restart or every new
upload.

Includes:

- SQLite chunk metadata store
- Replace-by-source ingestion (same filename = replace, never
  coexist with the outdated version)
- Incremental vector store updates (only the changed document
  is re-embedded)
- Startup logic that reuses persisted data instead of always
  rebuilding
- Dynamic document discovery (scan the documents folder,
  don't hardcode paths)

Explicitly OUT of scope for RAG v4 (belongs to RAG v5 instead):

- Any web framework, HTTP route, or HTML template
- Saving uploaded file bytes to disk (v4's ingestion functions
  assume the PDF path already exists on disk)

Must remain:

- 100% local
- 100% free

## Grounding Requirements

The model must:

- Use only retrieved document information.
- Never invent facts.
- Never use external knowledge.
- Never fill gaps with assumptions.

If the answer is unavailable:

Return: I don't have enough information in the provided document.

## Technical Data Preservation

The system must preserve:

- Numerical values
- Positive signs
- Negative signs
- Units
- Technical identifiers
- Conditions
- Specifications

Example: -50 V

must never become: 50 V

## Retrieval Design Principles

Ranking should prioritize:

- Technical identifiers
- Exact phrases
- Highly relevant chunks

over:

- Generic words
- Stop words
- Coincidental matches

Example: VCEO

should be considered more relevant than:
what
is
the

## Explainability

Every retrieval decision should be explainable.

Avoid:

- Magic numbers
- Opaque heuristics
- Hidden logic

Prefer:

- Documented scoring
- Named constants
- Clear algorithms

## Maintainability

Code should prioritize:

- Readability
- Simplicity
- Modularity
- Testability

over:

- Premature optimization
- Overengineering
- Complex abstractions

## Performance Philosophy

Priority order:
Correctness
↓
Grounding
↓
Retrieval Quality
↓
Maintainability
↓
Performance

Do not optimize unless a measurable bottleneck exists.

## Coding Standards

Prefer:

```python
def calculate_relevance_score(query, text):
    ...
```

over:

```python
def calc(q, t):
    ...
```

Use:

- Descriptive names
- Docstrings
- Small functions
- Clear responsibilities

## Module Responsibilities

### document_loader.py

Responsible only for:

- Document ingestion
- PDF extraction
- Metadata preservation

### text_cleaner.py

Responsible only for:

- Text normalization
- Noise removal

### chunker.py

Responsible only for:

- Chunk creation
- Chunk metadata
- Chunk boundaries

### retriever.py

Responsible only for:

- Scoring
- Ranking
- Retrieval

Must not contain:

- LLM logic
- Prompt construction

### generator.py

Responsible only for:

- Prompt creation
- Context assembly
- Answer generation

Must not perform retrieval.

### llm.py

Responsible only for:

- Communication with Ollama
- Model interaction

### embeddings.py

Responsible only for:

- Loading the local sentence-embedding model
- Converting text into semantic embedding vectors

Must not contain:

- Retrieval, ranking, or similarity search logic
- LLM logic or prompt construction

### vector_store.py

Responsible only for:

- Persisting chunk embeddings (and their metadata) to a local vector store
- Basic storage operations: add, count, reset, delete-by-source

Must not contain:

- Embedding generation (see embeddings.py - this module only stores vectors it's given, it never computes them)
- Similarity search / query logic (see semantic_search.py)

### semantic_search.py

Responsible only for:

- Embedding a user question
- Querying the vector store for the most similar chunks
- Shaping results to match the existing retrieval result format (chunk / score / confidence), so this can be compared against - and combined with - the lexical retriever

Must not contain:

- Embedding generation logic (see embeddings.py)
- Vector storage logic (see vector_store.py)
- LLM logic or prompt construction

### hybrid_retrieval.py

Responsible only for:

- Combining lexical retrieval (retriever.py) and semantic search (semantic_search.py) results into a single ranked list, using Reciprocal Rank Fusion (RRF)

Must not contain:

- Lexical scoring logic (see retriever.py)
- Embedding generation or vector search logic (see embeddings.py, vector_store.py, semantic_search.py)
- LLM logic or prompt construction

### evaluation.py

Responsible only for:

- Golden dataset definition
- Retrieval quality measurement (accuracy, Precision@K, Recall@K, MRR) - independently for lexical (retriever.py), semantic (semantic_search.py), and hybrid (hybrid_retrieval.py) retrieval
- Report generation

Must not perform retrieval logic itself (uses the retrieval modules above) and must not call the LLM.

### chunk_store.py (RAG v4)

Responsible only for:

- Storing chunk metadata (chunk_id, page, text, source) in SQLite
- Loading all stored chunks back into memory
- Deleting all chunks belonging to a given document (by source)
- Computing the next available chunk_id

Must not contain:

- Retrieval, scoring, or ranking logic
- LLM or prompt logic
- PDF extraction, cleaning, or chunking logic

### ingestion.py (RAG v4)

Responsible only for:

- Orchestrating extraction -> cleaning -> chunking for one document
- Assigning chunk_ids that continue from whatever is already persisted, so multiple documents never collide
- Removing a document's outdated chunks/vectors when it's replaced
- Keeping the chunk repository (chunk_store.py) and the vector store (vector_store.py) in sync for that one document

Must not contain:

- SQL/persistence details (delegated to chunk_store.py)
- Embedding computation itself (delegated to embeddings.py) or low-level vector storage details (delegated to vector_store.py) - this module only calls them in the right order
- Retrieval or LLM logic

## Git Workflow

For every completed feature:
git status
git add .
git commit -m "feat: description"
git push

Commit messages should follow:

- feat:
- fix:
- refactor:
- test:
- docs:
- chore:

Examples:
feat: add exact phrase matching
feat: implement confidence scoring
test: add retriever ranking tests
fix: preserve negative values in responses

## Required Workflow For AI Assistants

Whenever proposing a new feature:

### Step 1

Explain:

Why the feature is needed.

### Step 2

Explain:

What files will change.

### Step 3

Create tests first.

### Step 4

Implement code.

### Step 5

Run tests.

### Step 6

Review architecture impact.

## Forbidden Behaviors

Never:

- Introduce paid dependencies.
- Break existing tests.
- Remove grounding protections.
- Introduce hallucination-friendly prompts.
- Skip testing.
- Replace local models with cloud models.
- Suggest solutions that require subscriptions.

## End Goal

The final system should be capable of:
Load Documents
↓
Clean Text
↓
Create Chunks
↓
Retrieve Evidence
↓
Rank Evidence
↓
Generate Grounded Answers
↓
Provide Sources

while remaining:

- 100% Local
- 100% Free
- 100% Reproducible
- 100% Testable

and demonstrating skills expected from:

- AI Engineer
- RAG Engineer
- LLM Engineer
- AI Agent Engineer
- Generative AI Developer
