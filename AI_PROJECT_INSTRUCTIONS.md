## Current Status (Updated)

**RAG v1: Complete.**
**RAG v2: Complete.**

| Phase | Status |
|---|---|
| V2.1.1 Exact Phrase Matching | Done |
| V2.1.2 Technical Term Weighting | Done |
| V2.1.3 Ranking Improvements | Done (Part A: query-aware, deterministic tie-break; Part B length-normalization deferred, not needed yet) |
| V2.1.4 Confidence Scoring | Done (recalibrated: normalized per query/chunk pair, not per raw query, to avoid penalizing natural-language questions) |
| V2.1.5 No-Answer Detection | Done (confidence-threshold gate in main.py) |
| V2.2 Evaluation Framework | Done (golden dataset, 100% accuracy on real document) |
| V2.3 Retrieval Metrics | Done (Precision@K, Recall@K, MRR) |
| Multi-document support | Done (outside formal roadmap; validated with 2 unrelated real documents) |
| main.py RAG pipeline wiring | Done (outside formal roadmap; main.py previously bypassed retrieval entirely) |

92 automated tests passing, zero known regressions.

**Known deferred item:** stopword-aware scoring. Common words
can coincidentally inflate relevance scores. Not currently
causing a measured accuracy problem. Revisit before starting
RAG v3, or sooner if a real false-positive case is observed.

**Next planned phase: RAG v3** (embeddings, vector store,
semantic search), per the RAG v3 Rules section below.

### Testing convention established during V2

For any test that depends on the real PDF and its full
ingestion pipeline (document_loader -> text_cleaner ->
chunker), pair it with a second, synthetic-data test file that
can run without the real PDF or its external dependencies
(pymupdf, Ollama). Naming convention used: `test_X.py` (real
document) alongside `test_X_framework.py` or a synthetic
sibling file. This lets logic be validated independently of
environment/document availability.

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

Do NOT jump directly to embeddings or vector databases.

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

### evaluation.py

Responsible only for:

- Golden dataset definition
- Retrieval quality measurement (accuracy, Precision@K, Recall@K, MRR)
- Report generation

Must not perform retrieval logic itself (uses retriever.py) and must not call the LLM.

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
