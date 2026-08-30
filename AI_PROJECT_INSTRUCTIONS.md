## Current Status (Updated)

**RAG v1: Complete.**
**RAG v2: Complete.**
**RAG v3: Complete (V3.1-V3.4), wired end-to-end and validated with real Ollama runs.**
**RAG v4: Complete (V4.1-V4.5), persistent incremental storage layer.**
**RAG v5: Complete (V5.1-V5.5), browser-based interface for technicians.**
**RAG v6: Complete (V6.1-V6.3), extraction quality and document lifecycle control.**

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
| V5.1 Flask App Skeleton | Done (2026-08-30 - `src/webapp.py`, lazy-singleton state, home page listing loaded documents) |
| V5.2 Ask-a-Question Route | Done (2026-08-30 - `/ask`, friendly error message when Ollama is unavailable, empty-question validation) |
| V5.3 Upload / Replace Route | Done (2026-08-30 - `/upload`, `secure_filename`, `threading.Lock()` around state refresh, warns instead of failing on a zero-chunk PDF) |
| V5.4 One-Click Launcher | Done (2026-08-30 - `start_app.bat`; sets up the virtual environment on first run, opens the browser automatically; deliberate exception to test-first, see Testing convention below) |
| V5.5 Retrieval Fixes | Done (2026-08-30 - lexical safety net in `hybrid_retrieval.py` generalized to handle exact-score ties, not just a single best match; no-answer gate in `main.py` now checks confidence across ALL retrieved chunks, not just the top-ranked one; both confirmed live via browser and validated against the golden dataset with zero regression) |
| V6.1 Delete Document Route | Done (2026-08-30 - `POST /delete` in `webapp.py`, reusing a new `ingestion.delete_document` orchestration function that calls `chunk_store.delete_chunks_by_source` + `vector_store.delete_chunks_by_source` together, mirroring the same layering `add_or_replace_document`/`replace_document_vectors` already established; confirmation prompt via a plain `confirm()`, no JS framework; confirmed live: delete removes the document from the list, disk, and vector store, and a follow-up question about it correctly returns the no-answer fallback) |
| V6.2 Table-Aware Extraction | Done (2026-08-30 - `document_loader.py` gained `_extract_page_text`/`_reconstruct_table`/`_build_column_labels`/`_clean_cell`; detected tables (`page.find_tables()`) are reconstructed into `Symbol: X \| Parameter: Y \| ...: value \| Unit: Z` lines appended after the page's original plain text, purely additive so pages without tables are unaffected; diagnosed first with a real page of NE555N.pdf via `diagnose_tables.py` rather than guessed; 10 new synthetic tests written test-first, confirmed red (`ImportError`) before implementation; also fixed a latent test-infra gap found along the way - `tests/test_document_loader.py` had zero real tests and was silently executing a real PDF read on every `pytest` run because its name did not match the `*_manual.py` ignore glob, renamed to `tests/test_document_loader_manual.py`; validated live: the NE555N.pdf "turn off time" question that returned the no-answer fallback throughout v5.5 now answers correctly, `0.5 µs`, with no regression on the question that already worked) |
| V6.3 Golden Dataset Expansion | Done (2026-08-30 - new `NE555N_EVALUATION_DATASET` in `evaluation.py`, kept separate from `EVALUATION_DATASET` since `test_evaluation.py`/`test_retrieval_metrics_real.py` build their corpus from `sample.pdf` alone and would fail if NE555N.pdf questions were mixed in; new `tests/test_evaluation_ne555n.py` mirrors that same real-document pattern, pointed at NE555N.pdf; both questions from v5.5/v6.2 now pass automatically, including the stricter single-chunk `reciprocal_rank` check, turning a manual browser smoke test into permanent regression coverage) |
| V6.4 Safety-Net Threshold Recalibration | Deliberately not pursued - no new evidence. `LEXICAL_SAFETY_NET_THRESHOLD = 0.35` is still calibrated from only the original two v5.5 cases: V6.2 fixed the `toff` question at the extraction layer, before the safety net is ever consulted, so V6.3 produced no new rescue case to learn from. Revisiting the threshold now would be guessing, not calibrating. |

193 automated tests passing, zero known regressions.

**Known accepted trade-off (measured, not a bug):** unweighted Reciprocal Rank Fusion in hybrid retrieval improves ranking on paraphrased questions (MRR 0.46 vs. 0.42 for lexical alone) but costs ranking quality on literal, datasheet-vocabulary questions (MRR 0.79 vs. a perfect 1.0 for lexical alone) - which are the majority of realistic queries. A dedicated risk check confirmed this never causes the no-answer gate to reject an answerable question. Deliberately left as-is; see README.md's Known Limitations for the full writeup and candidate fixes (weighted RRF, confidence-gated fallback) if this needs revisiting later. Re-run after RAG v4's storage refactor and reproduced the exact same MRR numbers across all 16 questions - confirming v4 changed nothing about retrieval behavior.

**Known accepted risk (RAG v4, documented, not solved):** chunk metadata (SQLite) and the vector store (ChromaDB) are two separate stores updated in sequence, not inside one transaction. A failure between the two writes could leave a document's metadata and vectors out of sync. Accepted for now given a single local user with no concurrent writers.

**Resolved (RAG v5.5, 2026-08-30):** the two retrieval misses opened above were confirmed with `diagnose_retrieval.py` (temporary, never committed) and fixed with two small, targeted, test-first changes. (1) `hybrid_retrieval.py` gained a lexical safety net (`LEXICAL_SAFETY_NET_THRESHOLD = 0.35`) that force-includes a chunk lexical retrieval is highly confident about even when semantic search never found it and RRF fusion would otherwise drop it. (2) A live browser test then exposed a second gap - two chunks tied exactly on lexical score, and the safety net's original "single best match" design only ever looked at `lexical_results[0]`, missing its tied sibling. Generalized to check every chunk tied for the best score. (3) `main.py`'s no-answer confidence gate was changed to check `max(lexical_confidence, semantic_confidence)` across ALL retrieved chunks, not just the top-ranked one, since the safety net intentionally places a rescued chunk in the weakest slot, not rank #1, and the old gate would silently reject it there. All three fixes validated against the isolated original 32-chunk golden dataset (identical MRR/Precision/Recall to the documented baseline in every case) before being committed. 178 tests passing.

**Resolved (RAG v6.2, 2026-08-30): PDF table extraction losing column structure.** The limitation surfaced by v5.5 above is fixed. `diagnose_tables.py` (temporary, never committed) confirmed against a real page of NE555N.pdf that PyMuPDF's `page.find_tables()` does recover the row/column structure that `page.get_text()` throws away - the `toff` row extracted cleanly as `['t\noff', 'Turn off time (5) (V = V )\nreset CC', '', '0.5', '', '', '0.5', '', 'µs']`, with headers split across two merged rows and subscripted characters split onto their own cell line. `document_loader.py` now reconstructs each detected real table (2+ rows, filtering out degenerate 1-row artifacts like a page footer PyMuPDF also detects) into explicit `Symbol: X | Parameter: Y | ...: value | Unit: Z` facts, appended after the page's original plain text - purely additive, so any page without a table is provably unaffected. Validated with 191 tests (10 new, zero regressions) and live: the exact question that returned the no-answer fallback throughout v5.5 ("What is the turn off time of the NE555?") now answers correctly and grounded ("0.5 µs, Source: NE555N, page 5"), with no regression on the question that already worked. Known, accepted remaining gap: a table row where the source PDF packs two symbols into one visual row (e.g. `tr`/`tf`) does not split cleanly - not recoverable from `extract()`'s text alone without per-line bounding boxes, and no worse than pre-v6.2 behavior for that case.

**RAG v6: complete (V6.1-V6.3).** All three planned sub-phases are done and validated: document lifecycle control (delete), table-aware extraction, and permanent regression coverage for the exact bug class v5.5/v6.2 found. V6.4 (safety-net threshold recalibration) was deliberately not pursued - see its own phase-table row above for why. See `claude/rag-v6-plan.md` for the full plan and progress log. Remaining, non-blocking follow-ups: revisiting hybrid retrieval's fusion weighting and tie-break signals more broadly, reconstructing table rows that pack more than one symbol per row, and a fully standalone executable (PyInstaller) so end users don't need Python installed at all.

**Observed during V6.1's real smoke test (2026-08-30, not a bug):** deleting a document and immediately re-uploading a file under the same name, right after a fresh app start, was slow enough that Daniel had to cancel and retry once. Consistent with the already-documented one-time embedding-model cold-load cost (see V5.2's performance detour above) - deleting never touches the embedding model, but the very next upload's `generate_embeddings_for_chunks` call can be the first thing in a fresh process to load `sentence-transformers` from disk, whichever action triggers it first. Two immediate retries in the same running process, and a subsequent full app restart, both completed normally - consistent with a one-time cold-start cost, not a defect introduced by the delete feature. Not pursued further per the project's "don't optimize without a measured bottleneck" principle.

### RAG v6 Rules

Only start after V5 is stable (it now is - see Current Status above).

Focus: fixing the confirmed table-extraction gap from v5.5's real-document test, and giving the technician direct control over their own document library (delete), so an accidental wrong upload never has to be cleaned up by hand.

Sub-phases, in order:

- V6.1 Delete Document Route - done (see phase table above)
- V6.2 Table-Aware Extraction - done (see phase table above): tables detected per page (PyMuPDF `find_tables()`) are reconstructed into structured `Symbol | Parameter | ...: value | Unit` text appended to the page's plain text, with the plain text itself always preserved unchanged as the safe fallback when no real table is detected
- V6.3 Golden Dataset Expansion - done (see phase table above): `NE555N_EVALUATION_DATASET`, a separate dataset from `EVALUATION_DATASET` in `evaluation.py` (kept apart because `test_evaluation.py`/`test_retrieval_metrics_real.py` assert perfect accuracy/MRR against `sample.pdf` alone), NOT a general "one dataset per uploaded document" pattern - this only exists because NE555N.pdf is a permanent, committed reference document, not a real end user's own upload
- V6.4 Safety-Net Threshold Recalibration - deliberately not pursued (see phase table above): V6.2's fix resolved the only new confirmed case at the extraction layer, before the safety net is ever consulted, so there was no new rescue case to calibrate against

Explicitly OUT of scope for RAG v6:

- Full standalone packaging (PyInstaller) - a packaging concern, unrelated to data/extraction quality
- Auto-installing Python/Ollama/the model - already discussed and declined during v5
- A general RRF re-weighting overhaul - not yet justified by data volume

Must remain:

- 100% local
- 100% free
- No JavaScript framework required

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

### Testing convention established during V5

`webapp.py` tests use Flask's `test_client()` (`app.testing =
True`) and `monkeypatch.setattr(webapp, "_state", None)` to
reset the lazy-singleton state between tests. Every pipeline
call the routes make (`_get_state`, `add_or_replace_document`,
`replace_document_vectors`, `load_all_chunks`,
`answer_question`) is monkeypatched at the point of use, the
same convention as V4 - a webapp test verifies routing,
validation, and response rendering, never the real RAG
pipeline underneath it.

`pytest.ini` (`addopts = --ignore-glob=*_manual.py`) excludes
the `tests/*_manual.py` scripts from default collection.
Discovered during V5: pytest imports every `test_*.py` file
during collection, including files with no `def test_...`
functions, so these manual scripts (meant to be run
individually via `python -m tests.test_X_manual`) were making
real Ollama/embedding calls on every plain `pytest` invocation
- the root cause of a full-suite runtime that varied wildly
(28s-233s) across the whole project history. This is a
collection-configuration fix, not a code change to the manual
scripts themselves, which are unaffected and still runnable
individually.

`start_app.bat` (V5.4) is a deliberate, documented exception
to test-first development: it is a Windows batch launcher, not
Python logic, so there is no meaningful way to cover it with
`pytest`. It was instead verified manually in two scenarios -
a fresh install (temporarily renaming `.venv` to simulate a
new user) and a normal run (existing `.venv`) - both confirmed
working before considering V5.4 done.

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
↓
RAG v5

Do NOT jump directly to embeddings, vector databases, or a
user-facing interface before the stage that precedes it is
stable.

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

### RAG v5 Rules

Only start after V4 is stable (it now is - see Current Status above).

Focus: a browser-based interface so a technician never has to
touch a terminal, built entirely on top of RAG v4's
persistence layer - no new retrieval or storage logic.

Includes:

- Flask web app (`src/webapp.py`), single local user, no
  authentication
- Home page listing currently loaded documents
- Ask-a-question route, reusing `main.answer_question` as-is
- Upload/replace-a-document route, reusing
  `ingestion.add_or_replace_document` and
  `ingestion.replace_document_vectors` as-is - v5 saves the
  uploaded bytes to disk and calls v4's functions, it does not
  reimplement ingestion
- A `threading.Lock()` around refreshing the in-memory chunk
  list after an upload, since Flask's dev server can be
  multi-threaded
- A one-click launcher script for end users (`start_app.bat`)

Explicitly OUT of scope for RAG v5 (identified during V5.3's
real-document smoke test, deferred to future work - see Known
Limitations in README.md and the "Known open investigation"
note above):

- Any change to `chunker.py`'s splitting logic (e.g.
  table-aware chunking)
- Any change to `retriever.py`'s scoring or tie-break logic
- Full standalone packaging that removes the Python
  installation requirement (PyInstaller or similar)

Must remain:

- 100% local
- 100% free
- No JavaScript framework required (plain HTML forms are
  sufficient for this project's scope)

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

    def calculate_relevance_score(query, text):
        ...

over:

    def calc(q, t):
        ...

Use:

- Descriptive names
- Docstrings
- Small functions
- Clear responsibilities

## Module Responsibilities

### document_loader.py (RAG v1, RAG v6.2)

Responsible only for:

- Document ingestion
- PDF extraction, including table-aware reconstruction of
  detected tables into structured text (`_reconstruct_table`,
  `_build_column_labels`, `_clean_cell`, RAG v6.2) - always
  additive to the plain-text extraction, never a replacement
  for it
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

### ingestion.py (RAG v4, RAG v6.1)

Responsible only for:

- Orchestrating extraction -> cleaning -> chunking for one document
- Assigning chunk_ids that continue from whatever is already persisted, so multiple documents never collide
- Removing a document's outdated chunks/vectors when it's replaced, or entirely when it's deleted (`delete_document`, RAG v6.1)
- Keeping the chunk repository (chunk_store.py) and the vector store (vector_store.py) in sync for that one document

Must not contain:

- SQL/persistence details (delegated to chunk_store.py)
- Embedding computation itself (delegated to embeddings.py) or low-level vector storage details (delegated to vector_store.py) - this module only calls them in the right order
- Retrieval or LLM logic

### webapp.py (RAG v5, RAG v6.1)

Responsible only for:

- Flask routes: home page (`/`), ask-a-question (`/ask`),
  upload/replace-a-document (`/upload`),
  delete-a-document (`/delete`)
- Lazy-singleton in-memory state (`_get_state`), initialized
  once from `main.initialize_system` and refreshed after an
  upload, guarded by a `threading.Lock()`
- Summarizing loaded documents for display
- Basic input validation (empty question, missing file, non-PDF
  file) and turning pipeline exceptions into a friendly message

Must not contain:

- Retrieval, ranking, or confidence-scoring logic (see
  `hybrid_retrieval.py`, `retriever.py`)
- Ingestion or chunking logic (see `ingestion.py`,
  `chunker.py`) - this module only calls them in the right
  order, the same relationship `ingestion.py` has with
  `chunk_store.py`/`vector_store.py`
- Prompt construction or LLM calls (see `generator.py`,
  `llm.py`)

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
