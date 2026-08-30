# AI Document Intelligence

> A local AI assistant designed to help technicians
> retrieve and understand technical documentation
> during equipment troubleshooting.

## Project Motivation

This project was inspired by my experience working as a
Module Technician at Intel.

During equipment troubleshooting, technicians often need
to consult datasheets, equipment manuals, procedures and
technical documentation to diagnose failures and restore
equipment as quickly as possible.

This led to the idea of developing a local AI assistant
capable of processing technical documentation and helping
technicians retrieve relevant information during repairs.

## Problem

Technical troubleshooting can require searching through
large amounts of documentation.

This can increase:

- Diagnosis time
- Equipment downtime
- Time spent searching for technical information
- Dependence on individual experience

## Project Goal

Build a local AI-powered document intelligence system that
can:

- Ingest technical documents
- Extract and clean text
- Index technical information
- Retrieve relevant information
- Generate contextual answers using a local LLM
- Provide references to the original documentation
- Evaluate and measure its own retrieval quality
- Search across multiple, unrelated documents at once
- Persist its knowledge base across restarts, updating it
  incrementally as documents are added or replaced
- Let a technician upload documents and ask questions from
  a browser, with no command line required

## Real-World Background

During my experience at Intel, I contributed to improving
equipment availability from approximately 75% to more than
93% through:

- Process data analysis
- Identification of recurring failures
- Technical improvements
- Equipment troubleshooting
- Continuous improvement initiatives

I also developed Excel and Power BI tools to monitor
equipment performance, identify failure trends and support
data-driven decision making.

AI Document Intelligence is an evolution of this experience,
combining industrial troubleshooting, data analysis and
Artificial Intelligence.

## Getting Started (for end users)

AI Document Intelligence runs entirely on your own computer.
No account, no cloud service, and no internet connection is
required to use it once set up - your documents never leave
your machine.

### Requirements (one-time, before first use)

1. **Python 3.10+** - https://www.python.org/downloads/
   (make sure "Add python.exe to PATH" is checked during
   installation)
2. **Ollama** - https://ollama.com/download
3. **The Qwen2.5:7B model**, pulled once through Ollama:
ollama pull qwen2.5:7b

These three are separate programs, not something this
project can install for you - the same is true of any local
LLM tool (LM Studio, GPT4All, Ollama itself), not a
limitation specific to this project.

### Running the app

1. Download or clone this repository.
2. Double-click `start_app.bat` (or a shortcut to it - right
   click the file, "Create shortcut", and move that shortcut
   wherever is convenient).
   - The first time, it creates a Python virtual environment
     and installs every required library automatically. This
     only happens once and can take a few minutes.
   - Every time after that, it starts in seconds.
3. Your browser opens automatically at
   `http://127.0.0.1:5000`, ready to ask questions and
   upload documents. Closing the console window that opens
   alongside it stops the app.

## Architecture

Current architecture (RAG v5):
Browser (technician)
│
├── Upload / replace a document
│ │
│ ▼
│ PDF
│ │
│ ▼
│ Document Loader
│ │
│ ▼
│ Text Cleaner
│ │
│ ▼
│ Chunker (per document, globally unique chunk IDs)
│ │
│ ▼
│ Chunk Repository (SQLite) + Vector Store (ChromaDB)
│ updated incrementally - replacing a document by
│ filename deletes its old chunks/vectors first
│
└── Ask a question
│
▼
Chunk Repository (persisted, survives restarts)
│
┌───────────┴────────────┐
▼ ▼
Lexical Retriever Embeddings (sentence-transformers)
├── Term + technical- │
│ term weighting ▼
├── Exact phrase Vector Store (ChromaDB, local, offline)
│ matching │
└── Confidence scoring Semantic Search (cosine similarity)
│ │
└──────────┬───────────┘
▼
Hybrid Retrieval (Reciprocal Rank Fusion)
│
▼
No-Answer Gate (skips the LLM call when confidence
is too low)
│
▼
Generator
│
▼
Qwen 2.5 7B (Ollama)
│
▼
Grounded Answer + Source Attribution
│
▼
Rendered back in the browser

Quality is measured independently of the pipeline above via
an **Evaluation Framework**: a 16-question golden dataset
(12 questions using the datasheet's own vocabulary, 4
deliberately paraphrased in natural language) plus
Precision@K / Recall@K / Mean Reciprocal Rank, computed
**separately for lexical, semantic, and hybrid retrieval** so
the three strategies can be compared directly instead of
assumed.

## Technologies

- Python
- Flask (local web interface)
- Ollama
- Qwen 2.5 7B
- PyMuPDF
- sentence-transformers (local embedding model)
- ChromaDB (local, persistent vector store)
- SQLite (persistent chunk metadata)
- pytest (173 automated tests)
- Git
- PowerShell
- Local LLM inference

## Current Status

**RAG v1: Complete.**
**RAG v2: Complete (V2.1 - V2.3).**
**RAG v3: Complete (V3.1 - V3.4), wired end-to-end and validated with real Ollama runs.**
**RAG v4: Complete (V4.1 - V4.5), persistent incremental storage layer.**
**RAG v5: Complete (V5.1 - V5.4), browser-based interface for technicians.**

The system ingests one or more PDF documents, cleans and
chunks their text, and retrieves the most relevant passages
for a question using **hybrid retrieval**: lexical scoring
(term/technical-term/phrase matching, stopword-aware) fused
via Reciprocal Rank Fusion with semantic search over local
sentence-transformer embeddings stored in a local ChromaDB
vector store. It estimates its own confidence in each match
and generates a grounded answer using a local LLM - refusing
to answer when no document evidence supports a response.

The system also supports searching across **multiple,
unrelated documents at once**: each document's chunks are
scored independently, so an irrelevant document's content is
naturally filtered out without manual document selection.

As of RAG v4, the chunk repository and vector store persist
across restarts instead of being rebuilt from the source PDFs
every time the system starts: chunk metadata is stored in
SQLite (`data/chunk_store.db`) and the ChromaDB vector store
is reused as-is whenever data already exists. Adding or
replacing a document only re-processes and re-embeds that
one document, not the whole library, via a replace-by-filename
mechanism: uploading a file under an existing name deletes its
outdated chunks and vectors first, so an outdated document can
never coexist with, or be mistaken for, its replacement.

As of RAG v5, all of this is reachable from a browser instead
of the command line: a technician can ask a question, see
which documents are currently loaded, and upload a new PDF (or
replace an existing one by uploading it again under the same
name) without touching a terminal. A one-click launcher
(`start_app.bat`) sets up the Python environment automatically
on first run and opens the browser for every run after that -
Python and Ollama itself still need to be installed once, the
same as any local-LLM tool, but no Python knowledge or command
typing is required beyond that. This was validated end-to-end
with a real, previously-unseen document (an NE555 timer
datasheet): uploading it through the browser made it
searchable immediately, with no restart needed.

Retrieval quality is measured automatically and
comparatively, not just assumed: **173 automated tests**
currently pass, including a golden-dataset evaluation across
16 questions (12 matching the document's own technical
vocabulary, 4 deliberately paraphrased in natural language)
and IR-style retrieval metrics (Precision@K, Recall@K, Mean
Reciprocal Rank) computed independently for lexical, semantic,
and hybrid retrieval. This comparison surfaced a real, measured
trade-off - see Known Limitations below - which was
investigated with a dedicated risk check before deciding how
to respond to it.

## Current Progress

- [x] Local LLM setup
- [x] Project configuration
- [x] LLM abstraction
- [x] PDF ingestion (single and multi-document)
- [x] PDF text extraction
- [x] Text cleaning
- [x] Text chunking
- [x] Lexical retrieval (term matching, technical-term
      weighting, exact phrase matching, stopword-aware
      scoring, deterministic ranking)
- [x] Confidence scoring
- [x] No-answer detection (grounding fallback)
- [x] RAG pipeline (retrieval -> generation, fully wired)
- [x] Source attribution
- [x] Evaluation framework (golden dataset + accuracy report)
- [x] Retrieval metrics (Precision@K, Recall@K, MRR)
- [x] Multi-document support
- [x] Semantic embedding generation (RAG v3, V3.1)
- [x] Vector database (RAG v3, V3.2 - ChromaDB, local)
- [x] Semantic search (RAG v3, V3.3)
- [x] Hybrid retrieval (RAG v3, V3.4 - Reciprocal Rank
      Fusion, wired end-to-end into the live pipeline)
- [x] Comparative retrieval evaluation (lexical vs. semantic
      vs. hybrid, broken down by question style)
- [x] Persistent, incremental document storage (RAG v4 -
      SQLite chunk store, incremental vector store updates
      per document, startup reuse of persisted data, dynamic
      document discovery)
- [x] Technician-oriented interface (RAG v5 - browser-based:
      ask questions, view loaded documents, upload or replace
      documents, one-click launcher script)

## Known Limitations

- **Unweighted hybrid retrieval fusion is a measured
  trade-off, not a bug:** Reciprocal Rank Fusion currently
  weights lexical and semantic retrieval equally. Measured on
  a 16-question golden dataset: this improves ranking on
  paraphrased, natural-language questions (Mean Reciprocal
  Rank 0.46 vs. 0.42 for lexical alone) but costs ranking
  quality on literal, datasheet-vocabulary questions (MRR
  0.79 vs. a perfect 1.0 for lexical alone) - which make up
  the majority of realistic troubleshooting queries. A
  dedicated risk check confirmed this never causes the system
  to reject an answerable question (the no-answer confidence
  gate stayed comfortably above threshold in every measured
  case), and the generator receives every retrieved chunk as
  context regardless of exact rank, so answer correctness is
  governed mainly by recall, which stayed healthy throughout.
  Left unweighted deliberately for now; candidates for a
  future fix (weighted RRF, or falling back to hybrid only
  when lexical confidence is low) are documented but not
  implemented, since the measured real-world impact doesn't
  currently justify the engineering cost.
- **Confidence is a lexical/semantic signal, not a true
  probability:** it can't perfectly distinguish a real (if
  generic) match from a coincidental one. It catches
  clear-cut weak matches; this is an inherent limit of both
  retrieval methods, not something either alone fully solves.
- **Chunk metadata (SQLite) and the vector store (ChromaDB)
  are two separate stores, updated in sequence rather than in
  one transaction:** a failure between the two writes could
  leave a document's metadata and its vectors out of sync.
  Accepted for now given a single local user with no
  concurrent writers - the same kind of measured, documented
  trade-off as the hybrid retrieval fusion weighting above,
  not an oversight.
- **A real-world test with a new, previously-unseen document
  (an NE555 timer datasheet) surfaced two retrieval misses,
  currently under investigation, not yet resolved:** a
  question about a value that lives inside a multi-column
  table returned "I don't have enough information" even
  though the value is present in the document - likely
  because the chunker splits text by character count without
  understanding tables, so a value and its label can end up in
  different chunks, with neither one containing the complete
  fact. A second question returned an unrelated marketing
  bullet instead of the correct numeric specification - likely
  because it tied in lexical score with the chunk that actually
  had the answer, and the deterministic tie-break (lowest
  chunk ID wins) happened to favor the wrong one. Both
  hypotheses come from reading the actual retrieval code, not
  guesswork, but are being confirmed with a dedicated
  diagnostic script before any fix is proposed - consistent
  with this project's "evidence, not assumptions" approach to
  every previous bug.

## Design Philosophy

The project is intentionally designed to run locally,
without requiring paid AI APIs.

This allows technical documentation to remain within
the local environment and demonstrates how LLM-based
applications can be developed with local models.

Every feature is developed test-first: tests are written
before implementation, and the full suite is run after every
change to guarantee no regressions. The one deliberate
exception is `start_app.bat` (RAG v5.4): it is an operating
system launcher script, not Python logic, so there is no
meaningful way to cover it with `pytest` - it was instead
verified manually, once for a first-time setup and once for
a normal run, exactly as documented in its own section above.

## Future Vision

RAG v5 is now complete: a technician can ask questions,
upload or replace documents, and start the whole system with
a single double-click, with no command line involved beyond
a one-time Python/Ollama setup shared by any local-LLM tool.

The long-term goal remains transforming the project into a
technical troubleshooting assistant capable of helping
technicians:

1. Identify relevant documentation
2. Search technical specifications
3. Investigate recurring failures
4. Retrieve troubleshooting procedures
5. Reduce diagnosis time
6. Improve equipment recovery time

The next concrete milestone is investigating and addressing
the two retrieval misses documented above under Known
Limitations, using the NE555 datasheet as a new, real test
case - likely improvements to how the chunker handles
tabular data, and to how the retriever breaks ties between
chunks with an identical lexical score. Growing the evaluation
corpus with additional real documents, and a fully
standalone executable (no separate Python installation
required, most likely via PyInstaller) remain open,
non-blocking follow-ups.

## Author

Daniel Felipe Solano Quirós

B.Sc. Systems Engineering  
Electronics Technician

www.linkedin.com/in/daniel-felipe-solano-quiros

Costa Rica

