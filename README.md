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

## Architecture

Current architecture (RAG v4, complete):
PDF(s)
│
▼
Document Loader
│
▼
Text Cleaner
│
▼
Chunker (supports multiple documents, globally unique chunk IDs)
│
▼
Chunk Repository (persisted in SQLite - survives restarts)
│
├─────────────────────────────┐
▼                              ▼
Lexical Retriever         Embeddings (sentence-transformers)
├── Term + technical-term       │
│   weighting                   ▼
├── Exact phrase matching  Vector Store (ChromaDB, local, offline,
├── Stopword-aware scoring  persisted, updated incrementally)
└── Confidence scoring           │
│                          Semantic Search (cosine similarity)
│                                │
└──────────────┬─────────────────┘
               ▼
   Hybrid Retrieval (Reciprocal Rank Fusion)
               │
               ▼
   No-Answer Gate (skips the LLM call when confidence is too low)
               │
               ▼
          Generator
               │
               ▼
      Qwen 2.5 7B (Ollama)
               │
               ▼
Grounded Answer + Source Attribution

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
- Ollama
- Qwen 2.5 7B
- PyMuPDF
- sentence-transformers (local embedding model)
- ChromaDB (local, persistent vector store)
- SQLite (persistent chunk metadata)
- pytest (164 automated tests)
- Git
- PowerShell
- Local LLM inference

## Current Status

**RAG v1: Complete.**
**RAG v2: Complete (V2.1 - V2.3).**
**RAG v3: Complete (V3.1 - V3.4), wired end-to-end and validated with real Ollama runs.**
**RAG v4: Complete (V4.1 - V4.5), persistent incremental storage layer.**

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
This was validated with a transistor datasheet and an
unrelated plant biology paper searched together.

As of RAG v4, the chunk repository and vector store persist
across restarts instead of being rebuilt from the source PDFs
every time the system starts: chunk metadata is stored in
SQLite (`data/chunk_store.db`) and the ChromaDB vector store
is reused as-is whenever data already exists - confirmed with
a real run: a warm start loads 32 previously-ingested chunks
instantly, without even loading the embedding model. Adding
or replacing a document only re-processes and re-embeds that
one document, not the whole library, via a replace-by-filename
mechanism: uploading a file under an existing name deletes its
outdated chunks and vectors first, so an outdated document can
never coexist with, or be mistaken for, its replacement. The
documents folder is also scanned automatically
(`config.discover_document_paths`) instead of relying on a
hardcoded file list. This is the storage foundation the
upcoming technician-facing interface (RAG v5) will be built
on.

Retrieval quality is measured automatically and
comparatively, not just assumed: **164 automated tests**
currently pass with zero known regressions, including a
golden-dataset evaluation across 16 questions (12 matching
the document's own technical vocabulary, 4 deliberately
paraphrased in natural language) and IR-style retrieval
metrics (Precision@K, Recall@K, Mean Reciprocal Rank)
computed independently for lexical, semantic, and hybrid
retrieval. This comparison surfaced a real, measured
trade-off - see Known Limitations below - which was
investigated with a dedicated risk check before deciding how
to respond to it. The full comparison and the risk check were
re-run after RAG v4's storage refactor and reproduced the
exact same numbers, confirming the persistence work changed
nothing about retrieval behavior.

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
- [ ] Technician-oriented interface (currently CLI-only)

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

## Design Philosophy

The project is intentionally designed to run locally,
without requiring paid AI APIs.

This allows technical documentation to remain within
the local environment and demonstrates how LLM-based
applications can be developed with local models.

Every feature is developed test-first: tests are written
before implementation, and the full suite is run after every
change to guarantee no regressions.

## Future Vision

RAG v4 is now complete: chunk metadata and vector embeddings
persist across restarts instead of being rebuilt from the
source PDFs every time, updates are incremental per document
rather than full-corpus rebuilds, and documents are
discovered automatically from the documents folder instead of
a hardcoded list.

The long-term goal remains transforming the project into a
technical troubleshooting assistant capable of helping
technicians:

1. Identify relevant documentation
2. Search technical specifications
3. Investigate recurring failures
4. Retrieve troubleshooting procedures
5. Reduce diagnosis time
6. Improve equipment recovery time

The next concrete milestone is **RAG v5**: a technician-facing
web interface for uploading PDFs into the persistent library
built in v4 (replacing an outdated document by uploading a new
one under the same name) and asking questions through a
browser instead of the CLI. Growing the evaluation corpus with
additional real documents, and revisiting hybrid retrieval's
fusion weighting if a bigger or more paraphrase-heavy corpus
changes the current measured trade-off, remain open,
non-blocking follow-ups.

## Author

Daniel Felipe Solano Quirós

B.Sc. Systems Engineering  
Electronics Technician

www.linkedin.com/in/daniel-felipe-solano-quiros

Costa Rica
