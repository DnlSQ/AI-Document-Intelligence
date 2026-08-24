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

Current architecture (RAG v2, complete):
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
Chunk Repository
│
▼
Retriever
├── Lexical matching (term + technical-term weighting)
├── Exact phrase matching
├── Deterministic ranking (query-aware tie-break, chunk_id fallback)
└── Confidence scoring (per query/chunk pair)
│
▼
Top Relevant Chunks
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
an **Evaluation Framework**: a golden question dataset plus
Precision@K / Recall@K / Mean Reciprocal Rank metrics, run
automatically as part of the test suite.

## Technologies

- Python
- Ollama
- Qwen 2.5 7B
- PyMuPDF
- pytest (92 automated tests)
- Git
- PowerShell
- Local LLM inference

## Current Status

**RAG v1: Complete.**
**RAG v2: Complete (V2.1 - V2.3).**

The system ingests one or more PDF documents, cleans and
chunks their text, retrieves the most relevant passages for
a question using a weighted lexical scoring algorithm,
estimates its own confidence in each match, and generates a
grounded answer using a local LLM - refusing to answer when
no document evidence supports a response.

The system also supports searching across **multiple,
unrelated documents at once**: each document's chunks are
scored independently, so an irrelevant document's content is
naturally filtered out without manual document selection.
This was validated with a transistor datasheet and an
unrelated plant biology paper searched together.

Retrieval quality is measured automatically, not just
assumed: 92 automated tests currently pass with zero known
regressions, including a golden-dataset evaluation (100%
accuracy on real document questions) and IR-style retrieval
metrics (Precision@K, Recall@K, Mean Reciprocal Rank).

## Current Progress

- [x] Local LLM setup
- [x] Project configuration
- [x] LLM abstraction
- [x] PDF ingestion (single and multi-document)
- [x] PDF text extraction
- [x] Text cleaning
- [x] Text chunking
- [x] Lexical retrieval (term matching, technical-term
      weighting, exact phrase matching, deterministic ranking)
- [x] Confidence scoring
- [x] No-answer detection (grounding fallback)
- [x] RAG pipeline (retrieval -> generation, fully wired)
- [x] Source attribution
- [x] Evaluation framework (golden dataset + accuracy report)
- [x] Retrieval metrics (Precision@K, Recall@K, MRR)
- [x] Multi-document support
- [ ] Semantic embeddings (RAG v3)
- [ ] Vector database (RAG v3)
- [ ] Stopword-aware scoring (known limitation, deferred - see
      Known Limitations below)
- [ ] Technician-oriented interface (currently CLI-only)

## Known Limitations

- **Stopword noise in lexical scoring:** common words (e.g.
  "the", "is") can coincidentally inflate a chunk's relevance
  score. This hasn't caused a real accuracy problem so far
  (the golden dataset scores 100% accuracy and perfect MRR),
  but it's a known weak point of pure lexical/keyword
  retrieval. Deferred until either it causes an observed
  problem, or as prep work before RAG v3 (a cleaner lexical
  baseline makes for a fairer comparison against semantic
  search).
- **Confidence is a lexical signal**, not a true probability:
  it can't perfectly distinguish a real (if generic) keyword
  match from a coincidental one. It catches clear-cut weak
  matches; RAG v3's semantic retrieval is the planned deeper
  fix.

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

The long-term goal is to transform the project into a
technical troubleshooting assistant capable of helping
technicians:

1. Identify relevant documentation
2. Search technical specifications
3. Investigate recurring failures
4. Retrieve troubleshooting procedures
5. Reduce diagnosis time
6. Improve equipment recovery time

The next concrete milestone is **RAG v3**: replacing/augmenting
lexical retrieval with semantic embeddings and a local vector
database (FAISS or ChromaDB), enabling the system to match
questions and documents by meaning rather than exact keyword
overlap.

## Author

Daniel Felipe Solano Quirós

B.Sc. Systems Engineering  
Electronics Technician

www.linkedin.com/in/daniel-felipe-solano-quiros

Costa Rica
