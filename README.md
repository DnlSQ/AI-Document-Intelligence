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
- pytest (224 automated tests)
- Git
- PowerShell
- Local LLM inference

## Current Status

**RAG v1: Complete.**
**RAG v2: Complete (V2.1 - V2.3).**
**RAG v3: Complete (V3.1 - V3.4), wired end-to-end and validated with real Ollama runs.**
**RAG v4: Complete (V4.1 - V4.5), persistent incremental storage layer.**
**RAG v5: Complete (V5.1 - V5.5), browser-based interface for technicians.**
**RAG v6: Complete (V6.1 - V6.3), extraction quality and document lifecycle control.**
**RAG v7: In Progress (V7.1, V7.2.1, V7.2.2, V7.3.1 done), UI polish, measured retrieval-weight tuning, multi-symbol table row support, and persistent Q&A history.**

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
comparatively, not just assumed: **224 automated tests**
currently pass, including a golden-dataset evaluation across
16 questions (12 matching the document's own technical
vocabulary, 4 deliberately paraphrased in natural language)
and IR-style retrieval metrics (Precision@K, Recall@K, Mean
Reciprocal Rank) computed independently for lexical, semantic,
and hybrid retrieval. This comparison originally surfaced a
real, measured trade-off - investigated with a dedicated risk
check, and later addressed with real evidence rather than a
guess (RAG v7.2.1) - see Known Limitations below.

As of RAG v6.2, PDF extraction is table-aware: pages are still
extracted as plain text as before, but any table PyMuPDF can
detect (`page.find_tables()`) is additionally reconstructed
into explicit `Symbol: X | Parameter: Y | ...: value | Unit: Z`
facts appended to that page's text, instead of relying only on
the default flattened, column-less dump. This was validated
against a real, previously-unseen datasheet: a question that
consistently returned the no-answer fallback under the
flattened extraction ("What is the turn off time of the
NE555?") now answers correctly ("0.5 µs, Source: NE555N, page
5") with no regression on the question that already worked.

As of RAG v7.1, the browser interface shows each loaded document
by its filename only (not the full local path it's stored at),
and document deletion is a single button below the documents
table - enabled once a document is selected via radio button,
and confirmed with a dynamic prompt naming the selected file -
replacing the earlier per-row delete button design.

As of RAG v7.2.1, hybrid retrieval's Reciprocal Rank Fusion
accepts configurable weights for its lexical and semantic
contributions (defaulting to the original unweighted 1.0/1.0
formula). Measured against the full 18-question golden dataset
(16 sample.pdf + 2 NE555N.pdf questions), a production
weighting of lexical_weight=2.0/semantic_weight=1.0 strictly
improves Precision@K, Recall@K, and MRR over the unweighted
formula for both the overall dataset and literal
(datasheet-vocabulary) questions, while exactly matching the
unweighted baseline - no regression - on paraphrased questions.
This addresses the trade-off previously documented in Known
Limitations below with real measurement instead of a guess.

As of RAG v7.2.2, a table row where the source PDF packs more
than one symbol into a single visual row (e.g. `tr`/`tf`
sharing one row on the NE555N.pdf electrical-characteristics
table) is reconstructed into one independent fact per symbol,
closing the one known gap left open by RAG v6.2. The number of
symbols packed into the row is derived from the Parameter
column's own line count (corroborated by at least one value
column splitting into that same count), not from the Symbol
column's line count - the Symbol column can itself split each
packed symbol across its own base+subscript lines (e.g.
`tr`/`tf` extracting as `t\nr\nt\nf`, four lines for two
symbols, confirmed against the real PDF), so anchoring on it
directly would only work for whatever specific line count was
tested. This makes the fix adapt to any number of
lines-per-symbol rather than a hardcoded one. Live validation
then surfaced a second, unrelated gap: the newly-correct fact
was being retrieved, but a lexical safety net inside
`hybrid_retrieval.py` (originally built for RAG v5.5) was
evicting it from the final results to make room for an
unrelated page that only tied its lexical score because of a
boilerplate header repeated on nearly every page of the
document. The safety net now protects any result already tied
for the best lexical score, at sufficient confidence, from
being evicted to admit another member of that same tied group
- it now only ever sacrifices a genuinely weaker slot, or
declines to rescue at all if none exists. Both fixes were
validated live: "What is the output rise time of the NE555?"
and "What is the output fall time of the NE555?" now each
answer correctly and independently, citing page 5.

As of RAG v7.3.1, the browser interface shows a persistent
question/answer history in a right-hand column next to the main
page (SQLite-backed, `data/qa_history.db`), so a technician can
revisit a past answer without re-asking a question. `/ask` uses
the Post/Redirect/Get pattern: instead of rendering the answer
directly from the POST request, it saves the result and redirects
to the home page, which displays it once. This was a deliberate
fix for a real bug found during live validation - refreshing the
browser after asking a question used to resubmit the form,
silently duplicating the entry in history and triggering an extra,
wasted LLM call - not a design chosen up front. The same live
validation pass also surfaced an unrelated, separate issue (a
cross-document retrieval ambiguity, initially suspected as a
ranking bug) - see Known Limitations below for the investigation
and its real root cause.

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
- [x] Delete a loaded document from the browser (RAG v6.1 -
      removes it from disk, the chunk repository, and the
      vector store together, with a confirmation prompt)
- [x] Table-aware PDF extraction (RAG v6.2 - detected tables
      are reconstructed into explicit Symbol/Parameter/value/
      Unit facts, appended to the page's plain text; validated
      live against a real datasheet question that previously
      had no answer)
- [x] NE555N.pdf-specific golden dataset (RAG v6.3 - the two
      questions from the v5.5/v6.2 investigation are now a
      permanent automated regression check, not a manual
      browser smoke test)
- [x] Document filename display + single-button delete UI
      (RAG v7.1 - shows `display_name` instead of the full
      document path, one delete button below the table instead
      of one per row)
- [x] Configurable, measured hybrid retrieval weighting
      (RAG v7.2.1 - `hybrid_retrieve` accepts optional
      `lexical_weight`/`semantic_weight`; production tuned to
      2.0/1.0 based on real measurement against the full golden
      dataset, see Known Limitations below)
- [x] Multi-symbol table row splitting + lexical safety-net
      eviction protection (RAG v7.2.2 - a table row packing
      more than one symbol into a single visual row is split
      into one fact per symbol, with the symbol count derived
      from the Parameter column rather than a hardcoded line
      count; also fixes a safety-net bug, found via live
      validation, that could evict a correct, tied-best-score
      result to admit an unrelated, boilerplate-inflated one)
- [x] Persistent Q&A history (RAG v7.3.1 - `src/qa_history.py`,
      SQLite-backed; shown in a right-hand column on the home
      page; `/ask` uses the Post/Redirect/Get pattern so that
      refreshing the page after asking a question never resubmits
      the form, preventing duplicate history entries and wasted
      LLM calls - a fix for a real bug found via live validation,
      not a design chosen up front)

## Known Limitations

- **Resolved (RAG v7.2.1): unweighted hybrid retrieval fusion
  was a measured trade-off, not a bug.** Reciprocal Rank Fusion
  originally weighted lexical and semantic retrieval equally.
  Measured on the original 16-question golden dataset, this
  improved ranking on paraphrased, natural-language questions
  (MRR 0.46 vs. 0.42 for lexical alone) but cost ranking
  quality on literal, datasheet-vocabulary questions (MRR 0.79
  vs. a perfect 1.0 for lexical alone) - which make up the
  majority of realistic troubleshooting queries. A dedicated
  risk check confirmed this never caused the system to reject
  an answerable question, so it was deliberately left unweighted
  until real evidence justified a change. `hybrid_retrieve` now
  accepts optional `lexical_weight`/`semantic_weight` parameters
  (defaulting to the original unweighted 1.0/1.0, so the fusion
  mechanism's own tests stay unaffected), and
  `tests/test_hybrid_weighting_manual.py` measured several
  candidates against the expanded 18-question golden dataset
  (16 sample.pdf + 2 NE555N.pdf). `lexical_weight=2.0,
  semantic_weight=1.0` strictly beat the unweighted baseline on
  Precision@K, Recall@K, and MRR overall and on literal-only
  questions (e.g. MRR 0.70 -> 0.76 overall, 0.76 -> 0.83
  literal-only), while exactly matching the baseline on
  paraphrased questions - no regression. `main.py`'s
  `answer_question` now requests this tuned weighting in
  production, kept as an application-level decision separate
  from the fusion module's own neutral default.
- **Resolved (RAG v7.2.2): a table row packing more than one
  symbol into one visual row didn't split cleanly, and a
  related lexical safety-net bug could evict the correct
  answer once it did.** Confirmed against a real page of
  NE555N.pdf via `diagnose_multi_symbol_tables.py`/
  `diagnose_tr_tf_raw_cells.py`: the `tr`/`tf` row's Symbol
  column extracts as `t\nr\nt\nf` (4 lines, because PyMuPDF
  also splits each packed symbol across its own
  base+subscript lines) rather than the assumed `tr\ntf` (2
  lines) - a first attempt anchored on the Symbol column's own
  line count broke against this real shape and would have
  broken again for any future document with a different line
  count. `_split_multi_symbol_row` (`document_loader.py`) now
  derives the symbol count from the Parameter column instead
  (corroborated by at least one value column splitting into
  that same count, so a Parameter cell that merely text-wraps
  across several lines under one symbol is never
  misidentified), then groups the Symbol column's lines -
  however many there are - into that many even chunks. Live
  validation then surfaced a second, independent bug: the
  correct fact was being retrieved, but
  `_apply_lexical_safety_net` (RAG v5.5) was evicting it to
  admit an unrelated page whose lexical score only tied
  because of a header repeated on nearly every page of the
  document ("NE555 - SA555 - SE555"), not because it was
  actually relevant. The safety net's eviction step now
  excludes any slot that is itself tied for the best lexical
  score at sufficient confidence - the same bar a rescue
  candidate must clear - so it never sacrifices one strong
  match to admit another equally-strong one, and declines to
  rescue at all when no genuinely weaker slot exists. Both
  confirmed live: "What is the output rise time of the
  NE555?" and "What is the output fall time of the NE555?"
  now each answer correctly and independently, citing page 5
  (previously: a no-answer fallback with a garbled source
  citation). 210 tests passing, zero regressions. The NE555N.pdf
  golden dataset was later extended with two regression cases
  (`ne555_output_rise_time`/`ne555_output_fall_time`, keyword-
  checking for `tr`/`tf` as standalone tokens) that guard
  specifically against `_split_multi_symbol_row` ever regressing
  to the original garbled `Symbol: trtf` behavior.
- **Resolved (2026-09-01): a cross-document retrieval ambiguity
  turned out to be a desynced local store, not a ranking bug.**
  "What is the maximum collector-emitter voltage?" intermittently
  returned the no-answer fallback (citing NE555N.pdf) even though
  `sample.pdf` (a transistor datasheet) contains the answer
  verbatim. The initial diagnosis (`diagnose_collector_emitter_voltage.py`)
  suspected a ranking problem: the correct chunk scored
  lexical_confidence=0.270 but zero semantic_confidence, ranking
  #4 behind three NE555N.pdf chunks sharing generic vocabulary
  ("voltage", "maximum") with the query - with production
  `TOP_K_RESULTS=3`, it never reached the LLM. Measuring two
  candidate fixes (raising `TOP_K_RESULTS`, lowering
  `LEXICAL_SAFETY_NET_THRESHOLD`) against a freshly rebuilt copy
  of the full document corpus found the case already ranking #1
  with no change needed at all - directly contradicting the
  original diagnosis. The real cause: `data/chunk_store.db` (73
  persisted chunks) had drifted out of sync with the current PDF
  files (a fresh rebuild produces 83) - most likely because a
  document was added directly to `data/documents/` rather than
  through the browser's upload route, and `main.initialize_system`
  only performs a full ingest when the persisted store is
  completely empty, so it silently never picked up the new file.
  This is the same class of risk already documented below (SQLite
  and the vector store as two separate stores that can drift
  apart) - this was the first time it was actually observed, not
  a new risk. A second, unrelated bug compounded the investigation
  itself: `tests/test_hybrid_weighting_manual.py` (since RAG
  v7.2.1) and a new investigation script both called
  `vector_store.get_collection()` with no override, which
  defaults to the REAL on-disk production collection instead of
  an isolated one - running either script silently reset and
  overwrote the real vector store with a fresh rebuild as a side
  effect, which is what made the live bug intermittently seem to
  "fix itself" mid-investigation. Both scripts now use an isolated
  `chromadb.EphemeralClient` collection, matching the isolation
  convention already established for every other test that
  touches the vector store since RAG v4. Resolved with a clean,
  deliberate re-ingestion (clearing `data/chunk_store.db` and the
  vector store, then letting the app rebuild both from the current
  PDFs) - confirmed live, and confirmed via measurement that
  neither `TOP_K_RESULTS` nor `LEXICAL_SAFETY_NET_THRESHOLD` needed
  to change; the original ranking hypothesis was a false lead.
  Open follow-up, not yet pursued: `initialize_system` has no way
  to detect a document added directly to the documents folder once
  the store already holds any data - a candidate for a future
  robustness improvement.
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
- **Resolved (RAG v5.5): two retrieval misses found via a
  real, previously-unseen document (NE555 timer datasheet).**
  Root cause, confirmed with a diagnostic script rather than
  guessed: the correct chunk was the single best lexical match
  in both cases, but Reciprocal Rank Fusion still excluded it,
  for two distinct reasons. (1) It was invisible to semantic
  search entirely, so RRF's rank-based fusion scored it below
  chunks found by both methods. Fixed with a "lexical safety
  net": a chunk lexical retrieval is highly confident about
  (confidence >= 0.35) is force-included even when semantic
  search missed it. (2) A live browser test then exposed a
  second gap: two chunks tied exactly on lexical score - one a
  feature-list bullet only naming a spec, the other the data
  table entry actually holding its value - and the retriever's
  tie-break (lowest chunk ID wins) happened to favor the
  bullet. The safety net was generalized to check every chunk
  tied for the best score, not just the first. Both fixes
  validated against the original 16-question golden dataset in
  isolation: identical Precision/Recall/MRR to the previously
  documented baseline in every case, confirming zero
  regression. See the RAG v7.2.2 entry above for a third
  refinement to this same mechanism.
- **Resolved (RAG v6.2): PDF table extraction losing column
  structure.** The gap surfaced by the v5.5 investigation above
  - a multi-column datasheet table extracted as a flat sequence
  with no column alignment, e.g. `tr / tf / Output rise time /
  Output fall time / 100 / 100 / 200 / 200 / 100 / 100 / 300 /
  300 / ns / toff / Turn off time (5) (Vreset = VCC) / 0.5 /
  0.5 / µs` - is fixed. A diagnostic script
  (`diagnose_tables.py`) confirmed PyMuPDF's `page.find_tables()`
  does recover the real row/column structure that
  `page.get_text()` throws away. `document_loader.py` now
  reconstructs each detected table into explicit
  `Symbol: X | Parameter: Y | ...: value | Unit: Z` lines,
  appended after the page's original plain text - purely
  additive, so a page with no table (or only a degenerate one,
  like a footer PyMuPDF also detects as a 1-row "table")
  behaves exactly as before. Validated with 191 tests (zero
  regressions on `sample.pdf`/`plantas.pdf`) and, more
  importantly, live: the exact question that returned the
  no-answer fallback throughout v5.5 ("What is the turn off
  time of the NE555?") now answers correctly and grounded
  ("0.5 µs, Source: NE555N, page 5"). **Resolved (RAG v7.2.2):**
  a table row where the source PDF itself packs two symbols
  into one visual row (e.g. `tr`/`tf` sharing a row) now splits
  cleanly into one fact per symbol - see the RAG v7.2.2 entry
  above for the full fix and the safety-net bug it also
  surfaced. As of RAG v6.3, both original v5.5 questions are
  also a permanent, automated regression check
  (`tests/test_evaluation_ne555n.py`), not just a live
  smoke-test memory.
- **Safety-net threshold recalibration (RAG v6.4) deliberately
  not done - no new evidence to justify it.** `LEXICAL_SAFETY_NET_THRESHOLD
  = 0.35` (in `hybrid_retrieval.py`) is still calibrated from
  only the original two v5.5 cases: RAG v6.2's fix resolved the
  `toff` question at the extraction layer, before the safety
  net is ever consulted, so it produced no new rescue case to
  learn from. Revisiting the threshold with only the same two
  data points would be guessing, not calibrating - left as-is
  until a real new case actually appears. RAG v7.2.2 did surface
  a new real safety-net bug, but it concerned WHICH tied result
  gets evicted, not the 0.35 confidence bar itself - fixed
  there instead, not here. The threshold value remains
  untouched, still awaiting a case that actually calls it into
  question.

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

RAG v5 and RAG v6 are both complete: a technician can ask
questions, upload, replace, or delete documents, and start the
whole system with a single double-click, with no command line
involved beyond a one-time Python/Ollama setup shared by any
local-LLM tool. Every retrieval and extraction gap surfaced by
real, previously-unseen documents so far (RAG v5.5, RAG v6.2,
RAG v7.2.2) has been root-caused with evidence and fixed, not
just patched around, and RAG v6.3 turned both of v5.5's
original failing questions into permanent automated regression
coverage instead of a manual smoke test someone has to
remember to repeat.

The long-term goal remains transforming the project into a
technical troubleshooting assistant capable of helping
technicians:

1. Identify relevant documentation
2. Search technical specifications
3. Investigate recurring failures
4. Retrieve troubleshooting procedures
5. Reduce diagnosis time
6. Improve equipment recovery time

RAG v7 is in progress: V7.1 (filename display + single delete
button), V7.2.1 (measured hybrid retrieval weight tuning),
V7.2.2 (multi-symbol table row splitting plus a lexical
safety-net eviction fix - see Known Limitations above), and
V7.3.1 (persistent question/answer history with a
Post/Redirect/Get fix for a refresh-duplication bug found via
live validation) are all done. Still open, as an optional
stretch: resolving conversational follow-up questions before
retrieval (V7.3.2).

Open, non-blocking follow-ups: safety-net threshold
recalibration if a new real rescue case ever appears (RAG v6.4,
deliberately not pursued yet - see Known Limitations), making
`initialize_system` detect a document added directly to the
documents folder instead of only ingesting once when the store is
completely empty (found while investigating and resolving the
collector-emitter-voltage gap - see Known Limitations above), and
a fully standalone executable (no separate Python installation
required, most likely via PyInstaller).

## Author

Daniel Felipe Solano Quirós

B.Sc. Systems Engineering  
Electronics Technician

www.linkedin.com/in/daniel-felipe-solano-quiros

Costa Rica
