## Current Status (Updated)

**RAG v1: Complete.**
**RAG v2: Complete.**
**RAG v3: Complete (V3.1-V3.4), wired end-to-end and validated with real Ollama runs.**
**RAG v4: Complete (V4.1-V4.5), persistent incremental storage layer.**
**RAG v5: Complete (V5.1-V5.5), browser-based interface for technicians.**
**RAG v6: Complete (V6.1-V6.3), extraction quality and document lifecycle control.**
**RAG v7: In Progress (V7.1, V7.2.1, V7.2.2, V7.3.1 done), UI polish, measured retrieval-weight tuning, multi-symbol table row support, and persistent Q&A history.**

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
| V7.1 UI Cleanup | Done (2026-08-31 - `webapp.py`'s `_document_summary` now returns `display_name` (`os.path.basename(source)`) alongside `source`; `templates/index.html` shows one radio button per document row plus a single `#delete-btn` below the table, disabled until a document is selected, with a dynamic `confirm()` reading the selected row's display name - replacing the earlier per-row delete button; confirmed live in the browser) |
| V7.2.1 Weighted Hybrid Retrieval (Production Tuning) | Done (2026-08-31 - `hybrid_retrieve` gained optional `lexical_weight`/`semantic_weight` parameters, defaulting to `LEXICAL_WEIGHT = 1.0`/`SEMANTIC_WEIGHT = 1.0` so the fusion mechanism's own tests keep testing the neutral, unweighted formula unchanged; `tests/test_hybrid_weighting_manual.py` measured 5 candidate weight pairs against the full 18-question golden dataset (16 sample.pdf + 2 NE555N.pdf) with real embeddings; `lexical_weight=2.0, semantic_weight=1.0` strictly beat the unweighted baseline on Precision@K/Recall@K/MRR overall and literal-only, and exactly matched it (no regression) on paraphrased questions; `main.py` gained `PRODUCTION_LEXICAL_WEIGHT = 2.0`/`PRODUCTION_SEMANTIC_WEIGHT = 1.0` and `answer_question` now passes them explicitly to `hybrid_retrieve`, kept as an application-level tuning decision separate from the module's own neutral default) |
| V7.2.2 Multi-Symbol Table Rows | Done (2026-08-31 - `_split_multi_symbol_row` in `document_loader.py` derives the packed symbol count from the Parameter column, corroborated by a matching value column, then groups the Symbol column's own lines - however many there are - into that many even chunks; a first attempt anchored on the Symbol column's own line count broke against the real `tr`/`tf` row (`t\noptr\nt\nf`-shaped, i.e. `t\nr\nt\nf`, 4 lines for 2 symbols) and was redesigned test-first once real evidence (`diagnose_tr_tf_raw_cells.py`) confirmed the shape. Live validation then surfaced a second bug in `_apply_lexical_safety_net` (RAG v5.5): it could evict a correct, tied-best-score result to admit an unrelated, boilerplate-inflated one; fixed by protecting any already-tied, high-confidence result from eviction. Both confirmed live and with 210 passing tests - see the RAG v7.2.2 entry below) |
| V7.2.3 Safety-Net Threshold Recalibration (round 2) | Deferred - V7.2.2 did surface a new real safety-net bug, but it concerned WHICH tied result gets evicted, not the `LEXICAL_SAFETY_NET_THRESHOLD = 0.35` value itself (fixed there instead, see V7.2.2's row above); the threshold remains unrevisited until a case actually calls it into question |
| V7.3.1 Persistent Q&A History | Done (2026-08-31 - new `qa_history.py` module (SQLite-backed, mirroring `chunk_store.py`'s pattern), displayed in a right-hand column on the home page; `/ask` uses the Post/Redirect/Get pattern - storing its result in `session["ask_result"]` and redirecting to `home()`, which pops and displays it once - fixing a real refresh-duplication bug found via live validation, not a design chosen up front; 224 passing tests, zero regressions - see the RAG v7.3.1 entry below) |
| V7.3.2 Conversational Follow-Up Resolution | Planned, optional stretch - new `conversation.py` module that rewrites a follow-up question into a standalone one using history, via an LLM call, BEFORE retrieval; history may only resolve references in the current question, never supply answer content directly, to protect grounding |

224 automated tests passing, zero known regressions.

**Resolved (RAG v7.2.1, 2026-08-31): unweighted Reciprocal Rank Fusion in hybrid retrieval was a measured trade-off, not a bug.** It originally improved ranking on paraphrased questions (MRR 0.46 vs. 0.42 for lexical alone) but cost ranking quality on literal, datasheet-vocabulary questions (MRR 0.79 vs. a perfect 1.0 for lexical alone) - the majority of realistic queries. A dedicated risk check had confirmed this never caused the no-answer gate to reject an answerable question, so it was deliberately left unweighted until real evidence justified a change (re-run after RAG v4's storage refactor, reproducing the exact same MRR numbers - confirming v4 changed nothing about retrieval behavior). `hybrid_retrieve` now accepts optional `lexical_weight`/`semantic_weight` (see phase table above); `tests/test_hybrid_weighting_manual.py` measured candidates against the expanded 18-question golden dataset and found `lexical_weight=2.0, semantic_weight=1.0` strictly better than the unweighted baseline overall and literal-only, with zero regression on paraphrased questions. `main.py` now requests this tuning in production. See README.md's Known Limitations for the full numbers.

**Resolved (RAG v7.2.2, 2026-08-31): a table row packing more than one symbol into one visual row didn't split cleanly, and a related lexical safety-net bug could evict the correct answer once it did.** Confirmed against a real page of NE555N.pdf via `diagnose_multi_symbol_tables.py`/`diagnose_tr_tf_raw_cells.py` (temporary, never committed): the `tr`/`tf` row's Symbol column extracts as `t\nr\nt\nf` (4 lines, since PyMuPDF also splits each packed symbol across its own base+subscript lines) rather than the assumed `tr\ntf` (2 lines) - a first implementation anchored on the Symbol column's own line count broke against this real shape, and would have broken again for any future document with a different line count, per Daniel's explicit review. `_split_multi_symbol_row` (`document_loader.py`) now derives the symbol count from the Parameter column instead, corroborated by at least one value column splitting into that same count, then groups the Symbol column's own lines - however many there are - into that many even chunks, adapting to any lines-per-symbol count rather than a hardcoded one. Live validation then surfaced a second, independent bug: the correct fact was retrieved, but `_apply_lexical_safety_net` (RAG v5.5) evicted it to admit an unrelated page whose lexical score only tied because of a header repeated on nearly every page ("NE555 - SA555 - SE555"), not because it was relevant. The safety net's eviction step now excludes any slot already tied for the best lexical score at sufficient confidence - the same bar a rescue candidate must clear - so it never sacrifices one strong match to admit another equally strong one, and declines to rescue at all when no genuinely weaker slot exists. Both confirmed live: "What is the output rise time of the NE555?" and "What is the output fall time of the NE555?" now each answer correctly and independently, citing page 5. 210 tests passing (13 new since RAG v7.2.1: 11 in `test_document_loader.py` for the multi-symbol split and its redesign, 2 in `test_hybrid_retrieval.py` for the safety-net eviction fix), zero regressions. The NE555N.pdf golden dataset was later extended with two regression cases (`ne555_output_rise_time`/`ne555_output_fall_time`, checking `tr`/`tf` as standalone keyword tokens) that specifically guard against `_split_multi_symbol_row` ever regressing to the original garbled `Symbol: trtf` shape - data added to the existing dataset, no new test functions, count unchanged at 210. Committed as `8a5ad7e`.

**Resolved (RAG v7.3.1, 2026-08-31): a technician had no way to revisit a past question/answer without re-asking it, and fixing that surfaced a refresh-duplication bug live testing caught but the test suite didn't.** New `src/qa_history.py` (SQLite-backed, `data/qa_history.db`, mirroring `chunk_store.py`'s short-lived-connection-per-call pattern): `init_db`, `save_qa_pair` (auto-timestamps unless an explicit `asked_at` is given, for testability), `load_history` (most-recent-first, optional `limit`). Displayed in a right-hand column on the home page (design confirmed with Daniel first: same page, not a separate one) via a two-column flex layout in `templates/index.html`. `webapp.py` loads history in all four routes; `ask()` saves a Q&A pair only on the success path (never on an empty question or an LLM exception). Deliberately scoped OUT: history deletion, pagination, a separate `/history` page. 221 tests passing before live validation.

Live validation (real browser screenshots) then surfaced a bug the green test suite didn't catch: refreshing the page after asking a question resubmitted the form (`/ask` rendered `index.html` directly from the POST), silently duplicating the entry into history and wasting an Ollama call. Daniel questioned whether this was worth fixing rather than accepting it on faith ("porque seria un problema que al refrescar la pagina quede guardado en el resgistro?") - the honest answer: not a correctness/grounding bug, but it crowds the fixed 10-entry history window and wastes compute, a known web anti-pattern with a standard fix. After comparing three options (skip-duplicate patch / full Post-Redirect-Get / document as a known limitation), Daniel confirmed the full PRG pattern. Implemented test-first: `tests/test_webapp.py` updated first (3 tests changed to `follow_redirects=True`, 3 new, the key one asserting `response.status_code == 302`), confirmed red (223 passed, 1 failed - exactly the redirect-status test, as predicted) before writing the fix. `ask()` now stores its outcome in `session["ask_result"]` and redirects to `home()` in all three cases (empty question, LLM error, success) instead of rendering directly; `home()` pops the session value and displays it exactly once, so a later plain `GET /` (e.g. a refresh) never re-shows or re-submits it. Requires `app.secret_key` (static local value - single local user, nothing sensitive stored). `templates/index.html` needed no changes. Confirmed green at 224 tests, then re-validated live: no more "resubmit form?" browser dialog, no duplicate history entries, URL returns to `/` after asking. Committed as a single commit, `d8a7842` (nothing from this feature was committed before the PRG fix was found, so there was no natural point to split it into two commits the way V7.2.1/V7.2.2 were).

A separate, unrelated gap was found during the same live-validation pass, initially deferred and later investigated on its own (2026-09-01): "What is the maximum collector-emitter voltage?" intermittently returned the no-answer fallback even though `sample.pdf` contains the answer verbatim. The initial diagnosis (`diagnose_collector_emitter_voltage.py`) suspected a ranking problem - the correct chunk (lexical_confidence=0.270, never found semantically) ranking #4 behind three NE555N.pdf chunks sharing generic vocabulary with the query, with production `TOP_K_RESULTS=3` never letting the LLM see it. **This turned out to be a false lead.** Measuring candidate fixes (raising `TOP_K_RESULTS`, lowering `LEXICAL_SAFETY_NET_THRESHOLD`) with a new manual script (`tests/test_cross_document_ranking_manual.py`) against a freshly rebuilt corpus found the case already at rank #1 with no change needed. The real cause: `data/chunk_store.db` (73 persisted chunks) had drifted out of sync with the current PDFs (a fresh rebuild produces 83) - most likely a document added directly to `data/documents/` rather than through `/upload`, which `main.initialize_system` never detects once the persisted store already holds any data (it only does a full ingest when the store is completely empty). The investigation itself was also confounded by a second, unrelated bug: `tests/test_hybrid_weighting_manual.py` (since RAG v7.2.1) called `vector_store.get_collection()` with no override, defaulting to the REAL production collection instead of an isolated one - running it silently reset and overwrote the real vector store with a fresh rebuild, which is what made the live bug intermittently "fix itself" mid-investigation. Fixed: both manual scripts now use an isolated `chromadb.EphemeralClient` collection (committed as `13e11d4`), matching the RAG v4 isolation convention every other vector-store test already follows. Resolved with a deliberate, clean re-ingestion (clearing `data/chunk_store.db` and the vector store, then letting the app rebuild both from the current PDFs) - confirmed live and via measurement that neither `TOP_K_RESULTS` nor `LEXICAL_SAFETY_NET_THRESHOLD` needed to change. Open follow-up, not yet pursued: making `initialize_system` detect a document added directly to the folder rather than only ingesting once when the store is empty.

**Known accepted risk (RAG v4, documented, not solved):** chunk metadata (SQLite) and the vector store (ChromaDB) are two separate stores updated in sequence, not inside one transaction. A failure between the two writes could leave a document's metadata and vectors out of sync. Accepted for now given a single local user with no concurrent writers.

**Resolved (RAG v5.5, 2026-08-30):** the two retrieval misses opened above were confirmed with `diagnose_retrieval.py` (temporary, never committed) and fixed with two small, targeted, test-first changes. (1) `hybrid_retrieval.py` gained a lexical safety net (`LEXICAL_SAFETY_NET_THRESHOLD = 0.35`) that force-includes a chunk lexical retrieval is highly confident about even when semantic search never found it and RRF fusion would otherwise drop it. (2) A live browser test then exposed a second gap - two chunks tied exactly on lexical score, and the safety net's original "single best match" design only ever looked at `lexical_results[0]`, missing its tied sibling. Generalized to check every chunk tied for the best score. (3) `main.py`'s no-answer confidence gate was changed to check `max(lexical_confidence, semantic_confidence)` across ALL retrieved chunks, not just the top-ranked one, since the safety net intentionally places a rescued chunk in the weakest slot, not rank #1, and the old gate would silently reject it there. All three fixes validated against the isolated original 32-chunk golden dataset (identical MRR/Precision/Recall to the documented baseline in every case) before being committed. 178 tests passing. See the RAG v7.2.2 entry above for a fourth refinement to this same mechanism, found via live validation of the multi-symbol table fix.

**Resolved (RAG v6.2, 2026-08-30): PDF table extraction losing column structure.** The limitation surfaced by v5.5 above is fixed. `diagnose_tables.py` (temporary, never committed) confirmed against a real page of NE555N.pdf that PyMuPDF's `page.find_tables()` does recover the row/column structure that `page.get_text()` throws away - the `toff` row extracted cleanly as `['t\noff', 'Turn off time (5) (V = V )\nreset CC', '', '0.5', '', '', '0.5', '', 'µs']`, with headers split across two merged rows and subscripted characters split onto their own cell line. `document_loader.py` now reconstructs each detected real table (2+ rows, filtering out degenerate 1-row artifacts like a page footer PyMuPDF also detects) into explicit `Symbol: X | Parameter: Y | ...: value | Unit: Z` facts, appended after the page's original plain text - purely additive, so any page without a table is provably unaffected. Validated with 191 tests (10 new, zero regressions) and live: the exact question that returned the no-answer fallback throughout v5.5 ("What is the turn off time of the NE555?") now answers correctly and grounded ("0.5 µs, Source: NE555N, page 5"), with no regression on the question that already worked. **Resolved (RAG v7.2.2):** a table row where the source PDF packs two symbols into one visual row (e.g. `tr`/`tf`) now splits cleanly into one fact per symbol - see the RAG v7.2.2 entry above for the full fix and the safety-net bug it also surfaced.

**RAG v6: complete (V6.1-V6.3).** All three planned sub-phases are done and validated: document lifecycle control (delete), table-aware extraction, and permanent regression coverage for the exact bug class v5.5/v6.2 found. V6.4 (safety-net threshold recalibration) was deliberately not pursued - see its own phase-table row above for why. See `claude/rag-v6-plan.md` for the full plan and progress log. Remaining, non-blocking follow-up: a fully standalone executable (PyInstaller) so end users don't need Python installed at all - table rows packing more than one symbol per row were resolved in RAG v7.2.2 (see above).

**RAG v7: in progress (V7.1, V7.2.1, V7.2.2, V7.3.1 done).** V7.1 (filename display + single delete button), V7.2.1 (measured hybrid retrieval weight tuning), V7.2.2 (multi-symbol table row splitting plus a lexical safety-net eviction fix), and V7.3.1 (persistent Q&A history plus a Post/Redirect/Get fix for a refresh-duplication bug found via live validation) are all done and validated - see their phase-table rows above. Still open: V7.2.3 (safety-net recalibration, deferred pending new evidence), V7.3.2 (conversational follow-up resolution, optional stretch), and a not-yet-phased cross-document retrieval ranking gap found during V7.3.1's live validation (see the RAG v7.3.1 entry above).

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

### RAG v7 Rules

Only start after V6 is stable (it now is - see Current Status above).

Focus: three areas confirmed with Daniel before starting - UI/UX
cleanup, deeper retrieval quality (grounded in real measurement,
not guesses), and giving a technician a way to revisit past
questions/answers without re-asking, plus an optional step
towards resolving conversational follow-up questions.

Sub-phases, in order:

- V7.1 UI Cleanup - done (see phase table above): document
  filename display instead of full path, single delete button
  instead of one per row
- V7.2.1 Weighted Hybrid Retrieval (Production Tuning) - done
  (see phase table above): `hybrid_retrieve` gained optional,
  backward-compatible RRF weights; the production weighting
  itself was decided from real measurement
  (`tests/test_hybrid_weighting_manual.py`) against the full
  golden dataset, never guessed
- V7.2.2 Multi-Symbol Table Rows - done (see phase table
  above): diagnosed first (same evidence-first approach as
  `diagnose_tables.py`), redesigned once real evidence
  (`diagnose_tr_tf_raw_cells.py`) contradicted the first
  attempt's assumption, and also fixed a related
  `_apply_lexical_safety_net` eviction bug found via live
  validation
- V7.2.3 Safety-Net Threshold Recalibration (round 2) - deferred
  (see phase table above): only revisit `LEXICAL_SAFETY_NET_THRESHOLD`
  if new real evidence appears from V7.2.1/V7.2.2's work
- V7.3.1 Persistent Q&A History - done (see phase table above):
  new `qa_history.py` module (SQLite-backed, mirroring
  `chunk_store.py`'s single-responsibility pattern), shown in a
  right-hand column on the home page; also fixed a
  refresh-duplication bug found via live validation (not
  designed for up front) using the Post/Redirect/Get pattern on
  `/ask`
- V7.3.2 Conversational Follow-Up Resolution - planned, optional
  stretch: new `conversation.py` module, LLM-based query
  rewriting BEFORE retrieval, with a hard grounding rule -
  history may only resolve references in the current question
  (e.g. "what about its storage temperature?" after asking about
  a specific part), never supply answer content directly without
  going through retrieval again

Explicitly OUT of scope for RAG v7:

- Any change to the RAG v1-v6 architecture beyond hybrid
  retrieval's own optional weight parameters
- A general-purpose conversational agent - V7.3.2, if pursued,
  only resolves references for retrieval, it does not let the
  model answer from memory
- Full standalone packaging (PyInstaller) - unrelated to this
  phase's focus, tracked separately (see RAG v6's closure note
  above)

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

### document_loader.py (RAG v1, RAG v6.2, RAG v7.2.2)

Responsible only for:

- Document ingestion
- PDF extraction, including table-aware reconstruction of
  detected tables into structured text (`_reconstruct_table`,
  `_build_column_labels`, `_clean_cell`, RAG v6.2) - always
  additive to the plain-text extraction, never a replacement
  for it
- Splitting a table row that packs more than one symbol into a
  single visual row into one row per symbol
  (`_split_multi_symbol_row`, RAG v7.2.2), before table
  reconstruction runs - the symbol count is derived from the
  Parameter column (corroborated by a matching value column),
  not from the Symbol column's own line count
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

### hybrid_retrieval.py (RAG v3.4, RAG v7.2.1, RAG v7.2.2)

Responsible only for:

- Combining lexical retrieval (retriever.py) and semantic search (semantic_search.py) results into a single ranked list, using Reciprocal Rank Fusion (RRF)
- Accepting optional `lexical_weight`/`semantic_weight` parameters (RAG v7.2.1) to scale each method's RRF contribution before summing - both default to the module's own neutral `LEXICAL_WEIGHT = 1.0`/`SEMANTIC_WEIGHT = 1.0`, reproducing the original unweighted formula exactly. The PRODUCTION-tuned values (2.0/1.0, measured via `tests/test_hybrid_weighting_manual.py`) are an application-level decision that belongs to `main.py` (`PRODUCTION_LEXICAL_WEIGHT`/`PRODUCTION_SEMANTIC_WEIGHT`), not to this module's own default
- Protecting any result already tied for the best lexical score, at sufficient confidence, from being evicted by its own safety net (`_apply_lexical_safety_net`, RAG v7.2.2) - a rescue may only sacrifice a genuinely weaker slot, never another equally-strong one

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

### qa_history.py (RAG v7.3.1)

Responsible only for:

- Storing a question/answer pair with its timestamp
  (`save_qa_pair`) in SQLite (`data/qa_history.db`), mirroring
  `chunk_store.py`'s short-lived-connection-per-call pattern
- Loading past Q&A pairs back out, most-recent-first, with an
  optional `limit` (`load_history`)
- Creating the underlying table if it doesn't exist yet
  (`init_db`)

Must not contain:

- Flask routes or session handling (see `webapp.py`)
- Any decision about WHEN to save or WHAT to display - `webapp.py`
  decides that (e.g. never saving on an empty question or LLM
  error, capping the display to `HISTORY_DISPLAY_LIMIT`)
- Retrieval, ranking, or LLM logic

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

### webapp.py (RAG v5, RAG v6.1, RAG v7.1, RAG v7.3.1)

Responsible only for:

- Flask routes: home page (`/`), ask-a-question (`/ask`),
  upload/replace-a-document (`/upload`),
  delete-a-document (`/delete`)
- Lazy-singleton in-memory state (`_get_state`), initialized
  once from `main.initialize_system` and refreshed after an
  upload, guarded by a `threading.Lock()`
- Summarizing loaded documents for display, including a
  `display_name` (`os.path.basename(source)`, RAG v7.1) shown to
  the technician instead of the full document path - `source`
  itself is unchanged and still used internally (e.g. as the
  `/delete` form's identifying value)
- Basic input validation (empty question, missing file, non-PDF
  file) and turning pipeline exceptions into a friendly message
- Loading and displaying Q&A history (RAG v7.3.1, via
  `qa_history.load_history`), and saving a new Q&A pair on a
  successful `/ask` (via `qa_history.save_qa_pair`) - never on an
  empty question or an LLM error
- Post/Redirect/Get on `/ask` (RAG v7.3.1): stores the outcome in
  `session["ask_result"]` and redirects to `home()` instead of
  rendering the template directly, so refreshing the page after
  asking a question is a harmless `GET /` instead of a form
  resubmission; `home()` pops the session value so it displays
  exactly once

Must not contain:

- Retrieval, ranking, or confidence-scoring logic (see
  `hybrid_retrieval.py`, `retriever.py`)
- Ingestion or chunking logic (see `ingestion.py`,
  `chunker.py`) - this module only calls them in the right
  order, the same relationship `ingestion.py` has with
  `chunk_store.py`/`vector_store.py`
- Prompt construction or LLM calls (see `generator.py`,
  `llm.py`)
- Q&A history persistence logic itself (see `qa_history.py`) -
  this module only calls `save_qa_pair`/`load_history` at the
  right points

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
