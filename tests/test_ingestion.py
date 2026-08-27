"""
Tests for src.ingestion.add_or_replace_document: document-level
ingestion into the persistent chunk repository (chunk_store.py).

The PDF extraction / cleaning / chunking steps are mocked at the
point of use (src.ingestion.*) so these tests exercise only the
orchestration logic - not real PDF parsing - and stay fast and
deterministic. Each test uses an isolated, temporary SQLite file
via pytest's tmp_path fixture.
"""
from src.ingestion import add_or_replace_document
from src.chunk_store import load_all_chunks


def make_db_path(tmp_path):
    return str(tmp_path / "test_chunks.db")


def fake_chunks(source, count):
    return [
        {"chunk_id": i + 1, "page": 1, "text": f"{source} chunk {i + 1}", "source": source}
        for i in range(count)
    ]


def patch_pipeline(monkeypatch, chunks):
    """
    Stub out extraction/cleaning/chunking so add_or_replace_document
    only exercises persistence + chunk_id assignment.
    """
    monkeypatch.setattr(
        "src.ingestion.extract_text_from_pdf",
        lambda pdf_path: [{"page": 1, "text": "raw text"}]
    )
    monkeypatch.setattr(
        "src.ingestion.clean_text",
        lambda text: text
    )
    monkeypatch.setattr(
        "src.ingestion.create_document_chunks",
        lambda pages, source: chunks
    )


# ---------------------------------------------------------------
# ADDING A NEW DOCUMENT
# ---------------------------------------------------------------

def test_add_new_document_saves_chunks_starting_at_id_one(tmp_path, monkeypatch):
    db_path = make_db_path(tmp_path)
    patch_pipeline(monkeypatch, fake_chunks("a.pdf", 2))

    result = add_or_replace_document("a.pdf", db_path)

    saved = load_all_chunks(db_path)
    assert [chunk["chunk_id"] for chunk in saved] == [1, 2]
    assert all(chunk["source"] == "a.pdf" for chunk in saved)
    assert result == saved


def test_add_or_replace_document_returns_the_saved_chunks(tmp_path, monkeypatch):
    db_path = make_db_path(tmp_path)
    patch_pipeline(monkeypatch, fake_chunks("a.pdf", 1))

    result = add_or_replace_document("a.pdf", db_path)

    assert len(result) == 1
    assert result[0]["source"] == "a.pdf"


# ---------------------------------------------------------------
# MULTIPLE DOCUMENTS: CHUNK ID CONTINUITY
# ---------------------------------------------------------------

def test_second_document_continues_the_chunk_id_sequence(tmp_path, monkeypatch):
    db_path = make_db_path(tmp_path)

    patch_pipeline(monkeypatch, fake_chunks("a.pdf", 2))
    add_or_replace_document("a.pdf", db_path)

    patch_pipeline(monkeypatch, fake_chunks("b.pdf", 2))
    add_or_replace_document("b.pdf", db_path)

    saved = load_all_chunks(db_path)
    a_chunks = [c for c in saved if c["source"] == "a.pdf"]
    b_chunks = [c for c in saved if c["source"] == "b.pdf"]

    assert [c["chunk_id"] for c in a_chunks] == [1, 2]
    assert [c["chunk_id"] for c in b_chunks] == [3, 4]


# ---------------------------------------------------------------
# REPLACING AN EXISTING DOCUMENT
# ---------------------------------------------------------------

def test_replacing_a_document_removes_its_old_chunks(tmp_path, monkeypatch):
    db_path = make_db_path(tmp_path)

    patch_pipeline(monkeypatch, fake_chunks("a.pdf", 2))
    add_or_replace_document("a.pdf", db_path)

    patch_pipeline(monkeypatch, fake_chunks("a.pdf", 1))
    add_or_replace_document("a.pdf", db_path)

    saved = load_all_chunks(db_path)
    assert len(saved) == 1
    assert saved[0]["text"] == "a.pdf chunk 1"


def test_replacing_one_document_does_not_affect_another(tmp_path, monkeypatch):
    db_path = make_db_path(tmp_path)

    patch_pipeline(monkeypatch, fake_chunks("a.pdf", 2))
    add_or_replace_document("a.pdf", db_path)

    patch_pipeline(monkeypatch, fake_chunks("b.pdf", 2))
    add_or_replace_document("b.pdf", db_path)

    patch_pipeline(monkeypatch, fake_chunks("a.pdf", 1))
    add_or_replace_document("a.pdf", db_path)

    saved = load_all_chunks(db_path)
    b_chunks = [c for c in saved if c["source"] == "b.pdf"]

    assert len(b_chunks) == 2
    assert [c["chunk_id"] for c in b_chunks] == [3, 4]


def test_replacing_a_document_assigns_new_chunk_ids_not_reused_ones(tmp_path, monkeypatch):
    db_path = make_db_path(tmp_path)

    patch_pipeline(monkeypatch, fake_chunks("a.pdf", 2))
    add_or_replace_document("a.pdf", db_path)  # chunk_ids 1, 2

    patch_pipeline(monkeypatch, fake_chunks("b.pdf", 2))
    add_or_replace_document("b.pdf", db_path)  # chunk_ids 3, 4

    patch_pipeline(monkeypatch, fake_chunks("a.pdf", 1))
    add_or_replace_document("a.pdf", db_path)  # replaces a.pdf

    saved = load_all_chunks(db_path)
    a_chunks = [c for c in saved if c["source"] == "a.pdf"]

    # a.pdf's old ids (1, 2) were freed by the delete, but chunk_ids
    # must never be reused/reassigned - the new chunk must continue
    # from the current global max (4), not restart at 1.
    assert a_chunks[0]["chunk_id"] == 5
    