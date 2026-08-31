"""
Tests for src.webapp (RAG v5): Flask app skeleton, the read-only
home page (V5.1), and the ask-a-question route (V5.2).

_get_state and answer_question are mocked in every route test so
these tests never touch the real embedding model, SQLite,
ChromaDB, or Ollama - only the routing/rendering logic this app
adds.
"""
import src.webapp as webapp
import io

def make_client():
    webapp.app.testing = True
    return webapp.app.test_client()


def reset_state(monkeypatch):
    """
    _state is a module-level singleton (see _get_state's
    docstring) - reset it before any test that exercises the
    lazy-initialization behavior itself, so tests don't leak
    state into each other.
    """
    monkeypatch.setattr(webapp, "_state", None)


# ---------------------------------------------------------------
# _document_summary - pure logic, no Flask needed
# ---------------------------------------------------------------

def test_document_summary_groups_by_source_and_counts_chunks():
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "a", "source": "data/documents/b.pdf"},
        {"chunk_id": 2, "page": 1, "text": "b", "source": "data/documents/a.pdf"},
        {"chunk_id": 3, "page": 2, "text": "c", "source": "data/documents/a.pdf"},
    ]

    summary = webapp._document_summary(chunks)

    assert summary == [
        {"source": "data/documents/a.pdf", "display_name": "a.pdf", "chunk_count": 2},
        {"source": "data/documents/b.pdf", "display_name": "b.pdf", "chunk_count": 1},
    ]


def test_document_summary_returns_empty_list_for_no_chunks():
    assert webapp._document_summary([]) == []


def test_document_summary_display_name_strips_directory_path():
    """
    RAG v7.1: the browser should show just the filename, not the full
    storage path - display_name is for showing, source (the full
    path) is still what gets submitted to /upload's replace-by-name
    matching and /delete, unchanged.
    """
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "a", "source": "data/documents/NE555N.pdf"},
    ]

    summary = webapp._document_summary(chunks)

    assert summary == [
        {"source": "data/documents/NE555N.pdf", "display_name": "NE555N.pdf", "chunk_count": 1},
    ]

# ---------------------------------------------------------------
# _get_state - lazy singleton
# ---------------------------------------------------------------

def test_get_state_initializes_only_once(monkeypatch):
    reset_state(monkeypatch)

    call_count = []

    def fake_initialize_system():
        call_count.append(1)
        return ([{"chunk_id": 1, "page": 1, "text": "x", "source": "a.pdf"}], "fake-collection")

    monkeypatch.setattr(webapp, "initialize_system", fake_initialize_system)

    first = webapp._get_state()
    second = webapp._get_state()

    assert len(call_count) == 1
    assert first is second
    assert first["collection"] == "fake-collection"


# ---------------------------------------------------------------
# GET / - home page
# ---------------------------------------------------------------

def test_home_page_lists_loaded_documents(monkeypatch):
    reset_state(monkeypatch)
    monkeypatch.setattr(
        webapp,
        "_get_state",
        lambda: {
            "chunks": [
                {"chunk_id": 1, "page": 1, "text": "x", "source": "sample.pdf"},
                {"chunk_id": 2, "page": 2, "text": "y", "source": "sample.pdf"},
                {"chunk_id": 3, "page": 1, "text": "z", "source": "plantas.pdf"},
            ],
            "collection": "fake-collection",
        }
    )

    response = make_client().get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "sample.pdf" in body
    assert "plantas.pdf" in body


def test_home_page_shows_empty_message_when_no_documents(monkeypatch):
    reset_state(monkeypatch)
    monkeypatch.setattr(
        webapp,
        "_get_state",
        lambda: {"chunks": [], "collection": "fake-collection"}
    )

    response = make_client().get("/")

    assert response.status_code == 200
    assert "No documents loaded yet" in response.get_data(as_text=True)


# ---------------------------------------------------------------
# POST /ask - question route
# ---------------------------------------------------------------

def test_ask_returns_grounded_answer(monkeypatch):
    reset_state(monkeypatch)
    monkeypatch.setattr(
        webapp,
        "_get_state",
        lambda: {
            "chunks": [{"chunk_id": 1, "page": 1, "text": "x", "source": "sample.pdf"}],
            "collection": "fake-collection",
        }
    )

    captured = {}

    def fake_answer_question(question, chunks, collection=None):
        captured["question"] = question
        captured["chunks"] = chunks
        captured["collection"] = collection
        return "The maximum collector-emitter voltage is -50 V."

    monkeypatch.setattr(webapp, "answer_question", fake_answer_question)

    response = make_client().post(
        "/ask",
        data={"question": "What is the maximum collector-emitter voltage?"}
    )

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "The maximum collector-emitter voltage is -50 V." in body
    assert "What is the maximum collector-emitter voltage?" in body
    assert captured["question"] == "What is the maximum collector-emitter voltage?"
    assert captured["collection"] == "fake-collection"


def test_ask_shows_friendly_error_when_llm_unavailable(monkeypatch):
    reset_state(monkeypatch)
    monkeypatch.setattr(
        webapp,
        "_get_state",
        lambda: {"chunks": [], "collection": "fake-collection"}
    )

    def fake_answer_question(question, chunks, collection=None):
        raise ConnectionError("Ollama is not reachable")

    monkeypatch.setattr(webapp, "answer_question", fake_answer_question)

    response = make_client().post("/ask", data={"question": "Anything?"})

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "unavailable" in body.lower()


def test_ask_with_empty_question_shows_validation_message(monkeypatch):
    reset_state(monkeypatch)
    monkeypatch.setattr(
        webapp,
        "_get_state",
        lambda: {"chunks": [], "collection": "fake-collection"}
    )

    calls = []
    monkeypatch.setattr(
        webapp,
        "answer_question",
        lambda *args, **kwargs: calls.append(1)
    )

    response = make_client().post("/ask", data={"question": "   "})

    assert response.status_code == 200
    assert "Please enter a question" in response.get_data(as_text=True)
    assert calls == []


def test_upload_rejects_missing_file():
    client = make_client()
    response = client.post("/upload", data={}, content_type="multipart/form-data")
    assert b"Please choose a file to upload." in response.data


def test_upload_rejects_non_pdf_file():
    client = make_client()
    data = {"document": (io.BytesIO(b"just some text"), "notes.txt")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert b"Please upload a PDF file." in response.data


def test_upload_saves_and_replaces_document(monkeypatch, tmp_path):
    reset_state(monkeypatch)
    client = make_client()

    saved_paths = []

    def fake_add_or_replace_document(path):
        saved_paths.append(path)
        return [{"chunk_id": 1, "page": 1, "text": "hello", "source": path}]

    def fake_replace_document_vectors(chunks, source, collection=None):
        return len(chunks)

    def fake_load_all_chunks(db_path=None):
        return [{"chunk_id": 1, "page": 1, "text": "hello", "source": "data/documents/test.pdf"}]

    monkeypatch.setattr(webapp, "_get_state", lambda: {"chunks": [], "collection": None})
    monkeypatch.setattr(webapp, "add_or_replace_document", fake_add_or_replace_document)
    monkeypatch.setattr(webapp, "replace_document_vectors", fake_replace_document_vectors)
    monkeypatch.setattr(webapp, "load_all_chunks", fake_load_all_chunks)
    monkeypatch.setattr(webapp, "DOCUMENTS_FOLDER", str(tmp_path))

    data = {"document": (io.BytesIO(b"%PDF-1.4 fake content"), "test.pdf")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert saved_paths, "add_or_replace_document should have been called"
    assert b"test.pdf" in response.data
    assert b"uploaded" in response.data.lower()


def test_upload_warns_when_no_chunks_extracted(monkeypatch, tmp_path):
    reset_state(monkeypatch)
    client = make_client()

    monkeypatch.setattr(webapp, "_get_state", lambda: {"chunks": [], "collection": None})
    monkeypatch.setattr(webapp, "add_or_replace_document", lambda path: [])
    monkeypatch.setattr(webapp, "replace_document_vectors", lambda chunks, source, collection=None: 0)
    monkeypatch.setattr(webapp, "load_all_chunks", lambda db_path=None: [])
    monkeypatch.setattr(webapp, "DOCUMENTS_FOLDER", str(tmp_path))

    data = {"document": (io.BytesIO(b"%PDF-1.4 empty"), "scanned.pdf")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert b"no text could be extracted" in response.data.lower()


def test_upload_sanitizes_filename(monkeypatch, tmp_path):
    reset_state(monkeypatch)
    client = make_client()

    saved_paths = []

    def fake_add_or_replace_document(path):
        saved_paths.append(path)
        return [{"chunk_id": 1, "page": 1, "text": "x", "source": path}]

    monkeypatch.setattr(webapp, "_get_state", lambda: {"chunks": [], "collection": None})
    monkeypatch.setattr(webapp, "add_or_replace_document", fake_add_or_replace_document)
    monkeypatch.setattr(webapp, "replace_document_vectors", lambda chunks, source, collection=None: 1)
    monkeypatch.setattr(webapp, "load_all_chunks", lambda db_path=None: [])
    monkeypatch.setattr(webapp, "DOCUMENTS_FOLDER", str(tmp_path))

    data = {"document": (io.BytesIO(b"%PDF-1.4 x"), "../../evil name.pdf")}
    client.post("/upload", data=data, content_type="multipart/form-data")

    assert saved_paths
    assert ".." not in saved_paths[0]
    assert " " not in saved_paths[0]


def test_upload_refreshes_document_list(monkeypatch, tmp_path):
    reset_state(monkeypatch)
    client = make_client()

    monkeypatch.setattr(webapp, "_get_state", lambda: {"chunks": [], "collection": None})
    monkeypatch.setattr(webapp, "add_or_replace_document", lambda path: [{"chunk_id": 1, "page": 1, "text": "x", "source": path}])
    monkeypatch.setattr(webapp, "replace_document_vectors", lambda chunks, source, collection=None: 1)
    monkeypatch.setattr(
        webapp, "load_all_chunks",
        lambda db_path=None: [{"chunk_id": 1, "page": 1, "text": "x", "source": "data/documents/fresh.pdf"}],
    )
    monkeypatch.setattr(webapp, "DOCUMENTS_FOLDER", str(tmp_path))

    data = {"document": (io.BytesIO(b"%PDF-1.4 x"), "fresh.pdf")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert b"fresh.pdf" in response.data

# ---------------------------------------------------------------
# POST /delete - remove a document entirely (RAG v6.1)
# ---------------------------------------------------------------

def test_delete_removes_document_and_refreshes_list(monkeypatch, tmp_path):
    reset_state(monkeypatch)
    client = make_client()

    pdf_path = tmp_path / "old.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    source = str(pdf_path)

    monkeypatch.setattr(
        webapp,
        "_get_state",
        lambda: {
            "chunks": [{"chunk_id": 1, "page": 1, "text": "x", "source": source}],
            "collection": "fake-collection",
        }
    )

    delete_calls = []
    monkeypatch.setattr(
        webapp, "delete_document",
        lambda src, collection=None: delete_calls.append((src, collection))
    )
    monkeypatch.setattr(webapp, "load_all_chunks", lambda db_path=None: [])

    response = client.post("/delete", data={"source": source})

    assert response.status_code == 200
    assert delete_calls == [(source, "fake-collection")]
    assert not pdf_path.exists(), "the file itself should be removed from disk"
    assert b"deleted" in response.data.lower()
    assert b"No documents loaded yet" in response.data


def test_delete_rejects_unknown_source(monkeypatch):
    reset_state(monkeypatch)
    client = make_client()

    monkeypatch.setattr(
        webapp,
        "_get_state",
        lambda: {
            "chunks": [{"chunk_id": 1, "page": 1, "text": "x", "source": "sample.pdf"}],
            "collection": "fake-collection",
        }
    )

    delete_calls = []
    monkeypatch.setattr(
        webapp, "delete_document",
        lambda src, collection=None: delete_calls.append(src)
    )

    response = client.post("/delete", data={"source": "not_a_real_document.pdf"})

    assert response.status_code == 200
    assert delete_calls == []
    assert b"sample.pdf" in response.data
    assert b"valid document" in response.data.lower()


def test_delete_rejects_missing_source(monkeypatch):
    reset_state(monkeypatch)
    client = make_client()

    monkeypatch.setattr(
        webapp,
        "_get_state",
        lambda: {"chunks": [], "collection": "fake-collection"}
    )

    delete_calls = []
    monkeypatch.setattr(
        webapp, "delete_document",
        lambda src, collection=None: delete_calls.append(src)
    )

    response = client.post("/delete", data={})

    assert response.status_code == 200
    assert delete_calls == []
    