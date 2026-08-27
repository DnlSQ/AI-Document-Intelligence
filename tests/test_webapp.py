"""
Tests for src.webapp (RAG v5): Flask app skeleton, the read-only
home page (V5.1), and the ask-a-question route (V5.2).

_get_state and answer_question are mocked in every route test so
these tests never touch the real embedding model, SQLite,
ChromaDB, or Ollama - only the routing/rendering logic this app
adds.
"""
import src.webapp as webapp


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
        {"chunk_id": 1, "page": 1, "text": "a", "source": "b.pdf"},
        {"chunk_id": 2, "page": 1, "text": "b", "source": "a.pdf"},
        {"chunk_id": 3, "page": 2, "text": "c", "source": "a.pdf"},
    ]

    summary = webapp._document_summary(chunks)

    assert summary == [
        {"source": "a.pdf", "chunk_count": 2},
        {"source": "b.pdf", "chunk_count": 1},
    ]


def test_document_summary_returns_empty_list_for_no_chunks():
    assert webapp._document_summary([]) == []


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
    