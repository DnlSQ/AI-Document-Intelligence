"""
Tests for src.webapp (RAG v5.1): Flask app skeleton and the
read-only home page.

_get_state (and, through it, main.initialize_system) is mocked in
every route test so these tests never touch the real embedding
model, SQLite, or ChromaDB - only the routing/rendering logic
this phase adds.
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
    