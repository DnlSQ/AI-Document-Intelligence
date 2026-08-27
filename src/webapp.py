"""
Flask web interface (RAG v5).

Responsible only for:
    - HTTP routes: rendering pages, handling form submissions
    - Wiring requests to the existing RAG pipeline (main.py,
      ingestion.py) - it must not reimplement any retrieval,
      generation, or persistence logic itself

State (chunks + vector store collection) is loaded lazily on
first use via _get_state(), not at import time - mirrors the
lazy-singleton pattern embeddings.py already uses for its model,
so importing this module (e.g. in a test) never triggers a real
ingestion pass, real embeddings, or a real SQLite/ChromaDB read.
"""
from flask import Flask, render_template, request

from src.main import initialize_system, answer_question

# templates/ lives at the project root, one level up from src/ -
# keeps it alongside data/, tests/, docs/ rather than nested
# inside src/, matching the project's existing top-level layout.
app = Flask(__name__, template_folder="../templates")

_state = None


def _get_state():
    """
    Return the shared (chunks, collection) state, initializing it
    on first use via main.initialize_system - which itself reuses
    persisted data when available (see RAG v4) instead of always
    reprocessing every PDF from scratch.
    """
    global _state

    if _state is None:
        chunks, collection = initialize_system()
        _state = {"chunks": chunks, "collection": collection}

    return _state


def _document_summary(chunks):
    """
    Summarize the currently loaded chunks into a per-document
    view for the home page: one row per distinct source, with how
    many chunks it contributed.

    Args:
        chunks: List of chunk dicts (each with a "source" key).

    Returns:
        List of {"source": ..., "chunk_count": ...} dicts, sorted
        by source name for a stable display order.
    """
    counts = {}

    for chunk in chunks:
        source = chunk["source"]
        counts[source] = counts.get(source, 0) + 1

    return [
        {"source": source, "chunk_count": count}
        for source, count in sorted(counts.items())
    ]


@app.route("/")
def home():
    state = _get_state()
    documents = _document_summary(state["chunks"])

    return render_template("index.html", documents=documents)


@app.route("/ask", methods=["POST"])
def ask():
    """
    Answer a question submitted from the home page's form.

    Any failure to reach the LLM (most commonly: Ollama isn't
    running) is caught broadly and shown as a plain-language
    error instead of a raw stack trace - a technician using this
    page has no reason to see a Python traceback. Caught broadly
    (not a specific exception class from the ollama library)
    deliberately: this is a transport-boundary safety net, not
    retrieval/generation logic, and it shouldn't need to track
    which exact exception type a dependency two layers down
    happens to raise.
    """
    state = _get_state()
    documents = _document_summary(state["chunks"])
    question = request.form.get("question", "").strip()

    if not question:
        return render_template(
            "index.html",
            documents=documents,
            error="Please enter a question."
        )

    try:
        answer = answer_question(
            question,
            state["chunks"],
            collection=state["collection"]
        )
    except Exception:
        return render_template(
            "index.html",
            documents=documents,
            question=question,
            error="The AI assistant is unavailable right now. Make sure Ollama is running and try again."
        )

    return render_template(
        "index.html",
        documents=documents,
        question=question,
        answer=answer
    )


if __name__ == "__main__":
    app.run(debug=True)

    