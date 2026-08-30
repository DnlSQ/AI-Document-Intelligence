"""
Flask web interface (RAG v5).

Exposes the existing RAG pipeline (src.main) through a small local
web UI: a home page listing loaded documents, a form to ask
questions, and a form to upload/replace documents in the persistent
library built in RAG v4.
"""
import os
import threading

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.main import initialize_system, answer_question
from src.ingestion import add_or_replace_document, replace_document_vectors, delete_document
from src.chunk_store import load_all_chunks
from src.config import DOCUMENTS_FOLDER, CHUNK_DB_PATH

app = Flask(__name__, template_folder="../templates")

_state = None
_state_lock = threading.Lock()


def _get_state():
    global _state
    if _state is None:
        chunks, collection = initialize_system()
        _state = {"chunks": chunks, "collection": collection}
    return _state


def _document_summary(chunks):
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
    state = _get_state()
    documents = _document_summary(state["chunks"])
    question = request.form.get("question", "").strip()
    if not question:
        return render_template("index.html", documents=documents, error="Please enter a question.")
    try:
        answer = answer_question(question, state["chunks"], collection=state["collection"])
    except Exception:
        return render_template(
            "index.html", documents=documents, question=question,
            error="The AI assistant is unavailable right now. Make sure Ollama is running and try again."
        )
    return render_template("index.html", documents=documents, question=question, answer=answer)


@app.route("/upload", methods=["POST"])
def upload():
    state = _get_state()
    documents = _document_summary(state["chunks"])

    uploaded_file = request.files.get("document")
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template(
            "index.html", documents=documents,
            upload_error="Please choose a file to upload."
        )

    filename = secure_filename(uploaded_file.filename)
    if not filename.lower().endswith(".pdf"):
        return render_template(
            "index.html", documents=documents,
            upload_error="Please upload a PDF file."
        )

    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    saved_path = f"{DOCUMENTS_FOLDER.rstrip('/')}/{filename}"
    uploaded_file.save(saved_path)

    chunks = add_or_replace_document(saved_path)
    replace_document_vectors(chunks, saved_path, collection=state["collection"])

    with _state_lock:
        state["chunks"] = load_all_chunks(CHUNK_DB_PATH)

    documents = _document_summary(state["chunks"])

    if not chunks:
        return render_template(
            "index.html", documents=documents,
            upload_warning=(
                f"'{filename}' was uploaded, but no text could be extracted from it "
                "(it may be a scanned or image-only PDF)."
            )
        )

    return render_template(
        "index.html", documents=documents,
        upload_message=f"'{filename}' was uploaded and is ready to use."
    )

@app.route("/delete", methods=["POST"])
def delete():
    state = _get_state()
    documents = _document_summary(state["chunks"])

    source = request.form.get("source", "").strip()
    known_sources = {doc["source"] for doc in documents}

    if not source or source not in known_sources:
        return render_template(
            "index.html", documents=documents,
            delete_error="Please choose a valid document to delete."
        )

    delete_document(source, collection=state["collection"])

    if os.path.exists(source):
        os.remove(source)

    with _state_lock:
        state["chunks"] = load_all_chunks(CHUNK_DB_PATH)

    documents = _document_summary(state["chunks"])

    return render_template(
        "index.html", documents=documents,
        delete_message=f"'{source}' was deleted."
    )

if __name__ == "__main__":
    app.run(debug=True)
    