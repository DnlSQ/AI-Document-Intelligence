"""
Flask web interface (RAG v5).

Exposes the existing RAG pipeline (src.main) through a small local
web UI: a home page listing loaded documents, a form to ask
questions, and a form to upload/replace documents in the persistent
library built in RAG v4.

RAG v7.3.1 adds a persistent Q&A history panel, and uses the
Post/Redirect/Get pattern on /ask: instead of rendering index.html
directly from the POST, /ask stores its result in the session and
redirects to home() (GET /). This means refreshing the page after
asking a question just re-runs a harmless GET instead of resubmitting
the form - so it no longer duplicates the entry into history or
triggers an extra Ollama call.
"""
import os
import threading

from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from src.main import initialize_system, answer_question
from src.ingestion import add_or_replace_document, replace_document_vectors, delete_document
from src.chunk_store import load_all_chunks
from src.qa_history import save_qa_pair, load_history
from src.config import DOCUMENTS_FOLDER, CHUNK_DB_PATH, MAX_UPLOAD_SIZE_BYTES

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Local, single-user tool with no authentication. The session is
# only ever used to pass the result of the last /ask POST to the
# following GET / (see ask()/home() below), so a static local key
# is fine here - there is nothing sensitive in it and nothing that
# needs to survive across app restarts.
app.secret_key = "ai-document-intelligence-local-dev-key"

# V8.4.1: reject an upload before it ever reaches the /upload route
# if the request body exceeds this size - Werkzeug enforces it at
# the WSGI level and raises RequestEntityTooLarge, handled by the
# errorhandler below so the user sees the app's normal error box
# instead of a generic 413 page.

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_BYTES
# Number of most recent Q&A pairs shown in the history panel.
HISTORY_DISPLAY_LIMIT = 10

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
        {
            "source": source,
            "display_name": os.path.basename(source),
            "chunk_count": count,
        }
        for source, count in sorted(counts.items())
    ]

# V8.4.1: minimal "magic bytes" signatures used to confirm an
# uploaded file's content actually matches its extension, not just
# its name - catches the simple, common case of a file renamed to
# look like a supported document. .docx and .xlsx are both
# OOXML/ZIP-based formats, so they share the same outer ZIP
# signature; this does not distinguish a mislabeled .xlsx from a
# mislabeled .docx (both are still valid ZIP archives), which is a
# known, accepted limitation - deeper validation would mean opening
# and inspecting the archive's internal structure, not worth the
# complexity unless a real case shows it's needed.
PDF_SIGNATURE = b"%PDF-"
ZIP_SIGNATURE = b"PK\x03\x04"


def _content_matches_extension(file_bytes, extension):
    """
    Check that a file's actual content matches what its extension
    claims, using each format's standard magic-byte signature.

    Args:
        file_bytes: The raw bytes read from the uploaded file.
        extension: Lowercase extension including the dot (e.g. ".pdf").

    Returns:
        True if the content's signature matches the extension's
        expected format, False otherwise (including for an
        extension this function doesn't recognize).
    """
    if extension == ".pdf":
        return file_bytes.startswith(PDF_SIGNATURE)
    if extension in (".docx", ".xlsx"):
        return file_bytes.startswith(ZIP_SIGNATURE)
    return False


def _is_within_documents_folder(path, folder):
    """
    Defense-in-depth check that a resolved upload path stays inside
    the documents folder. werkzeug's secure_filename (already
    applied to every uploaded filename before this is called) already
    strips path separators and ".." segments, so this should never
    actually trigger - it exists as an explicit, tested guarantee
    rather than an implicit assumption.
    """
    folder_abs = os.path.abspath(folder)
    path_abs = os.path.abspath(path)
    return path_abs == folder_abs or path_abs.startswith(folder_abs + os.sep)

@app.route("/")
def home():
    state = _get_state()
    documents = _document_summary(state["chunks"])
    history = load_history(limit=HISTORY_DISPLAY_LIMIT)

    # Post/Redirect/Get: /ask stores its outcome here and redirects
    # to this route instead of rendering the template directly.
    # session.pop removes it as soon as it's read, so it is shown
    # exactly once - a later plain GET / (e.g. pressing refresh)
    # renders a clean page instead of re-showing or re-submitting
    # the same question.
    ask_result = session.pop("ask_result", {})

    return render_template(
        "index.html", documents=documents, history=history,
        question=ask_result.get("question"),
        answer=ask_result.get("answer"),
        error=ask_result.get("error"),
    )


@app.route("/ask", methods=["POST"])
def ask():
    state = _get_state()
    question = request.form.get("question", "").strip()

    if not question:
        session["ask_result"] = {"error": "Please enter a question."}
        return redirect(url_for("home"))

    try:
        answer = answer_question(question, state["chunks"], collection=state["collection"])
    except Exception:
        session["ask_result"] = {
            "question": question,
            "error": "The AI assistant is unavailable right now. Make sure Ollama is running and try again.",
        }
        return redirect(url_for("home"))

    save_qa_pair(question, answer)
    session["ask_result"] = {"question": question, "answer": answer}
    return redirect(url_for("home"))


@app.route("/upload", methods=["POST"])
def upload():
    state = _get_state()
    documents = _document_summary(state["chunks"])
    history = load_history(limit=HISTORY_DISPLAY_LIMIT)

    uploaded_file = request.files.get("document")
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template(
            "index.html", documents=documents, history=history,
            upload_error="Please choose a file to upload."
        )

    filename = secure_filename(uploaded_file.filename)
    extension = os.path.splitext(filename)[1].lower()
    if extension not in (".pdf", ".docx", ".xlsx"):
        return render_template(
            "index.html", documents=documents, history=history,
            upload_error="Please upload a PDF, Word (.docx), or Excel (.xlsx) file."
        )

    file_bytes = uploaded_file.read()
    if not _content_matches_extension(file_bytes, extension):
        return render_template(
            "index.html", documents=documents, history=history,
            upload_error=(
                f"'{filename}' doesn't look like a valid {extension} file "
                "(its content doesn't match the file type)."
            )
        )

    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    saved_path = f"{DOCUMENTS_FOLDER.rstrip('/')}/{filename}"

    if not _is_within_documents_folder(saved_path, DOCUMENTS_FOLDER):
        return render_template(
            "index.html", documents=documents, history=history,
            upload_error="Invalid file name."
        )

    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    chunks = add_or_replace_document(saved_path)
    replace_document_vectors(chunks, saved_path, collection=state["collection"])

    with _state_lock:
        state["chunks"] = load_all_chunks(CHUNK_DB_PATH)

    documents = _document_summary(state["chunks"])

    if not chunks:
        return render_template(
            "index.html", documents=documents, history=history,
                        upload_warning=(
                f"'{filename}' was uploaded, but no text could be extracted from it "
                "(it may be empty, a scanned/image-only PDF, or a Word document "
                "with no readable text)."
            )
        )

    return render_template(
        "index.html", documents=documents, history=history,
        upload_message=f"'{filename}' was uploaded and is ready to use."
    )

@app.route("/delete", methods=["POST"])
def delete():
    state = _get_state()
    documents = _document_summary(state["chunks"])
    history = load_history(limit=HISTORY_DISPLAY_LIMIT)

    source = request.form.get("source", "").strip()
    known_sources = {doc["source"] for doc in documents}

    if not source or source not in known_sources:
        return render_template(
            "index.html", documents=documents, history=history,
            delete_error="Please choose a valid document to delete."
        )

    delete_document(source, collection=state["collection"])

    if os.path.exists(source):
        os.remove(source)

    with _state_lock:
        state["chunks"] = load_all_chunks(CHUNK_DB_PATH)

    documents = _document_summary(state["chunks"])

    return render_template(
        "index.html", documents=documents, history=history,
        delete_message=f"'{source}' was deleted."
    )

if __name__ == "__main__":
    app.run(debug=True)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    state = _get_state()
    documents = _document_summary(state["chunks"])
    history = load_history(limit=HISTORY_DISPLAY_LIMIT)
    max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
    return render_template(
        "index.html", documents=documents, history=history,
        upload_error=f"That file is too large. Please upload a file smaller than {max_mb} MB."
    ), 413
