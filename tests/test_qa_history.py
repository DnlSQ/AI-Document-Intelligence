"""
Tests for the Q&A history persistence layer (src/qa_history.py),
RAG v7.3.1.

Uses pytest's tmp_path fixture for a fresh SQLite file per test,
mirroring test_chunk_store.py's isolation pattern (V4 testing
convention) - never touches the real data/qa_history.db.
"""
from src.qa_history import init_db, save_qa_pair, load_history


def test_init_db_creates_the_database_file(tmp_path):
    db_path = str(tmp_path / "qa_history.db")

    init_db(db_path)

    assert (tmp_path / "qa_history.db").exists()


def test_save_qa_pair_persists_a_question_and_answer(tmp_path):
    db_path = str(tmp_path / "qa_history.db")

    save_qa_pair("What is the max voltage?", "-50 V", db_path=db_path)

    history = load_history(db_path=db_path)

    assert len(history) == 1
    assert history[0]["question"] == "What is the max voltage?"
    assert history[0]["answer"] == "-50 V"
    assert history[0]["asked_at"]  # a timestamp was recorded automatically


def test_save_qa_pair_uses_the_provided_timestamp_when_given(tmp_path):
    db_path = str(tmp_path / "qa_history.db")

    save_qa_pair(
        "question", "answer", db_path=db_path,
        asked_at="2026-08-31T12:00:00+00:00"
    )

    history = load_history(db_path=db_path)

    assert history[0]["asked_at"] == "2026-08-31T12:00:00+00:00"


def test_load_history_returns_most_recent_first(tmp_path):
    db_path = str(tmp_path / "qa_history.db")

    save_qa_pair("first question", "first answer", db_path=db_path)
    save_qa_pair("second question", "second answer", db_path=db_path)
    save_qa_pair("third question", "third answer", db_path=db_path)

    history = load_history(db_path=db_path)

    assert [entry["question"] for entry in history] == [
        "third question", "second question", "first question"
    ]


def test_load_history_respects_limit(tmp_path):
    db_path = str(tmp_path / "qa_history.db")

    for i in range(5):
        save_qa_pair(f"question {i}", f"answer {i}", db_path=db_path)

    history = load_history(db_path=db_path, limit=2)

    assert len(history) == 2
    assert [entry["question"] for entry in history] == ["question 4", "question 3"]


def test_load_history_on_empty_database_returns_empty_list(tmp_path):
    db_path = str(tmp_path / "qa_history.db")

    history = load_history(db_path=db_path)

    assert history == []
    