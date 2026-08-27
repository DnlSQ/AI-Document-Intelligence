"""
Tests for src.config.discover_document_paths (RAG v4.5): scanning
the documents folder for PDFs instead of relying on a hardcoded
list, so the system picks up documents added directly to the
folder (and, later, ones uploaded through the web interface)
without a code change.
"""
from src.config import discover_document_paths


def test_discover_document_paths_finds_pdf_files_in_folder(tmp_path):
    (tmp_path / "a.pdf").write_text("fake pdf content")
    (tmp_path / "b.pdf").write_text("fake pdf content")

    found = discover_document_paths(str(tmp_path))

    assert found == [
        f"{tmp_path}/a.pdf",
        f"{tmp_path}/b.pdf",
    ]


def test_discover_document_paths_ignores_non_pdf_files(tmp_path):
    (tmp_path / "a.pdf").write_text("fake pdf content")
    (tmp_path / "notes.txt").write_text("not a pdf")

    found = discover_document_paths(str(tmp_path))

    assert found == [f"{tmp_path}/a.pdf"]


def test_discover_document_paths_returns_empty_list_for_empty_folder(tmp_path):
    assert discover_document_paths(str(tmp_path)) == []


def test_discover_document_paths_returns_empty_list_when_folder_does_not_exist(tmp_path):
    missing_folder = str(tmp_path / "does_not_exist")

    assert discover_document_paths(missing_folder) == []


def test_discover_document_paths_is_case_insensitive_to_extension(tmp_path):
    (tmp_path / "A.PDF").write_text("fake pdf content")

    found = discover_document_paths(str(tmp_path))

    assert found == [f"{tmp_path}/A.PDF"]


def test_discover_document_paths_returns_sorted_paths(tmp_path):
    (tmp_path / "zebra.pdf").write_text("fake pdf content")
    (tmp_path / "alpha.pdf").write_text("fake pdf content")

    found = discover_document_paths(str(tmp_path))

    assert found == [
        f"{tmp_path}/alpha.pdf",
        f"{tmp_path}/zebra.pdf",
    ]
    