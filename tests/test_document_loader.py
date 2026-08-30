"""
Tests for document_loader.py's table-aware extraction (RAG v6.2).

All synthetic - no real PDF or pymupdf dependency, per the testing
convention established in V2: real-document behavior is instead
validated by tests/test_document_loader_manual.py plus an end-to-end
browser + Ollama re-test of the original NE555N.pdf questions.

The synthetic table below reuses the real column/row shape confirmed
by RAG v6.2.1's diagnose_tables.py against page 5 of NE555N.pdf (a
two-row merged header, a "toff" data row with a subscript split
across lines) instead of a simplified made-up shape, so these tests
actually protect the real failure case v5.5 found.
"""
from src.document_loader import (
    _clean_cell,
    _build_column_labels,
    _reconstruct_table,
    _extract_page_text,
)


class FakeTable:
    """Stand-in for a pymupdf table object - _reconstruct_table only
    ever calls .extract() on it, so that's all this needs to provide."""

    def __init__(self, rows):
        self._rows = rows

    def extract(self):
        return self._rows


class FakeTableFinder:
    def __init__(self, tables):
        self.tables = tables


class FakePage:
    """Stand-in for a pymupdf.Page - only the two methods
    _extract_page_text actually calls."""

    def __init__(self, plain_text, tables):
        self._plain_text = plain_text
        self._tables = tables

    def get_text(self):
        return self._plain_text

    def find_tables(self):
        return FakeTableFinder(self._tables)


# The real row shape confirmed by diagnose_tables.py (RAG v6.2.1)
# against NE555N.pdf, page 5.
NE555_TOFF_TABLE_ROWS = [
    ["Symbol", "Parameter", "SE555", None, None, "NE555 - SA555", None, None, "Unit"],
    [None, None, "Min.", "Typ.", "Max.", "Min.", "Typ.", "Max.", None],
    ["t\noff", "Turn off time (5) (V = V )\nreset CC", "", "0.5", "", "", "0.5", "", "µs"],
]


def test_clean_cell_joins_subscript_split_symbol_with_no_separator():
    assert _clean_cell("t\noff", join_with_space=False) == "toff"


def test_clean_cell_joins_wrapped_text_with_spaces():
    result = _clean_cell("Turn off time (5) (V = V )\nreset CC", join_with_space=True)
    assert result == "Turn off time (5) (V = V ) reset CC"


def test_clean_cell_handles_empty_text():
    assert _clean_cell("", join_with_space=False) == ""
    assert _clean_cell(None, join_with_space=True) == ""


def test_build_column_labels_combines_merged_two_row_header():
    column_labels, first_data_row = _build_column_labels(NE555_TOFF_TABLE_ROWS)

    assert first_data_row == 2
    assert column_labels[2] == "SE555 Min."
    assert column_labels[3] == "SE555 Typ."
    assert column_labels[5] == "NE555 - SA555 Min."
    assert column_labels[6] == "NE555 - SA555 Typ."
    assert column_labels[8] == "Unit"


def test_reconstruct_table_produces_correct_toff_fact():
    table = FakeTable(NE555_TOFF_TABLE_ROWS)

    reconstructed = _reconstruct_table(table)

    assert "Symbol: toff" in reconstructed
    assert "Parameter: Turn off time" in reconstructed
    assert "SE555 Typ.: 0.5" in reconstructed
    assert "NE555 - SA555 Typ.: 0.5" in reconstructed
    assert "Unit: µs" in reconstructed
    # Empty Min./Max. columns for this row must not appear as facts.
    assert "SE555 Min.:" not in reconstructed
    assert "SE555 Max.:" not in reconstructed


def test_reconstruct_table_skips_tables_with_fewer_than_two_rows():
    footer_table = FakeTable([["", "Doc ID 2182 Rev 6 5/20"]])

    assert _reconstruct_table(footer_table) == ""


def test_reconstruct_table_skips_rows_with_no_symbol():
    rows = [
        ["Symbol", "Parameter", "Unit"],
        ["", "some stray note with no symbol", ""],
        ["toff", "Turn off time", "µs"],
    ]
    reconstructed = _reconstruct_table(FakeTable(rows))

    assert "stray note" not in reconstructed
    assert "Symbol: toff" in reconstructed


def test_extract_page_text_appends_reconstructed_table_to_plain_text():
    plain_text = "some flattened table text toff 0.5 µs"
    page = FakePage(plain_text, tables=[FakeTable(NE555_TOFF_TABLE_ROWS)])

    result = _extract_page_text(page)

    assert result.startswith(plain_text)
    assert "Symbol: toff" in result
    assert "SE555 Typ.: 0.5" in result


def test_extract_page_text_falls_back_to_plain_text_when_no_table_detected():
    plain_text = "a normal page with no tables at all"
    page = FakePage(plain_text, tables=[])

    assert _extract_page_text(page) == plain_text


def test_extract_page_text_falls_back_when_only_degenerate_tables_detected():
    plain_text = "a normal page"
    footer_table = FakeTable([["", "Doc ID 2182 Rev 6 5/20"]])
    page = FakePage(plain_text, tables=[footer_table])

    assert _extract_page_text(page) == plain_text
    