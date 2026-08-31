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
    _split_multi_symbol_row,
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

# The real row shape confirmed by diagnose_multi_symbol_tables.py
# (RAG v7.2.2) against NE555N.pdf, page 5: "tr" and "tf" (Output
# rise/fall time) share one visual table row, with every value
# column split into two matching lines - one per symbol - unlike
# toff's "t\noff" (one symbol split by a subscript, whose Parameter
# cell just happens to wrap across two lines too, but no value
# column splits to match).
NE555_TR_TF_TABLE_ROWS = [
    ["Symbol", "Parameter", "SE555", None, None, "NE555 - SA555", None, None, "Unit"],
    [None, None, "Min.", "Typ.", "Max.", "Min.", "Typ.", "Max.", None],
    ["tr\ntf", "Output rise time\nOutput fall time", "100\n100", "200\n200", "", "100\n100", "300\n300", "", "ns"],
]


def test_split_multi_symbol_row_splits_matching_symbol_and_parameter_lines():
    row = NE555_TR_TF_TABLE_ROWS[2]

    sub_rows = _split_multi_symbol_row(row)

    assert len(sub_rows) == 2
    assert sub_rows[0][0] == "tr"
    assert sub_rows[0][1] == "Output rise time"
    assert sub_rows[1][0] == "tf"
    assert sub_rows[1][1] == "Output fall time"


def test_split_multi_symbol_row_assigns_matching_per_symbol_values():
    sub_rows = _split_multi_symbol_row(NE555_TR_TF_TABLE_ROWS[2])

    # Column 2 = SE555 Min. ("100\n100"), column 6 = NE555 Typ. ("300\n300")
    assert sub_rows[0][2] == "100"
    assert sub_rows[1][2] == "100"
    assert sub_rows[0][6] == "300"
    assert sub_rows[1][6] == "300"


def test_split_multi_symbol_row_broadcasts_shared_single_line_columns():
    sub_rows = _split_multi_symbol_row(NE555_TR_TF_TABLE_ROWS[2])

    # Column 8 = Unit ("ns"), a single value shared by both symbols.
    assert sub_rows[0][8] == "ns"
    assert sub_rows[1][8] == "ns"


def test_split_multi_symbol_row_returns_none_for_a_single_symbol_split_by_subscript():
    """
    toff's Parameter cell happens to wrap across 2 lines too (same
    count as its "t\\noff" symbol split), but none of its value
    columns split to match - the corroborating evidence tr/tf has
    and toff doesn't. Must NOT be mistaken for a packed multi-symbol
    row, or the already-correct toff fact would be destroyed.
    """
    toff_row = NE555_TOFF_TABLE_ROWS[2]

    assert _split_multi_symbol_row(toff_row) is None


def test_split_multi_symbol_row_returns_none_when_parameter_count_does_not_match():
    row = [
        "V\nOH",
        "High level output voltage\nsome extra condition\nanother condition",
        "13", "12.5", "", "12.7", "12.5", "", "V",
    ]

    assert _split_multi_symbol_row(row) is None


def test_split_multi_symbol_row_returns_none_when_a_value_column_count_is_inconsistent():
    row = ["tr\ntf", "Output rise time\nOutput fall time", "100\n200\n300", "", "", "", "", "", "ns"]

    assert _split_multi_symbol_row(row) is None


def test_split_multi_symbol_row_returns_none_for_a_normal_single_line_symbol():
    row = ["toff", "Turn off time", "", "0.5", "", "", "0.5", "", "µs"]

    assert _split_multi_symbol_row(row) is None


def test_reconstruct_table_produces_separate_facts_for_a_packed_multi_symbol_row():
    table = FakeTable(NE555_TR_TF_TABLE_ROWS)

    reconstructed = _reconstruct_table(table)

    assert "Symbol: tr | Parameter: Output rise time" in reconstructed
    assert "Symbol: tf | Parameter: Output fall time" in reconstructed
    assert "SE555 Typ.: 200" in reconstructed
    assert "NE555 - SA555 Typ.: 300" in reconstructed
    assert "Unit: ns" in reconstructed
    # Never merged into one garbled identifier.
    assert "Symbol: trtf" not in reconstructed


def test_reconstruct_table_still_reconstructs_toff_correctly_when_a_packed_row_is_also_present():
    """
    Regression check: a table containing BOTH a packed multi-symbol
    row (tr/tf) and a normal single-symbol row (toff) - the real
    shape of NE555N.pdf's page 5 table - must reconstruct both
    correctly, with no cross-contamination between them.
    """
    combined_rows = NE555_TR_TF_TABLE_ROWS[:2] + [NE555_TR_TF_TABLE_ROWS[2], NE555_TOFF_TABLE_ROWS[2]]
    table = FakeTable(combined_rows)

    reconstructed = _reconstruct_table(table)

    assert "Symbol: tr |" in reconstructed
    assert "Symbol: tf |" in reconstructed
    assert "Symbol: toff |" in reconstructed

def test_split_multi_symbol_row_regroups_symbol_lines_when_each_symbol_is_itself_subscript_split():
    """
    RAG v7.2.2 round 2: PyMuPDF can split EACH packed symbol across
    its own base+subscript lines too (e.g. "tr" renders as "t\\nr",
    "tf" as "t\\nf", giving a packed cell of 4 lines total for 2
    symbols, not 2) - the real shape confirmed against NE555N.pdf's
    tr/tf row. The count must come from the corroborating
    Parameter/value columns, then the Symbol column's lines are
    grouped into that many equal chunks - 4 lines / 2 symbols = 2
    lines per symbol here, but this adapts to any divisible count.
    """
    row = ["t\nr\nt\nf", "Output rise time\nOutput fall time", "100\n100", "200\n200", "", "100\n100", "300\n300", "", "ns"]

    sub_rows = _split_multi_symbol_row(row)

    assert sub_rows is not None
    assert sub_rows[0][0] == "tr"
    assert sub_rows[1][0] == "tf"
    assert sub_rows[0][1] == "Output rise time"
    assert sub_rows[1][1] == "Output fall time"


def test_split_multi_symbol_row_returns_none_when_symbol_lines_do_not_divide_evenly():
    row = ["t\nr\nt", "Output rise time\nOutput fall time", "100\n100", "", "", "", "", "", "ns"]

    assert _split_multi_symbol_row(row) is None
    