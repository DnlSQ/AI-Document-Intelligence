"""
PDF extraction, including table-aware reconstruction (RAG v6.2).

PyMuPDF's default page.get_text() flattens multi-column datasheet
tables into a linear, column-less text stream - confirmed during
RAG v5.5 to make some technical facts genuinely ambiguous, causing
the LLM to correctly decline answering rather than guess. PyMuPDF's
page.find_tables() does detect the real row/column structure (RAG
v6.2.1's diagnose_tables.py confirmed this against a real page of
NE555N.pdf), so this module now uses it to add clean, unambiguous
"Symbol: X | Parameter: Y | ...: value | Unit: Z" facts alongside
the original plain text - never instead of it, so no page can lose
information it already had.
"""
import pymupdf


def extract_text_from_pdf(file_path):
    """
    Extracts text from every page of a PDF, preserving page numbers.

    Each page's text is the original plain-text extraction, plus any
    detected tables reconstructed into explicit structured facts
    appended at the end - see _extract_page_text.
    """
    document = pymupdf.open(file_path)
    pages = []
    for page_number, page in enumerate(document, start=1):
        text = _extract_page_text(page)
        pages.append({"page": page_number, "text": text})
    document.close()
    return pages


def _extract_page_text(page):
    """
    Extracts one page's text, augmented with any real tables it
    contains.

    Always returns the plain-text extraction unchanged as a prefix -
    this is the safe fallback required by RAG v6.2.3: a page with no
    table, or only degenerate/non-data tables (see
    _reconstruct_table), behaves exactly as before. Detected tables
    only ever ADD text, so this can't cause a regression on pages
    that already worked correctly (sample.pdf, plantas.pdf).
    """
    plain_text = page.get_text()

    reconstructed_sections = [
        reconstructed
        for reconstructed in (
            _reconstruct_table(table) for table in page.find_tables().tables
        )
        if reconstructed
    ]

    if not reconstructed_sections:
        return plain_text

    return plain_text + "\n\n" + "\n\n".join(reconstructed_sections)


def _reconstruct_table(table):
    """
    Turns one detected table's rows into explicit
    "Symbol: X | Parameter: Y | <column>: value | ... | Unit: Z"
    lines, instead of PyMuPDF's default columnless text dump.

    Skips tables with fewer than 2 rows outright: a real data table
    needs at least a header row and a data row, so anything smaller
    is a non-data artifact - RAG v6.2.1 confirmed PyMuPDF detects a
    page footer as a 1-row, 2-column "table" on the same NE555N.pdf
    page that has the real data table.

    Rows whose Symbol column is empty are skipped too (e.g. stray
    notes or continuation rows with no technical identifier of their
    own) - a data row that names no symbol has nothing to attribute
    a value to.
    """
    rows = table.extract()

    if len(rows) < 2:
        return ""

    column_labels, first_data_row = _build_column_labels(rows)

    lines = []
    for row in rows[first_data_row:]:
        symbol = _clean_cell(row[0], join_with_space=False)

        if not symbol:
            continue

        parameter = _clean_cell(row[1], join_with_space=True)

        parts = [f"Symbol: {symbol}"]
        if parameter:
            parts.append(f"Parameter: {parameter}")

        for column_index in range(2, len(row)):
            value = _clean_cell(row[column_index], join_with_space=True)
            if not value:
                continue
            label = (
                column_labels[column_index]
                if column_index < len(column_labels)
                else f"Column {column_index}"
            )
            parts.append(f"{label}: {value}")

        lines.append(" | ".join(parts))

    return "\n".join(lines)


def _build_column_labels(table_rows):
    """
    Combines a table's header row(s) into one label per column.

    Datasheet tables can split header labels across two rows with
    merged cells - e.g. row 0 = ["Symbol", "Parameter", "SE555",
    None, None, ...], row 1 = [None, None, "Min.", "Typ.", "Max.",
    ...] (confirmed via RAG v6.2.1's diagnose_tables.py). A merged
    cell's continuation extracts as None, so row 0 is forward-filled
    left-to-right before combining with row 1.

    Row 1 is treated as a second header row only when its first two
    columns (Symbol, Parameter) are both empty - a real data row
    always names a symbol or a parameter, so a blank pair there means
    row 1 is still part of the header, not the first data row.

    Returns (column_labels, first_data_row_index).
    """
    header_row = table_rows[0]

    filled_header = []
    last_seen = ""
    for cell in header_row:
        if cell:
            last_seen = cell.strip()
        filled_header.append(last_seen)

    if len(table_rows) > 1:
        second_row = table_rows[1]
        looks_like_header_continuation = not second_row[0] and not second_row[1]
        if looks_like_header_continuation:
            column_labels = [
                f"{filled_header[i]} {second_row[i]}".strip()
                if second_row[i]
                else filled_header[i]
                for i in range(len(filled_header))
            ]
            return column_labels, 2

    return filled_header, 1


def _clean_cell(cell_text, join_with_space):
    """
    Joins a table cell's internal lines back into a single string.

    PyMuPDF's table extraction splits subscripted characters onto
    their own line inside a cell (e.g. the symbol "toff" extracts as
    "t\\noff"). Joining with an empty separator reconstructs these
    correctly. Cells with genuine wrapped free text (e.g. a
    multi-line Parameter description) need a space instead, or words
    would run together. join_with_space picks which behavior a
    column needs: False for compact technical identifiers (Symbol),
    where a space would break the identifier itself; True for
    everything else.
    """
    if not cell_text:
        return ""
    separator = " " if join_with_space else ""
    return separator.join(line.strip() for line in cell_text.split("\n") if line.strip())
