"""
Tests for multi-document support.

Validates the original hypothesis behind this feature: when
chunks from multiple, topically unrelated documents are combined
into a single repository, the retriever must naturally surface
chunks from the RIGHT document for a given question - without
any manual document selection - because chunks from an unrelated
document score at or near 0 and get filtered out.

The plant biology text used here is real content extracted from
the uploaded "Plantas.pdf" (Reino Plantae: Características y
clasificación), not placeholder text.
"""

from src.main import build_chunk_repository
from src.retriever import retrieve_relevant_chunks


TRANSISTOR_PDF = "data/documents/sample.pdf"
PLANTS_PDF = "data/documents/plantas.pdf"


def _combined_chunks():
    """
    A small, realistic combined repository: a few chunks from
    the transistor datasheet, a few from the real plants PDF
    text.
    """

    transistor_chunks = [
        {
            "chunk_id": 1,
            "page": 2,
            "text": "VCEO collector-emitter voltage open base -50 V",
            "source": TRANSISTOR_PDF,
        },
        {
            "chunk_id": 2,
            "page": 1,
            "text": "IO output current -500 mA maximum rating",
            "source": TRANSISTOR_PDF,
        },
    ]

    plant_chunks = [
        {
            "chunk_id": 3,
            "page": 1,
            "text": (
                "El Reino Plantae incluye organismos eucariontas, "
                "pluricelulares y fotosinteticos, con paredes "
                "celulares de celulosa."
            ),
            "source": PLANTS_PDF,
        },
        {
            "chunk_id": 4,
            "page": 2,
            "text": (
                "Las gimnospermas presentan semilla desnuda y conos "
                "o estrobilos, mientras que las angiospermas "
                "desarrollan flores y frutos."
            ),
            "source": PLANTS_PDF,
        },
        {
            "chunk_id": 5,
            "page": 2,
            "text": (
                "Los tejidos de conduccion reciben el nombre de "
                "xilema y floema; el xilema transporta agua y "
                "sales minerales, mientras que el floema transporta "
                "azucares."
            ),
            "source": PLANTS_PDF,
        },
    ]

    return transistor_chunks + plant_chunks


# ---------------------------------------------------------------------
# build_chunk_repository - combining real documents
# ---------------------------------------------------------------------

def test_build_chunk_repository_reads_default_document_list():
    """
    Sanity check on config: both documents must be configured,
    and the newly added plants PDF must be present alongside the
    original transistor datasheet.
    """

    from src.config import DOCUMENT_PATHS

    assert TRANSISTOR_PDF in DOCUMENT_PATHS
    assert PLANTS_PDF in DOCUMENT_PATHS


# ---------------------------------------------------------------------
# retrieve_relevant_chunks - cross-document disambiguation
# ---------------------------------------------------------------------

def test_technical_query_surfaces_transistor_chunks_not_plant_chunks():
    """
    The original hypothesis this feature was built to test: a
    technical electronics question must retrieve chunks from the
    transistor datasheet, never from the unrelated plants PDF,
    even though both are searched together.
    """

    chunks = _combined_chunks()

    results = retrieve_relevant_chunks(
        "What is the maximum collector-emitter voltage?",
        chunks,
        top_k=3
    )

    assert results
    assert results[0]["chunk"]["source"] == TRANSISTOR_PDF

    returned_sources = {result["chunk"]["source"] for result in results}
    assert PLANTS_PDF not in returned_sources


def test_biology_query_surfaces_plant_chunks_not_transistor_chunks():
    """
    The reverse case: a biology question about gymnosperms must
    retrieve chunks from the plants PDF, never from the unrelated
    transistor datasheet.
    """

    chunks = _combined_chunks()

    results = retrieve_relevant_chunks(
        "Que son las gimnospermas?",
        chunks,
        top_k=3
    )

    assert results
    assert results[0]["chunk"]["source"] == PLANTS_PDF

    returned_sources = {result["chunk"]["source"] for result in results}
    assert TRANSISTOR_PDF not in returned_sources


def test_xylem_query_surfaces_correct_plant_chunk():
    """
    A second, distinct biology query (about xylem/phloem) must
    also correctly surface its matching plant chunk over the
    unrelated gymnosperms chunk and all transistor chunks -
    confirming this isn't a fluke tied to one specific query.
    """

    chunks = _combined_chunks()

    results = retrieve_relevant_chunks(
        "Que transporta el xilema?",
        chunks,
        top_k=3
    )

    assert results
    assert results[0]["chunk"]["chunk_id"] == 5
    assert results[0]["chunk"]["source"] == PLANTS_PDF


def test_unrelated_document_chunks_score_zero_and_are_filtered_out():
    """
    Direct confirmation of the mechanism that makes cross-document
    search work without manual selection: chunks from the wrong
    document score 0 for an off-topic query and are excluded from
    results entirely (not just ranked lower).
    """

    plant_only_chunks = [
        chunk for chunk in _combined_chunks() if chunk["source"] == PLANTS_PDF
    ]

    results = retrieve_relevant_chunks(
        "What is the maximum collector-emitter voltage?",
        plant_only_chunks,
        top_k=3
    )

    assert results == []
    