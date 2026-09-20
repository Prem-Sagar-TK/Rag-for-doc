"""Ingestion and chunking tests for PDF, TXT, and CSV."""

from __future__ import annotations


def test_pdf_ingestion_preserves_page_metadata(ingestion, sample_pdf_bytes):
    doc = ingestion.ingest_bytes(filename="employee_policy.pdf", data=sample_pdf_bytes)
    assert doc.status.value == "ready"
    assert doc.file_type.value == "pdf"
    assert doc.chunk_count > 0

    chunks = ingestion.store.get_chunks_for_document(doc.id)
    assert chunks
    pages = {c["metadata"].get("page_number") for c in chunks}
    assert 1 in pages or "1" in {str(p) for p in pages}
    for c in chunks:
        meta = c["metadata"]
        assert meta["document_id"] == doc.id
        assert meta["filename"] == "employee_policy.pdf"
        assert meta["file_type"] == "pdf"
        assert meta.get("chunk_id")


def test_txt_ingestion_and_chunking(ingestion, sample_txt_bytes):
    doc = ingestion.ingest_bytes(filename="remote_work.txt", data=sample_txt_bytes)
    assert doc.status.value == "ready"
    assert doc.file_type.value == "txt"
    assert doc.chunk_count >= 1
    chunks = ingestion.store.get_chunks_for_document(doc.id)
    assert any("three days" in (c["content"] or "").lower() for c in chunks)
    for c in chunks:
        assert c["metadata"]["file_type"] == "txt"
        assert c["metadata"]["document_id"] == doc.id


def test_csv_ingestion_preserves_columns_and_rows(ingestion, sample_csv_bytes):
    doc = ingestion.ingest_bytes(filename="employees.csv", data=sample_csv_bytes)
    assert doc.status.value == "ready"
    assert doc.file_type.value == "csv"
    assert doc.chunk_count == 2  # one chunk per row

    chunks = ingestion.store.get_chunks_for_document(doc.id)
    assert len(chunks) == 2
    for c in chunks:
        meta = c["metadata"]
        assert meta["file_type"] == "csv"
        assert meta.get("row_number") is not None
        assert "name" in str(meta.get("columns", "")).lower() or "name" in c["content"].lower()
        assert "role" in c["content"].lower()

    contents = " ".join(c["content"] for c in chunks)
    assert "Alice Johnson" in contents
    assert "Engineer" in contents


def test_unsupported_file_type(ingestion):
    import pytest

    with pytest.raises(ValueError, match="Unsupported"):
        ingestion.ingest_bytes(filename="notes.docx", data=b"hello")


def test_chunking_unit_pdf():
    from app.rag.chunking import chunk_pdf_pages

    pages = [
        (1, "The company provides 20 days of annual leave."),
        (2, "Employees receive 10 days of sick leave."),
    ]
    chunks = chunk_pdf_pages(pages, document_id="d1", filename="policy.pdf")
    assert len(chunks) >= 2
    assert all(c.metadata["page_number"] in (1, 2) for c in chunks)
    assert all(c.metadata["chunk_id"].startswith("chunk_") for c in chunks)


def test_chunking_unit_csv():
    from app.rag.chunking import chunk_csv_rows

    rows = [{"name": "Alice", "role": "Engineer"}]
    chunks = chunk_csv_rows(
        rows, ["name", "role"], document_id="d1", filename="e.csv"
    )
    assert len(chunks) == 1
    assert chunks[0].metadata["row_number"] == 1
    assert chunks[0].metadata["columns"] == ["name", "role"]
    assert "Alice" in chunks[0].content
