"""Type-aware document chunking strategies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


@dataclass
class DocumentChunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_chunk_id() -> str:
    return f"chunk_{uuid.uuid4().hex[:12]}"


def chunk_pdf_pages(
    pages: list[tuple[int, str]],
    *,
    document_id: str,
    filename: str,
) -> list[DocumentChunk]:
    """Chunk PDF text while preserving page numbers."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[DocumentChunk] = []
    for page_number, text in pages:
        text = (text or "").strip()
        if not text:
            continue
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = _make_chunk_id()
            chunks.append(
                DocumentChunk(
                    content=piece,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "file_type": "pdf",
                        "page_number": page_number,
                        "chunk_id": chunk_id,
                    },
                )
            )
    return chunks


def chunk_txt(
    text: str,
    *,
    document_id: str,
    filename: str,
) -> list[DocumentChunk]:
    """Semantic/token-aware chunking for plain text."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[DocumentChunk] = []
    for piece in splitter.split_text(text or ""):
        piece = piece.strip()
        if not piece:
            continue
        chunk_id = _make_chunk_id()
        chunks.append(
            DocumentChunk(
                content=piece,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": "txt",
                    "page_number": None,
                    "chunk_id": chunk_id,
                },
            )
        )
    return chunks


def chunk_csv_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    document_id: str,
    filename: str,
    group_size: int = 1,
) -> list[DocumentChunk]:
    """Create row-level (or grouped) chunks that preserve column names."""
    chunks: list[DocumentChunk] = []
    if not rows:
        return chunks

    for start in range(0, len(rows), group_size):
        group = rows[start : start + group_size]
        lines: list[str] = []
        for offset, row in enumerate(group):
            row_num = start + offset + 1  # 1-indexed
            parts = [f"{col}: {row.get(col, '')}" for col in columns]
            lines.append(f"Row {row_num}: " + " | ".join(parts))
        content = "\n".join(lines)
        chunk_id = _make_chunk_id()
        row_number = start + 1
        chunks.append(
            DocumentChunk(
                content=content,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": "csv",
                    "page_number": None,
                    "chunk_id": chunk_id,
                    "row_number": row_number,
                    "columns": columns,
                },
            )
        )
    return chunks


def normalize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Flatten metadata for Chroma (no nested lists of non-primitives beyond strings)."""
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if key == "columns" and isinstance(value, list):
            out[key] = ",".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def parse_columns_meta(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    return [c for c in value.split(",") if c]
