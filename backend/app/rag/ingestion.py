"""Type-aware document ingestion pipeline."""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Optional, Union

import pymupdf as fitz
import pandas as pd

from app.config import get_settings
from app.models.schemas import DocumentInfo, DocumentStatus, FileType
from app.rag.chunking import chunk_csv_rows, chunk_pdf_pages, chunk_txt
from app.vectorstore import get_vector_store


SUPPORTED_EXTENSIONS = {
    ".pdf": FileType.PDF,
    ".txt": FileType.TXT,
    ".csv": FileType.CSV,
}


class DocumentRegistry:
    """Simple JSON-backed document metadata store."""

    def __init__(self, path: Optional[Path] = None):
        settings = get_settings()
        self.path = path or settings.documents_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def _read(self) -> dict:
        import json

        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def _write(self, data: dict) -> None:
        import json

        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def upsert(self, doc: DocumentInfo) -> DocumentInfo:
        data = self._read()
        data[doc.id] = doc.model_dump(mode="json")
        self._write(data)
        return doc

    def get(self, document_id: str) -> Optional[DocumentInfo]:
        data = self._read()
        raw = data.get(document_id)
        if not raw:
            return None
        return DocumentInfo.model_validate(raw)

    def list(self) -> list[DocumentInfo]:
        data = self._read()
        docs = [DocumentInfo.model_validate(v) for v in data.values()]
        return sorted(docs, key=lambda d: d.created_at, reverse=True)

    def delete(self, document_id: str) -> bool:
        data = self._read()
        if document_id not in data:
            return False
        del data[document_id]
        self._write(data)
        return True

    def clear(self) -> None:
        self._write({})


class IngestionService:
    def __init__(
        self,
        store=None,
        registry: Optional[DocumentRegistry] = None,
    ):
        self.settings = get_settings()
        self.store = store or get_vector_store()
        self.registry = registry or DocumentRegistry()
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)

    def detect_type(self, filename: str) -> FileType:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}. Supported: PDF, TXT, CSV")
        return SUPPORTED_EXTENSIONS[ext]

    def ingest_bytes(
        self,
        *,
        filename: str,
        data: bytes,
        document_id: Optional[str] = None,
    ) -> DocumentInfo:
        file_type = self.detect_type(filename)
        doc_id = document_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        doc = DocumentInfo(
            id=doc_id,
            filename=filename,
            file_type=file_type,
            status=DocumentStatus.PROCESSING,
            chunk_count=0,
            created_at=now,
            size_bytes=len(data),
        )
        self.registry.upsert(doc)

        # Persist original file
        dest = self.settings.upload_dir / f"{doc_id}_{filename}"
        dest.write_bytes(data)

        try:
            chunks = self._build_chunks(file_type, filename, data, doc_id)
            count = self.store.add_chunks(chunks)
            doc.status = DocumentStatus.READY
            doc.chunk_count = count
            doc.error_message = None
        except Exception as exc:  # noqa: BLE001
            doc.status = DocumentStatus.ERROR
            doc.error_message = str(exc)
            doc.chunk_count = 0
        self.registry.upsert(doc)
        return doc

    def _build_chunks(self, file_type: FileType, filename: str, data: bytes, doc_id: str):
        if file_type == FileType.PDF:
            pages = self._extract_pdf_pages(data)
            return chunk_pdf_pages(pages, document_id=doc_id, filename=filename)
        if file_type == FileType.TXT:
            text = data.decode("utf-8", errors="replace")
            return chunk_txt(text, document_id=doc_id, filename=filename)
        if file_type == FileType.CSV:
            rows, columns = self._parse_csv(data)
            return chunk_csv_rows(
                rows,
                columns,
                document_id=doc_id,
                filename=filename,
                group_size=1,
            )
        raise ValueError(f"Unsupported type: {file_type}")

    @staticmethod
    def _extract_pdf_pages(data: bytes) -> list[tuple[int, str]]:
        doc = fitz.open(stream=data, filetype="pdf")
        pages: list[tuple[int, str]] = []
        try:
            for i, page in enumerate(doc, start=1):
                pages.append((i, page.get_text("text") or ""))
        finally:
            doc.close()
        return pages

    @staticmethod
    def _parse_csv(data: bytes) -> tuple[list[dict], list[str]]:
        # Prefer pandas for robust parsing; fall back to csv module
        try:
            df = pd.read_csv(io.BytesIO(data))
            columns = [str(c) for c in df.columns.tolist()]
            rows = df.fillna("").astype(str).to_dict(orient="records")
            return rows, columns
        except Exception:
            text = data.decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            columns = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
            return rows, columns

    def delete_document(self, document_id: str) -> bool:
        doc = self.registry.get(document_id)
        if not doc:
            return False
        self.store.delete_document(document_id)
        # Remove uploaded file if present
        for path in self.settings.upload_dir.glob(f"{document_id}_*"):
            try:
                path.unlink()
            except OSError:
                pass
        return self.registry.delete(document_id)

    def list_documents(self) -> list[DocumentInfo]:
        return self.registry.list()

    def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        return self.registry.get(document_id)
