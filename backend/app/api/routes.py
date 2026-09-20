"""FastAPI route handlers."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkInfo,
    ChunkMetadata,
    DocumentInfo,
    EvaluationSummary,
    HealthResponse,
)
from app.rag.chunking import parse_columns_meta
from app.rag.ingestion import DocumentRegistry, IngestionService
from app.rag.pipeline import RAGPipeline
from app.evaluation.runner import EvaluationRunner
from app.vectorstore import get_vector_backend_name, get_vector_store

router = APIRouter(prefix="/api")


def _ingestion() -> IngestionService:
    return IngestionService()


def _pipeline() -> RAGPipeline:
    return RAGPipeline()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    registry = DocumentRegistry()
    # Ensure store is initialized so backend name is accurate
    get_vector_store()
    return HealthResponse(
        status="ok",
        openai_configured=bool(settings.openai_api_key)
        and not settings.use_fake_embeddings,
        document_count=len(registry.list()),
        relevance_threshold=settings.relevance_threshold,
        top_k=settings.top_k,
        vector_backend=get_vector_backend_name(),
    )


@router.post("/documents/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        doc = _ingestion().ingest_bytes(filename=file.filename, data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if doc.status.value == "error":
        raise HTTPException(
            status_code=422,
            detail=doc.error_message or "Ingestion failed",
        )
    return doc


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return _ingestion().list_documents()


@router.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    ok = _ingestion().delete_document(document_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True, "id": document_id}


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkInfo])
def get_chunks(document_id: str) -> list[ChunkInfo]:
    doc = _ingestion().get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    store = get_vector_store()
    raw = store.get_chunks_for_document(document_id)
    chunks: list[ChunkInfo] = []
    for item in raw:
        meta = item.get("metadata") or {}
        page = meta.get("page_number")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        row = meta.get("row_number")
        try:
            row = int(row) if row is not None else None
        except (TypeError, ValueError):
            row = None
        chunks.append(
            ChunkInfo(
                chunk_id=str(meta.get("chunk_id") or item.get("chunk_id")),
                content=item.get("content") or "",
                metadata=ChunkMetadata(
                    document_id=str(meta.get("document_id") or document_id),
                    filename=str(meta.get("filename") or doc.filename),
                    file_type=str(meta.get("file_type") or doc.file_type.value),
                    page_number=page,
                    chunk_id=str(meta.get("chunk_id") or item.get("chunk_id")),
                    row_number=row,
                    columns=parse_columns_meta(meta.get("columns")),
                ),
            )
        )
    return chunks


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return _pipeline().chat(question, document_ids=request.document_ids)


@router.post("/evaluation/run", response_model=EvaluationSummary)
def run_evaluation(document_ids: Optional[list[str]] = None) -> EvaluationSummary:
    return EvaluationRunner(_pipeline()).run(document_ids=document_ids)
