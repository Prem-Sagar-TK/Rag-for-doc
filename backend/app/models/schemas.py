"""Pydantic request/response schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class FileType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    CSV = "csv"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: FileType
    status: DocumentStatus
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    size_bytes: int = 0


class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    file_type: str
    page_number: Optional[int] = None
    chunk_id: str
    row_number: Optional[int] = None
    columns: Optional[list[str]] = None


class ChunkInfo(BaseModel):
    chunk_id: str
    content: str
    metadata: ChunkMetadata


class SourceCitation(BaseModel):
    filename: str
    page: Optional[int] = None
    chunk_id: str
    score: float
    row_number: Optional[int] = None
    preview: Optional[str] = None


class RetrievedChunkDebug(BaseModel):
    filename: str
    page: Optional[int] = None
    chunk_id: str
    score: float
    used: bool
    preview: str
    row_number: Optional[int] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    document_ids: Optional[list[str]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    abstained: bool = False
    debug: dict[str, Any] = Field(default_factory=dict)


class EvaluationCaseResult(BaseModel):
    question: str
    expected_answer: Optional[str] = None
    expected_source: Optional[str] = None
    expected_abstain: bool = False
    actual_answer: str
    sources: list[SourceCitation]
    abstained: bool
    retrieval_hit: bool
    answer_correct: Optional[bool] = None
    citation_correct: Optional[bool] = None
    max_score: float = 0.0


class EvaluationSummary(BaseModel):
    total_cases: int
    retrieval_hit_rate: float
    average_max_relevance: float
    answer_correctness: Optional[float] = None
    abstention_rate: float
    citation_correctness: Optional[float] = None
    cases: list[EvaluationCaseResult]


class HealthResponse(BaseModel):
    status: str
    openai_configured: bool
    document_count: int
    relevance_threshold: float
    top_k: int
    vector_backend: str = "unknown"
