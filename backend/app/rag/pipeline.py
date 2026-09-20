"""RAG chat orchestration service."""

from __future__ import annotations

from typing import Optional

from app.models.schemas import ChatResponse, SourceCitation
from app.rag.generator import AnswerGenerator
from app.rag.retriever import Retriever


class RAGPipeline:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[AnswerGenerator] = None,
    ):
        self.retriever = retriever or Retriever()
        self.generator = generator or AnswerGenerator()

    def chat(
        self,
        question: str,
        *,
        document_ids: Optional[list[str]] = None,
    ) -> ChatResponse:
        retrieval = self.retriever.retrieve(question, document_ids=document_ids)
        result = self.generator.generate(question, retrieval)
        sources = result["sources"]
        if sources and isinstance(sources[0], dict):
            sources = [SourceCitation.model_validate(s) for s in sources]
        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            abstained=result["abstained"],
            debug=result["debug"],
        )
