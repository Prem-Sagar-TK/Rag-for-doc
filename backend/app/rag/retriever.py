"""Retrieval with configurable relevance threshold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import get_settings
from app.vectorstore import get_vector_store


@dataclass
class RetrievalResult:
    query: str
    all_hits: list[dict[str, Any]] = field(default_factory=list)
    relevant_hits: list[dict[str, Any]] = field(default_factory=list)
    threshold: float = 0.0
    top_k: int = 0
    abstain: bool = False

    @property
    def max_score(self) -> float:
        if not self.all_hits:
            return 0.0
        return max(h["score"] for h in self.all_hits)


class Retriever:
    def __init__(self, store=None):
        self.store = store or get_vector_store()
        self.settings = get_settings()

    def retrieve(
        self,
        query: str,
        *,
        document_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> RetrievalResult:
        k = top_k if top_k is not None else self.settings.top_k
        thr = threshold if threshold is not None else self.settings.relevance_threshold

        hits = self.store.similarity_search(
            query,
            top_k=k,
            document_ids=document_ids,
        )
        relevant = [h for h in hits if h["score"] >= thr]
        return RetrievalResult(
            query=query,
            all_hits=hits,
            relevant_hits=relevant,
            threshold=thr,
            top_k=k,
            abstain=len(relevant) == 0,
        )
