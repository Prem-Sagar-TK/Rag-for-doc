"""LLM answer generation grounded in retrieved context."""

from __future__ import annotations

import re
from typing import Any

from openai import OpenAI

from app.config import get_settings
from app.models.schemas import SourceCitation
from app.rag.prompts import (
    ABSTENTION_MESSAGE,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.rag.retriever import RetrievalResult

_STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "many",
    "much",
    "does",
    "do",
    "did",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "company",
    "companies",
    "policy",
    "policies",
    "employee",
    "employees",
}


def _preview(text: str, n: int = 160) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _content_tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if t not in _STOP and len(t) > 2
    }


def _token_in_context(token: str, context: str) -> bool:
    if token in context:
        return True
    stems = {token}
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            stems.add(token[: -len(suffix)])
    stems.add(token + "s")
    stems.add(token + "ed")
    stems.add(token + "ing")
    return any(s in context for s in stems if len(s) > 2)


def context_covers_question(question: str, hits: list[dict[str, Any]]) -> bool:
    """Require distinctive question tokens to appear in retrieved context.

    Prevents weakly similar 'leave' chunks from answering 'maternity leave'
    when the word maternity never appears in retrieved text.
    """
    q_tokens = _content_tokens(question)
    if not q_tokens:
        return True
    context = " ".join(h.get("content") or "" for h in hits).lower()
    return all(_token_in_context(t, context) for t in q_tokens)


def build_context(hits: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata") or {}
        header = (
            f"[{i}] source={meta.get('filename')} "
            f"page={meta.get('page_number')} "
            f"chunk_id={meta.get('chunk_id')} "
            f"row={meta.get('row_number')}"
        )
        parts.append(f"{header}\n{hit.get('content', '')}")
    return "\n\n".join(parts)


def citations_from_hits(hits: list[dict[str, Any]]) -> list[SourceCitation]:
    sources: list[SourceCitation] = []
    for hit in hits:
        meta = hit.get("metadata") or {}
        page = meta.get("page_number")
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError):
                page = None
        row = meta.get("row_number")
        if row is not None:
            try:
                row = int(row)
            except (TypeError, ValueError):
                row = None
        sources.append(
            SourceCitation(
                filename=str(meta.get("filename") or "unknown"),
                page=page,
                chunk_id=str(meta.get("chunk_id") or ""),
                score=round(float(hit.get("score", 0.0)), 4),
                row_number=row,
                preview=_preview(hit.get("content", "")),
            )
        )
    return sources


def debug_payload(retrieval: RetrievalResult, used_count: int) -> dict[str, Any]:
    retrieved = []
    used_ids = {h["metadata"].get("chunk_id") for h in retrieval.relevant_hits}
    for hit in retrieval.all_hits:
        meta = hit.get("metadata") or {}
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
        cid = meta.get("chunk_id")
        retrieved.append(
            {
                "filename": meta.get("filename"),
                "page": page,
                "chunk_id": cid,
                "score": round(float(hit.get("score", 0.0)), 4),
                "used": cid in used_ids,
                "preview": _preview(hit.get("content", "")),
                "row_number": row,
            }
        )
    return {
        "query": retrieval.query,
        "relevance_threshold": retrieval.threshold,
        "top_k": retrieval.top_k,
        "chunks_retrieved": len(retrieval.all_hits),
        "chunks_used": used_count,
        "max_score": round(retrieval.max_score, 4),
        "retrieved_chunks": retrieved,
    }


class AnswerGenerator:
    def __init__(self):
        self.settings = get_settings()

    def generate(self, question: str, retrieval: RetrievalResult) -> dict[str, Any]:
        if retrieval.abstain:
            return {
                "answer": ABSTENTION_MESSAGE,
                "sources": [],
                "abstained": True,
                "debug": debug_payload(retrieval, 0),
            }

        if not context_covers_question(question, retrieval.relevant_hits):
            return {
                "answer": ABSTENTION_MESSAGE,
                "sources": [],
                "abstained": True,
                "debug": {
                    **debug_payload(retrieval, 0),
                    "groundedness_failed": True,
                },
            }

        context = build_context(retrieval.relevant_hits)
        sources = citations_from_hits(retrieval.relevant_hits)
        debug = debug_payload(retrieval, len(retrieval.relevant_hits))

        if self.settings.use_fake_embeddings or not self.settings.openai_api_key:
            answer = self._extractive_answer(question, retrieval.relevant_hits)
            return {
                "answer": answer,
                "sources": sources,
                "abstained": False,
                "debug": debug,
            }

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(question, context),
                },
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
        abstained = self._looks_like_abstention(answer)
        return {
            "answer": answer,
            "sources": [] if abstained else sources,
            "abstained": abstained,
            "debug": debug,
        }

    @staticmethod
    def _looks_like_abstention(answer: str) -> bool:
        lowered = answer.lower()
        markers = [
            "couldn't find enough relevant",
            "could not find enough relevant",
            "couldn't find that information",
            "could not find that information",
            "not enough relevant information",
            "insufficient information in the uploaded",
        ]
        return any(m in lowered for m in markers)

    @staticmethod
    def _extractive_answer(question: str, hits: list[dict[str, Any]]) -> str:
        """Simple grounded fallback used in tests without OpenAI."""
        if not hits:
            return ABSTENTION_MESSAGE
        best = max(hits, key=lambda h: h.get("score", 0.0))
        return (best.get("content") or "").strip()
