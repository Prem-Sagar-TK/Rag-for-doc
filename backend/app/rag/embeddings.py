"""Embedding providers for RAG."""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicFakeEmbeddings(Embeddings):
    """Deterministic bag-of-tokens embeddings for offline tests.

    Shared tokens produce aligned dimensions so related queries retrieve
    matching chunks without calling an external API.
    """

    def __init__(self, size: int = 256):
        self.size = size

    def _tokens(self, text: str) -> list[str]:
        text = (text or "").lower()
        tokens = _TOKEN_RE.findall(text)
        # Add bigrams for short phrases like "annual leave"
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.size
        tokens = self._tokens(text)
        if not tokens:
            tokens = ["empty"]
        # TF weighting
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, tf in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            # Primary bucket
            idx = int.from_bytes(digest[:4], "little") % self.size
            weight = 1.0 + math.log(tf)
            vec[idx] += weight
            # Secondary bucket for smoother collisions
            idx2 = int.from_bytes(digest[4:8], "little") % self.size
            vec[idx2] += 0.5 * weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


def get_embeddings() -> Embeddings:
    settings = get_settings()
    if settings.use_fake_embeddings or not settings.openai_api_key:
        return DeterministicFakeEmbeddings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
