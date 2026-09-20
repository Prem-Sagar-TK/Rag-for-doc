"""Lightweight persistent vector store (numpy cosine similarity).

Used when ChromaDB / chroma-hnswlib cannot be installed (e.g. Python 3.14
without a C++ toolchain). The Docker image uses ChromaDB on Python 3.12.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.config import get_settings
from app.rag.chunking import DocumentChunk, normalize_metadata
from app.rag.embeddings import get_embeddings


class NumpyVectorStore:
    """Disk-backed vector collection with the same surface as VectorStore."""

    def __init__(self, persist_directory: Optional[str] = None):
        settings = get_settings()
        self.path = Path(persist_directory or settings.chroma_dir) / "numpy_store"
        self.path.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.path / "meta.json"
        self._emb_path = self.path / "embeddings.npy"
        self._embeddings = get_embeddings()
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        if self._meta_path.exists():
            raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._ids = list(raw.get("ids") or [])
            self._documents = list(raw.get("documents") or [])
            self._metadatas = list(raw.get("metadatas") or [])
        if self._emb_path.exists() and self._ids:
            self._matrix = np.load(self._emb_path)
        else:
            self._matrix = None

    def _save(self) -> None:
        self._meta_path.write_text(
            json.dumps(
                {
                    "ids": self._ids,
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if self._matrix is not None and len(self._ids):
            np.save(self._emb_path, self._matrix)
        elif self._emb_path.exists():
            self._emb_path.unlink()

    @property
    def collection(self):
        return self

    def count(self) -> int:
        return len(self._ids)

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        ids = [c.metadata["chunk_id"] for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [normalize_metadata(c.metadata) for c in chunks]
        vectors = np.array(
            self._embeddings.embed_documents(documents), dtype=np.float32
        )
        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        if self._matrix is None or len(self._ids) == 0:
            self._matrix = vectors
            self._ids = ids
            self._documents = documents
            self._metadatas = metadatas
        else:
            self._matrix = np.vstack([self._matrix, vectors])
            self._ids.extend(ids)
            self._documents.extend(documents)
            self._metadatas.extend(metadatas)
        self._save()
        return len(chunks)

    def delete_document(self, document_id: str) -> None:
        keep = [
            i
            for i, meta in enumerate(self._metadatas)
            if meta.get("document_id") != document_id
        ]
        if len(keep) == len(self._ids):
            return
        self._ids = [self._ids[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        if keep and self._matrix is not None:
            self._matrix = self._matrix[keep]
        else:
            self._matrix = None
        self._save()

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        if not self._ids or self._matrix is None:
            return []

        q = np.array(self._embeddings.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q) or 1.0
        q = q / q_norm

        indices = list(range(len(self._ids)))
        if document_ids:
            allowed = set(document_ids)
            indices = [
                i for i in indices if self._metadatas[i].get("document_id") in allowed
            ]
        if not indices:
            return []

        subset = self._matrix[indices]
        scores = subset @ q  # cosine similarity
        order = np.argsort(-scores)[:top_k]

        hits: list[dict[str, Any]] = []
        for rank in order:
            i = indices[int(rank)]
            score = float(scores[int(rank)])
            # Clamp numerical noise
            score = max(0.0, min(1.0, score))
            meta = dict(self._metadatas[i])
            meta.setdefault("chunk_id", self._ids[i])
            hits.append(
                {
                    "content": self._documents[i],
                    "metadata": meta,
                    "score": score,
                    "distance": 1.0 - score,
                }
            )
        return hits

    def get_chunks_for_document(self, document_id: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cid, doc, meta in zip(self._ids, self._documents, self._metadatas):
            if meta.get("document_id") == document_id:
                out.append(
                    {
                        "chunk_id": cid,
                        "content": doc,
                        "metadata": dict(meta),
                    }
                )
        return out

    def reset(self) -> None:
        self._ids = []
        self._documents = []
        self._metadatas = []
        self._matrix = None
        self._save()
