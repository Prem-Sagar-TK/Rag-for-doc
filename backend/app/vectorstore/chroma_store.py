"""ChromaDB vector store wrapper (preferred when native deps are available)."""

from __future__ import annotations

from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.rag.chunking import DocumentChunk, normalize_metadata
from app.rag.embeddings import get_embeddings


class ChromaVectorStore:
    """Persistent Chroma collection for document chunks."""

    COLLECTION = "document_chunks"

    def __init__(self, persist_directory: Optional[str] = None):
        settings = get_settings()
        path = persist_directory or str(settings.chroma_dir)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embeddings = get_embeddings()

    @property
    def collection(self):
        return self._collection

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        ids = [c.metadata["chunk_id"] for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [normalize_metadata(c.metadata) for c in chunks]
        embeddings = self._embeddings.embed_documents(documents)
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def delete_document(self, document_id: str) -> None:
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception:
            pass

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []

        query_embedding = self._embeddings.embed_query(query)
        where = None
        if document_ids:
            if len(document_ids) == 1:
                where = {"document_id": document_ids[0]}
            else:
                where = {"document_id": {"$in": document_ids}}

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, max(self._collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        hits: list[dict[str, Any]] = []
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]

        for doc, meta, dist, cid in zip(docs, metas, distances, ids):
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            meta = dict(meta or {})
            meta.setdefault("chunk_id", cid)
            hits.append(
                {
                    "content": doc or "",
                    "metadata": meta,
                    "score": score,
                    "distance": float(dist),
                }
            )
        return hits

    def get_chunks_for_document(self, document_id: str) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        results = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        out: list[dict[str, Any]] = []
        for cid, doc, meta in zip(
            results.get("ids") or [],
            results.get("documents") or [],
            results.get("metadatas") or [],
        ):
            out.append(
                {
                    "chunk_id": cid,
                    "content": doc or "",
                    "metadata": dict(meta or {}),
                }
            )
        return out

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.COLLECTION)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
