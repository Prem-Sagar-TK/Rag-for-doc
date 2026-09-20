"""Vector store factory: prefer ChromaDB, fall back to numpy store."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Union

from app.config import get_settings


class VectorStoreProtocol(Protocol):
    def add_chunks(self, chunks: Any) -> int: ...
    def delete_document(self, document_id: str) -> None: ...
    def similarity_search(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]: ...
    def get_chunks_for_document(self, document_id: str) -> list[dict[str, Any]]: ...
    def reset(self) -> None: ...


def _chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401
        import chromadb.config  # noqa: F401

        return True
    except Exception:
        return False


_store: Optional[Any] = None
_backend_name: str = "unknown"


def get_vector_backend_name() -> str:
    return _backend_name


def get_vector_store() -> Any:
    global _store, _backend_name
    if _store is not None:
        return _store

    settings = get_settings()
    force_numpy = getattr(settings, "use_fake_embeddings", False) and not _chroma_available()

    if _chroma_available() and not force_numpy:
        try:
            from app.vectorstore.chroma_store import ChromaVectorStore

            _store = ChromaVectorStore()
            _backend_name = "chromadb"
            return _store
        except Exception:
            pass

    from app.vectorstore.numpy_store import NumpyVectorStore

    _store = NumpyVectorStore()
    _backend_name = "numpy"
    return _store


def reset_vector_store_singleton() -> None:
    global _store, _backend_name
    _store = None
    _backend_name = "unknown"


# Backwards-compatible alias used in imports
VectorStore = Any  # type: ignore[misc]
