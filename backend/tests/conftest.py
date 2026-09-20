"""Shared pytest fixtures for isolated RAG environments."""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

# Force offline / deterministic mode before importing app modules
os.environ["USE_FAKE_EMBEDDINGS"] = "true"
os.environ["OPENAI_API_KEY"] = ""
os.environ["RELEVANCE_THRESHOLD"] = "0.35"
os.environ["TOP_K"] = "5"
os.environ["CHUNK_SIZE"] = "400"
os.environ["CHUNK_OVERLAP"] = "50"


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    upload = tmp_path / "uploads"
    chroma = tmp_path / "chroma"
    docs_db = tmp_path / "documents.json"
    upload.mkdir()
    chroma.mkdir()

    monkeypatch.setenv("USE_FAKE_EMBEDDINGS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.35")
    monkeypatch.setenv("TOP_K", "5")
    monkeypatch.setenv("UPLOAD_DIR", str(upload))
    monkeypatch.setenv("CHROMA_DIR", str(chroma))
    monkeypatch.setenv("DOCUMENTS_DB", str(docs_db))

    from app.config import get_settings
    from app.vectorstore import reset_vector_store_singleton

    get_settings.cache_clear()
    reset_vector_store_singleton()
    yield tmp_path
    get_settings.cache_clear()
    reset_vector_store_singleton()


@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Acme Corp Employee Leave Policy\n\n"
        "The company provides 20 days of annual leave per calendar year.\n"
        "Employees are entitled to 10 days of paid sick leave each year.\n",
        fontsize=11,
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Holiday Closures\nThe company observes national public holidays.\n",
        fontsize=11,
    )
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def sample_txt_bytes() -> bytes:
    return (
        b"Remote Work Guidelines\n"
        b"Eligible employees may work remotely up to three days per week.\n"
        b"Core collaboration hours are 10:00 AM to 3:00 PM local time.\n"
    )


@pytest.fixture()
def sample_csv_bytes() -> bytes:
    return (
        b"name,role,department,location\n"
        b"Alice Johnson,Engineer,Platform,Austin\n"
        b"Bob Smith,Designer,Product,Remote\n"
    )


@pytest.fixture()
def client(tmp_env):
    from app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def ingestion(tmp_env):
    from app.rag.ingestion import DocumentRegistry, IngestionService
    from app.vectorstore import get_vector_store

    return IngestionService(store=get_vector_store(), registry=DocumentRegistry())
