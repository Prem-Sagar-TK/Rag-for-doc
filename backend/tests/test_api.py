"""API endpoint tests via FastAPI TestClient."""

from __future__ import annotations


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "relevance_threshold" in body
    assert "top_k" in body


def test_upload_list_delete_flow(client, sample_txt_bytes):
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("remote_work.txt", sample_txt_bytes, "text/plain")},
    )
    assert upload.status_code == 200
    doc = upload.json()
    assert doc["status"] == "ready"
    assert doc["chunk_count"] >= 1

    listed = client.get("/api/documents")
    assert listed.status_code == 200
    assert any(d["id"] == doc["id"] for d in listed.json())

    chunks = client.get(f"/api/documents/{doc['id']}/chunks")
    assert chunks.status_code == 200
    assert len(chunks.json()) >= 1
    meta = chunks.json()[0]["metadata"]
    assert meta["filename"] == "remote_work.txt"
    assert meta["file_type"] == "txt"

    deleted = client.delete(f"/api/documents/{doc['id']}")
    assert deleted.status_code == 200
    listed2 = client.get("/api/documents")
    assert all(d["id"] != doc["id"] for d in listed2.json())


def test_upload_pdf_and_csv(client, sample_pdf_bytes, sample_csv_bytes):
    pdf = client.post(
        "/api/documents/upload",
        files={"file": ("employee_policy.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert pdf.status_code == 200
    assert pdf.json()["file_type"] == "pdf"

    csv_res = client.post(
        "/api/documents/upload",
        files={"file": ("employees.csv", sample_csv_bytes, "text/csv")},
    )
    assert csv_res.status_code == 200
    assert csv_res.json()["file_type"] == "csv"
    assert csv_res.json()["chunk_count"] == 2


def test_chat_returns_debug_and_sources(client, sample_pdf_bytes):
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("employee_policy.pdf", sample_pdf_bytes, "application/pdf")},
    )
    doc_id = upload.json()["id"]
    chat = client.post(
        "/api/chat",
        json={"question": "How many days of annual leave?", "document_ids": [doc_id]},
    )
    assert chat.status_code == 200
    body = chat.json()
    assert "answer" in body
    assert "debug" in body
    assert "relevance_threshold" in body["debug"]
    assert "retrieved_chunks" in body["debug"]


def test_chat_abstain_high_threshold(client, sample_pdf_bytes, monkeypatch, tmp_env):
    from app.config import get_settings
    from app.vectorstore import reset_vector_store_singleton

    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.99")
    get_settings.cache_clear()
    reset_vector_store_singleton()

    upload = client.post(
        "/api/documents/upload",
        files={"file": ("employee_policy.pdf", sample_pdf_bytes, "application/pdf")},
    )
    doc_id = upload.json()["id"]
    chat = client.post(
        "/api/chat",
        json={"question": "What is the capital of France?", "document_ids": [doc_id]},
    )
    assert chat.status_code == 200
    body = chat.json()
    assert body["abstained"] is True
    assert "couldn't find enough relevant information" in body["answer"].lower()
    assert body["sources"] == []


def test_evaluation_endpoint(client, sample_pdf_bytes, sample_txt_bytes, sample_csv_bytes):
    client.post(
        "/api/documents/upload",
        files={"file": ("employee_policy.pdf", sample_pdf_bytes, "application/pdf")},
    )
    client.post(
        "/api/documents/upload",
        files={"file": ("remote_work.txt", sample_txt_bytes, "text/plain")},
    )
    client.post(
        "/api/documents/upload",
        files={"file": ("employees.csv", sample_csv_bytes, "text/csv")},
    )
    res = client.post("/api/evaluation/run")
    assert res.status_code == 200
    body = res.json()
    assert body["total_cases"] >= 1
    assert "retrieval_hit_rate" in body
    assert "abstention_rate" in body
    assert "cases" in body
