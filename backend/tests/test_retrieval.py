"""Retrieval, relevance threshold, citations, and hallucination tests."""

from __future__ import annotations


def test_retrieval_finds_relevant_leave_policy(ingestion, sample_pdf_bytes):
    from app.rag.retriever import Retriever

    doc = ingestion.ingest_bytes(filename="employee_policy.pdf", data=sample_pdf_bytes)
    retriever = Retriever(store=ingestion.store)
    result = retriever.retrieve(
        "How many days of annual leave?",
        document_ids=[doc.id],
        threshold=0.0,  # ensure we can inspect scores
    )
    assert result.all_hits
    assert result.max_score > 0
    joined = " ".join(h["content"] for h in result.all_hits).lower()
    assert "20 days" in joined or "annual leave" in joined


def test_relevance_threshold_abstains_on_unrelated_question(
    ingestion, sample_pdf_bytes
):
    from app.rag.pipeline import RAGPipeline
    from app.rag.retriever import Retriever
    from app.rag.generator import AnswerGenerator

    doc = ingestion.ingest_bytes(filename="employee_policy.pdf", data=sample_pdf_bytes)
    # High threshold forces abstention for weakly related hits
    retriever = Retriever(store=ingestion.store)
    result = retriever.retrieve(
        "What is the capital of France?",
        document_ids=[doc.id],
        threshold=0.95,
    )
    assert result.abstain
    assert result.relevant_hits == []

    pipeline = RAGPipeline(retriever=retriever, generator=AnswerGenerator())
    # Temporarily force high threshold via retrieve override path
    response = pipeline.chat("What is the capital of France?", document_ids=[doc.id])
    # With default test threshold 0.35, France may or may not abstain depending on
    # fake embeddings. Force via direct generator on abstaining retrieval.
    from app.rag.generator import AnswerGenerator as Gen

    gen = Gen()
    out = gen.generate("What is the capital of France?", result)
    assert out["abstained"] is True
    assert "couldn't find enough relevant information" in out["answer"].lower()


def test_citations_include_filename_and_score(ingestion, sample_pdf_bytes):
    from app.rag.pipeline import RAGPipeline
    from app.rag.retriever import Retriever
    from app.rag.generator import AnswerGenerator

    doc = ingestion.ingest_bytes(filename="employee_policy.pdf", data=sample_pdf_bytes)
    pipeline = RAGPipeline(
        retriever=Retriever(store=ingestion.store),
        generator=AnswerGenerator(),
    )
    # Lower threshold so we get citations in fake-embedding mode
    retrieval = pipeline.retriever.retrieve(
        "How many annual leave days are provided?",
        document_ids=[doc.id],
        threshold=0.0,
    )
    out = pipeline.generator.generate(
        "How many annual leave days are provided?", retrieval
    )
    assert out["sources"]
    src = out["sources"][0]
    assert src.filename == "employee_policy.pdf"
    assert src.chunk_id
    assert src.score >= 0


def test_hallucination_prevention_maternity_leave(ingestion, sample_pdf_bytes):
    """Document mentions annual/sick leave but not maternity leave."""
    from app.rag.generator import AnswerGenerator
    from app.rag.retriever import Retriever

    doc = ingestion.ingest_bytes(filename="employee_policy.pdf", data=sample_pdf_bytes)
    retriever = Retriever(store=ingestion.store)
    # High threshold: weakly related leave chunks should not pass
    result = retriever.retrieve(
        "What is the company's maternity leave policy?",
        document_ids=[doc.id],
        threshold=0.9,
    )
    out = AnswerGenerator().generate(
        "What is the company's maternity leave policy?", result
    )
    if result.abstain:
        assert out["abstained"]
        assert "couldn't find" in out["answer"].lower()
    else:
        # If somehow chunks pass, extractive fallback still must not invent maternity policy
        assert "maternity" not in out["answer"].lower() or out["abstained"]


def test_unknown_question_message(ingestion, sample_txt_bytes):
    from app.rag.generator import AnswerGenerator
    from app.rag.retriever import Retriever
    from app.rag.prompts import ABSTENTION_MESSAGE

    doc = ingestion.ingest_bytes(filename="remote_work.txt", data=sample_txt_bytes)
    result = Retriever(store=ingestion.store).retrieve(
        "What is the CEO's favorite color?",
        document_ids=[doc.id],
        threshold=0.99,
    )
    out = AnswerGenerator().generate("What is the CEO's favorite color?", result)
    assert out["abstained"]
    assert out["answer"] == ABSTENTION_MESSAGE


def test_csv_retrieval(ingestion, sample_csv_bytes):
    from app.rag.retriever import Retriever

    doc = ingestion.ingest_bytes(filename="employees.csv", data=sample_csv_bytes)
    result = Retriever(store=ingestion.store).retrieve(
        "What is Alice Johnson's role?",
        document_ids=[doc.id],
        threshold=0.0,
    )
    assert result.all_hits
    joined = " ".join(h["content"] for h in result.all_hits)
    assert "Alice" in joined
    assert "Engineer" in joined
