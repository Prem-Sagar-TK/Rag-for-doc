"""Evaluation pipeline unit tests."""

from __future__ import annotations


def test_evaluation_runner_produces_metrics(
    ingestion, sample_pdf_bytes, sample_txt_bytes, sample_csv_bytes
):
    from app.evaluation.runner import EvaluationRunner
    from app.rag.generator import AnswerGenerator
    from app.rag.pipeline import RAGPipeline
    from app.rag.retriever import Retriever

    ingestion.ingest_bytes(filename="employee_policy.pdf", data=sample_pdf_bytes)
    ingestion.ingest_bytes(filename="remote_work.txt", data=sample_txt_bytes)
    ingestion.ingest_bytes(filename="employees.csv", data=sample_csv_bytes)

    pipeline = RAGPipeline(
        retriever=Retriever(store=ingestion.store),
        generator=AnswerGenerator(),
    )
    runner = EvaluationRunner(pipeline)
    dataset = [
        {
            "question": "How many days of annual leave does the company provide?",
            "expected_answer": "20 days",
            "expected_source": "employee_policy",
            "expected_abstain": False,
        },
        {
            "question": "What is the capital of France?",
            "expected_answer": None,
            "expected_source": None,
            "expected_abstain": True,
        },
    ]
    summary = runner.run(dataset=dataset)
    assert summary.total_cases == 2
    assert 0.0 <= summary.retrieval_hit_rate <= 1.0
    assert 0.0 <= summary.abstention_rate <= 1.0
    assert summary.cases


def test_answer_match_helper():
    from app.evaluation.runner import _answer_matches

    assert _answer_matches("The company provides 20 days of annual leave.", "20 days")
    assert not _answer_matches("No information available.", "20 days")
