"""Evaluation framework for RAG quality metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.models.schemas import EvaluationCaseResult, EvaluationSummary, SourceCitation
from app.rag.pipeline import RAGPipeline


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _answer_matches(actual: str, expected: str) -> bool:
    """Loose containment match for short expected answer phrases."""
    a = _normalize(actual)
    e = _normalize(expected)
    if not e:
        return False
    if e in a:
        return True
    # Token overlap heuristic
    e_tokens = set(e.split())
    a_tokens = set(a.split())
    if not e_tokens:
        return False
    overlap = len(e_tokens & a_tokens) / len(e_tokens)
    return overlap >= 0.6


def _citation_matches(sources: list[SourceCitation], expected_source: str) -> bool:
    expected = (expected_source or "").lower()
    for s in sources:
        if expected in (s.filename or "").lower():
            return True
    return False


DEFAULT_DATASET: list[dict[str, Any]] = [
    {
        "id": "leave_days",
        "question": "How many days of annual leave does the company provide?",
        "expected_answer": "20 days",
        "expected_source": "employee_policy",
        "expected_abstain": False,
    },
    {
        "id": "sick_leave",
        "question": "How many sick leave days are employees entitled to?",
        "expected_answer": "10 days",
        "expected_source": "employee_policy",
        "expected_abstain": False,
    },
    {
        "id": "maternity_missing",
        "question": "What is the company's maternity leave policy?",
        "expected_answer": None,
        "expected_source": None,
        "expected_abstain": True,
    },
    {
        "id": "unrelated_france",
        "question": "What is the capital of France?",
        "expected_answer": None,
        "expected_source": None,
        "expected_abstain": True,
    },
    {
        "id": "csv_alice_role",
        "question": "What is Alice Johnson's role?",
        "expected_answer": "Engineer",
        "expected_source": "employees",
        "expected_abstain": False,
    },
    {
        "id": "txt_remote",
        "question": "What is the remote work policy?",
        "expected_answer": "three days",
        "expected_source": "remote_work",
        "expected_abstain": False,
    },
]


class EvaluationRunner:
    def __init__(self, pipeline: Optional[RAGPipeline] = None):
        self.pipeline = pipeline or RAGPipeline()

    def load_dataset(self, path: Optional[Path] = None) -> list[dict[str, Any]]:
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        # Prefer project data path
        candidates = [
            Path(__file__).resolve().parents[3] / "data" / "evaluation" / "dataset.json",
            Path("../data/evaluation/dataset.json"),
            Path("data/evaluation/dataset.json"),
        ]
        for c in candidates:
            if c.exists():
                return json.loads(c.read_text(encoding="utf-8"))
        return DEFAULT_DATASET

    def run(
        self,
        dataset: Optional[list[dict[str, Any]]] = None,
        document_ids: Optional[list[str]] = None,
    ) -> EvaluationSummary:
        cases_data = dataset if dataset is not None else self.load_dataset()
        results: list[EvaluationCaseResult] = []

        for case in cases_data:
            question = case["question"]
            response = self.pipeline.chat(question, document_ids=document_ids)
            expected_abstain = bool(case.get("expected_abstain", False))
            expected_answer = case.get("expected_answer")
            expected_source = case.get("expected_source")

            max_score = float((response.debug or {}).get("max_score") or 0.0)
            retrieval_hit = max_score > 0 and not (
                expected_abstain and response.abstained
            )
            # For abstention cases, a "hit" means we correctly found nothing relevant
            # or correctly abstained; for answer cases, hit means we retrieved something
            if expected_abstain:
                retrieval_hit = response.abstained or max_score < float(
                    (response.debug or {}).get("relevance_threshold") or 0.72
                )
            else:
                retrieval_hit = (not response.abstained) and len(response.sources) > 0

            answer_correct: Optional[bool] = None
            citation_correct: Optional[bool] = None

            if expected_abstain:
                answer_correct = response.abstained
            elif expected_answer:
                answer_correct = (not response.abstained) and _answer_matches(
                    response.answer, expected_answer
                )

            if expected_source and not expected_abstain:
                citation_correct = _citation_matches(response.sources, expected_source)

            results.append(
                EvaluationCaseResult(
                    question=question,
                    expected_answer=expected_answer,
                    expected_source=expected_source,
                    expected_abstain=expected_abstain,
                    actual_answer=response.answer,
                    sources=response.sources,
                    abstained=response.abstained,
                    retrieval_hit=retrieval_hit,
                    answer_correct=answer_correct,
                    citation_correct=citation_correct,
                    max_score=max_score,
                )
            )

        total = len(results) or 1
        hit_rate = sum(1 for r in results if r.retrieval_hit) / total
        avg_rel = sum(r.max_score for r in results) / total
        abstention_rate = sum(1 for r in results if r.abstained) / total

        graded_answers = [r.answer_correct for r in results if r.answer_correct is not None]
        answer_correctness = (
            sum(1 for v in graded_answers if v) / len(graded_answers)
            if graded_answers
            else None
        )

        graded_cites = [
            r.citation_correct for r in results if r.citation_correct is not None
        ]
        citation_correctness = (
            sum(1 for v in graded_cites if v) / len(graded_cites)
            if graded_cites
            else None
        )

        return EvaluationSummary(
            total_cases=len(results),
            retrieval_hit_rate=round(hit_rate, 4),
            average_max_relevance=round(avg_rel, 4),
            answer_correctness=(
                round(answer_correctness, 4) if answer_correctness is not None else None
            ),
            abstention_rate=round(abstention_rate, 4),
            citation_correctness=(
                round(citation_correctness, 4)
                if citation_correctness is not None
                else None
            ),
            cases=results,
        )
