"""Seed sample documents and run the evaluation suite.

Usage (from backend/):
  python -m app.evaluation.run_cli
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Prefer offline mode unless an API key is present
if not os.getenv("OPENAI_API_KEY"):
    os.environ.setdefault("USE_FAKE_EMBEDDINGS", "true")
    os.environ.setdefault("RELEVANCE_THRESHOLD", "0.35")


def main() -> None:
    from app.config import get_settings
    from app.evaluation.runner import EvaluationRunner
    from app.rag.ingestion import DocumentRegistry, IngestionService
    from app.rag.pipeline import RAGPipeline
    from app.vectorstore import get_vector_backend_name, get_vector_store, reset_vector_store_singleton

    get_settings.cache_clear()
    reset_vector_store_singleton()

    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    store = get_vector_store()
    store.reset()
    registry = DocumentRegistry()
    registry.clear()
    ingestion = IngestionService(store=store, registry=registry)

    root = Path(__file__).resolve().parents[3]
    sample_dir = root / "data" / "sample_docs"
    for name in [
        "employee_policy.txt",
        "employee_policy.pdf",
        "remote_work.txt",
        "employees.csv",
    ]:
        path = sample_dir / name
        if path.exists():
            doc = ingestion.ingest_bytes(filename=name, data=path.read_bytes())
            print(f"Ingested {name}: status={doc.status} chunks={doc.chunk_count}")

    summary = EvaluationRunner(RAGPipeline()).run()
    out = {
        "vector_backend": get_vector_backend_name(),
        "use_fake_embeddings": settings.use_fake_embeddings
        or not bool(settings.openai_api_key),
        "relevance_threshold": settings.relevance_threshold,
        "top_k": settings.top_k,
        "metrics": {
            "total_cases": summary.total_cases,
            "retrieval_hit_rate": summary.retrieval_hit_rate,
            "average_max_relevance": summary.average_max_relevance,
            "answer_correctness": summary.answer_correctness,
            "abstention_rate": summary.abstention_rate,
            "citation_correctness": summary.citation_correctness,
        },
        "cases": [c.model_dump(mode="json") for c in summary.cases],
    }

    results_path = root / "data" / "evaluation" / "latest_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["metrics"], indent=2))
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
