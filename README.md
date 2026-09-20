# RAG Document Chatbot

Production-style **Retrieval-Augmented Generation** app for PDF, TXT, and CSV files.

Users upload documents, the system performs **type-aware chunking → embeddings → vector retrieval → relevance thresholding → grounded generation → citations**, and refuses to answer when retrieved context is not relevant enough.

---

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────────┐
│   React UI   │────▶│                 FastAPI                      │
│  (Vite/TS)   │◀────│  /api/documents  /api/chat  /api/evaluation  │
└──────────────┘     └───────────────┬──────────────────────────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │         RAG Pipeline           │
                     │  ingestion → chunking          │
                     │  → embeddings → vector store   │
                     │  → threshold → LLM / extractive│
                     └───────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
        ChromaDB (Docker /     Numpy cosine store     OpenAI embeddings
        Python 3.12)           (local fallback)       + chat (optional)
```

### Type-aware ingestion

| Type | Pipeline |
|------|----------|
| **PDF** | PyMuPDF page extraction → page metadata preserved → recursive chunking → embed → store |
| **TXT** | UTF-8 decode → semantic/token-aware recursive chunking → embed → store |
| **CSV** | pandas/csv parse → row-level chunks with column names → embed → store |

Every chunk stores metadata: `document_id`, `filename`, `file_type`, `page_number`, `chunk_id` (+ `row_number`, `columns` for CSV).

### Relevance threshold (anti-hallucination)

After vector search (`TOP_K`), only chunks with similarity ≥ `RELEVANCE_THRESHOLD` are used.

If none pass:

> I couldn't find enough relevant information in the uploaded documents to answer that question.

An additional **groundedness check** ensures distinctive question tokens appear in the retrieved context (e.g. “maternity leave” will not be answered from a generic leave policy that never mentions maternity).

---

## RAG flow

```
User Query
  → Query Processing
  → Embedding
  → Vector Search (top_k)
  → Similarity Scores
  → Relevance Threshold filter
  → Groundedness check
  → Context Assembly
  → LLM (or extractive offline fallback)
  → Answer + Sources + Debug metadata
```

The chat API returns retrieval debug fields used by the frontend **RAG Pipeline** panel.

---

## How to run

### 1. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
# Optional (Python 3.11–3.12 + C++/wheels): pip install -r requirements-chroma.txt

copy .env.example .env   # Windows
# cp .env.example .env  # Unix
# Edit OPENAI_API_KEY
```

```bash
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies `/api` to the backend.

### 3. Docker Compose

```bash
cd rag-document-chatbot
cp backend/.env.example backend/.env   # set OPENAI_API_KEY
docker compose up --build
```

- API: http://localhost:8000  
- UI: http://localhost:5173  

Docker uses **Python 3.12 + ChromaDB**. Local installs on Python 3.14 without native Chroma wheels automatically fall back to the **numpy vector store** (same RAG interface).

### Sample documents

`data/sample_docs/` contains:

- `employee_policy.pdf` / `.txt`
- `remote_work.txt`
- `employees.csv`

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health + threshold / backend info |
| `POST` | `/api/documents/upload` | Upload PDF/TXT/CSV |
| `GET` | `/api/documents` | List documents |
| `DELETE` | `/api/documents/{id}` | Delete document + vectors |
| `GET` | `/api/documents/{id}/chunks` | Inspect stored chunks |
| `POST` | `/api/chat` | Ask a grounded question |
| `POST` | `/api/evaluation/run` | Run evaluation suite |

### Chat response shape

```json
{
  "answer": "The company provides 20 days of annual leave.",
  "sources": [
    {
      "filename": "employee_policy.pdf",
      "page": 1,
      "chunk_id": "chunk_…",
      "score": 0.89
    }
  ],
  "abstained": false,
  "debug": {
    "query": "…",
    "relevance_threshold": 0.72,
    "retrieved_chunks": [],
    "chunks_used": 3
  }
}
```

---

## Testing

```bash
cd backend
pytest -v
```

**Latest local run:** **20 passed**

Coverage includes:

1. PDF / TXT / CSV ingestion  
2. Metadata preservation & chunking  
3. Retrieval  
4. Relevance threshold abstention  
5. Citation generation  
6. Unknown / hallucination-tempting questions  
7. Evaluation pipeline  
8. FastAPI TestClient routes  

---

## Evaluation results

Produced by running:

```bash
cd backend
# Offline deterministic run (no API key)
set USE_FAKE_EMBEDDINGS=true
set RELEVANCE_THRESHOLD=0.30
set OPENAI_API_KEY=
python -m app.evaluation.run_cli
```

Results are written to `data/evaluation/latest_results.json`.

### Metrics from the latest actual run

| Metric | Value |
|--------|------:|
| Cases | 6 |
| Retrieval hit rate | **1.0** |
| Average max relevance | **0.3695** |
| Answer correctness | **1.0** |
| Abstention rate | **0.3333** (2/6 expected abstentions) |
| Citation correctness | **1.0** |

**Notes**

- These numbers were produced with **deterministic fake embeddings** + numpy store (no OpenAI calls).  
- With a real `OPENAI_API_KEY`, use `RELEVANCE_THRESHOLD≈0.72` and re-run evaluation; do not paste unverified OpenAI metrics into the README.  
- Full case-level output: `data/evaluation/latest_results.json`.

Dataset: `data/evaluation/dataset.json` (leave policy Q&A, CSV role lookup, remote work, maternity abstention, “capital of France” abstention).

---

## Frontend features

- Drag-and-drop upload with PDF / TXT / CSV badges  
- Document list, processing status, delete  
- Chat with Markdown answers  
- Expandable source citations + retrieval scores  
- “Why this answer?” explanation  
- Expandable **RAG Pipeline** debug panel  
- Clear conversation, loading/error states, dark/light mode  

---

## Configuration

| Env var | Default | Meaning |
|---------|---------|---------|
| `OPENAI_API_KEY` | — | Enables OpenAI embeddings + chat |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Generation model |
| `TOP_K` | `5` | Vector search breadth |
| `RELEVANCE_THRESHOLD` | `0.72` | Minimum similarity to use a chunk |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Text splitter settings |
| `USE_FAKE_EMBEDDINGS` | `false` | Force offline embeddings |

---

## Project structure

```
rag-document-chatbot/
├── backend/
│   ├── app/
│   │   ├── api/routes.py
│   │   ├── rag/          # ingestion, chunking, embeddings, retriever, generator, prompts
│   │   ├── evaluation/   # dataset runner + CLI
│   │   ├── models/
│   │   ├── vectorstore/  # Chroma + numpy fallback
│   │   ├── config.py
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── requirements-chroma.txt
├── frontend/             # React + Vite + Tailwind
├── data/sample_docs/
├── data/evaluation/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Known limitations

1. **ChromaDB on Python 3.14 / Windows** may require Visual C++ build tools (`chroma-hnswlib`). Without it, the app uses the numpy persistent store. Prefer **Python 3.12 + Docker** for Chroma.  
2. **Offline / fake embeddings** are deterministic bag-of-tokens vectors — useful for tests, not a substitute for OpenAI embeddings in production.  
3. **Without `OPENAI_API_KEY`**, answers are **extractive** (best matching chunk text), not LLM-summarized.  
4. **PDF quality** depends on text extractability (scanned/image PDFs need OCR — not included).  
5. **CSV chunking** is row-level; very wide tables may produce long chunks.  
6. **Relevance threshold** needs tuning per embedding model; 0.72 targets OpenAI cosine scores.  
7. Single-process document registry (JSON file) — fine for demo / single-node, not multi-replica production.  

---

## License

MIT — adapt freely for demos and learning.
