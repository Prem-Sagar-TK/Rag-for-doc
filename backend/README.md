# RAG Document Chatbot — Backend API

Backend service for the **RAG Document Chatbot**, engineered with **FastAPI**, **ChromaDB**, **SentenceTransformers**, and **OpenAI/LangChain**.

Features type-aware parsing for PDF, TXT, and CSV files, deterministic chunking with metadata preservation, vector retrieval with cosine similarity scoring, relevance thresholding to minimize hallucinations, and evaluation metrics.

---

## 🏛️ Directory Structure

```text
backend/
├── app/
│   ├── api/             # FastAPI endpoints (/api/documents, /api/chat, /api/evaluation)
│   ├── evaluation/      # RAG evaluation and answer relevance scoring metrics
│   ├── models/          # Pydantic data models for documents, chunks, and chat messages
│   ├── rag/             # Chunking, prompt engineering, citation extraction, and LLM chains
│   ├── vectorstore/     # ChromaDB vector store integration and embedding adapters
│   ├── config.py        # Application settings and environment configuration
│   └── main.py          # FastAPI application entry point
├── data/
│   ├── chroma/          # Persistent local Chroma vector database
│   ├── uploads/         # Uploaded raw documents
│   └── documents.json   # Indexed document metadata catalog
├── tests/               # Pytest suite for ingestion, retrieval, evaluation, and API
├── requirements.txt     # Python backend dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Set Up Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```env
OPENAI_API_KEY=your-api-key-here
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./data/chroma
```

### 4. Run the Server
```bash
python -m uvicorn app.main:app --reload --port 8000
```
Interactive docs: `http://localhost:8000/docs`

---

## 🧪 Testing

```bash
pytest tests/ -v
```

