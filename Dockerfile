FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-chroma.txt backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements-chroma.txt

COPY backend/app ./app
COPY data ./data

ENV UPLOAD_DIR=/app/data/uploads \
    CHROMA_DIR=/app/data/chroma \
    DOCUMENTS_DB=/app/data/documents.json \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
