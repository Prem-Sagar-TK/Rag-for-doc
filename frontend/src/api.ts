export type FileType = 'pdf' | 'txt' | 'csv'
export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'error'

export interface DocumentInfo {
  id: string
  filename: string
  file_type: FileType
  status: DocumentStatus
  chunk_count: number
  error_message?: string | null
  created_at: string
  size_bytes: number
}

export interface SourceCitation {
  filename: string
  page?: number | null
  chunk_id: string
  score: number
  row_number?: number | null
  preview?: string | null
}

export interface RetrievedChunkDebug {
  filename: string
  page?: number | null
  chunk_id: string
  score: number
  used: boolean
  preview: string
  row_number?: number | null
}

export interface ChatDebug {
  query: string
  relevance_threshold: number
  top_k: number
  chunks_retrieved: number
  chunks_used: number
  max_score: number
  retrieved_chunks: RetrievedChunkDebug[]
}

export interface ChatResponse {
  answer: string
  sources: SourceCitation[]
  abstained: boolean
  debug: ChatDebug
}

export interface HealthResponse {
  status: string
  openai_configured: boolean
  document_count: number
  relevance_threshold: number
  top_k: number
  vector_backend?: string
}

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? JSON.stringify(body)
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res.json() as Promise<T>
}

export async function getHealth(): Promise<HealthResponse> {
  return handle(await fetch(`${API_BASE}/api/health`))
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  return handle(await fetch(`${API_BASE}/api/documents`))
}

export async function uploadDocument(file: File): Promise<DocumentInfo> {
  const form = new FormData()
  form.append('file', file)
  return handle(
    await fetch(`${API_BASE}/api/documents/upload`, {
      method: 'POST',
      body: form,
    }),
  )
}

export async function deleteDocument(id: string): Promise<void> {
  await handle(await fetch(`${API_BASE}/api/documents/${id}`, { method: 'DELETE' }))
}

export async function chat(
  question: string,
  documentIds?: string[],
): Promise<ChatResponse> {
  return handle(
    await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        document_ids: documentIds?.length ? documentIds : null,
      }),
    }),
  )
}
