import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  FileText,
  FileSpreadsheet,
  FileType2,
  Moon,
  Sun,
  Trash2,
  Upload,
  Send,
  Eraser,
  ChevronDown,
  ChevronRight,
  Loader2,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react'
import {
  chat,
  deleteDocument,
  getHealth,
  listDocuments,
  uploadDocument,
  type ChatDebug,
  type ChatResponse,
  type DocumentInfo,
  type HealthResponse,
  type SourceCitation,
} from './api'

type Role = 'user' | 'assistant'

interface Message {
  id: string
  role: Role
  content: string
  sources?: SourceCitation[]
  debug?: ChatDebug
  abstained?: boolean
  error?: boolean
}

function fileIcon(type: string) {
  if (type === 'pdf') return <FileText className="h-4 w-4" />
  if (type === 'csv') return <FileSpreadsheet className="h-4 w-4" />
  return <FileType2 className="h-4 w-4" />
}

function typeBadge(type: string) {
  const colors: Record<string, string> = {
    pdf: 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-200',
    csv: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200',
    txt: 'bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-200',
  }
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${colors[type] ?? 'bg-slate-100 text-slate-700'}`}
    >
      {type}
    </span>
  )
}

function statusBadge(status: string) {
  if (status === 'ready') {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-[var(--ok)]">
        <CheckCircle2 className="h-3.5 w-3.5" /> Ready
      </span>
    )
  }
  if (status === 'processing' || status === 'pending') {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-[var(--warn)]">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Processing
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs text-[var(--danger)]">
      <AlertCircle className="h-3.5 w-3.5" /> Error
    </span>
  )
}

function SourceList({ sources }: { sources: SourceCitation[] }) {
  const [openId, setOpenId] = useState<string | null>(null)
  if (!sources.length) return null
  return (
    <div className="mt-3 space-y-2">
      <div className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
        Sources
      </div>
      {sources.map((s) => {
        const key = `${s.chunk_id}-${s.score}`
        const open = openId === key
        return (
          <button
            key={key}
            type="button"
            onClick={() => setOpenId(open ? null : key)}
            className="w-full rounded-lg border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-left transition hover:border-[var(--accent)]"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="truncate text-sm font-medium">{s.filename}</div>
                <div className="mt-0.5 text-xs text-[var(--muted)]">
                  {s.page != null ? `Page ${s.page}` : s.row_number != null ? `Row ${s.row_number}` : '—'}
                  {' · '}
                  {s.chunk_id}
                  {' · '}
                  score {s.score.toFixed(2)}
                </div>
              </div>
              {open ? <ChevronDown className="h-4 w-4 shrink-0" /> : <ChevronRight className="h-4 w-4 shrink-0" />}
            </div>
            {open && s.preview ? (
              <p className="mt-2 text-xs leading-relaxed text-[var(--muted)]">{s.preview}</p>
            ) : null}
          </button>
        )
      })}
    </div>
  )
}

function WhyAnswer({ debug, abstained }: { debug?: ChatDebug; abstained?: boolean }) {
  const [open, setOpen] = useState(false)
  if (!debug) return null
  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 text-xs font-medium text-[var(--accent)]"
      >
        {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        Why this answer?
      </button>
      {open ? (
        <div className="mt-2 rounded-lg border border-[var(--border)] bg-[var(--bg)] p-3 text-xs leading-relaxed text-[var(--muted)]">
          {abstained ? (
            <p>
              No retrieved chunks met the relevance threshold ({debug.relevance_threshold}). The
              system abstained instead of guessing.
            </p>
          ) : (
            <p>
              Answer grounded in {debug.chunks_used} chunk(s) above threshold{' '}
              {debug.relevance_threshold}. Max similarity score: {debug.max_score.toFixed(2)}.
            </p>
          )}
        </div>
      ) : null}
    </div>
  )
}

function RagDebugPanel({ debug }: { debug?: ChatDebug }) {
  const [open, setOpen] = useState(true)
  if (!debug) return null
  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--panel)] shadow-[var(--shadow)]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
      >
        <span className="text-sm font-semibold">RAG Pipeline</span>
        {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
      </button>
      {open ? (
        <div className="space-y-3 border-t border-[var(--border)] px-4 py-3 text-sm">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
              Query
            </div>
            <p className="mt-1">&ldquo;{debug.query}&rdquo;</p>
          </div>
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
              Retrieved chunks
            </div>
            <ol className="mt-2 space-y-2">
              {debug.retrieved_chunks.map((c, i) => (
                <li
                  key={`${c.chunk_id}-${i}`}
                  className={`rounded-lg border px-3 py-2 text-xs ${
                    c.used
                      ? 'border-[var(--accent)] bg-[var(--accent-soft)]'
                      : 'border-[var(--border)] opacity-70'
                  }`}
                >
                  <div className="font-medium">
                    {i + 1}. {c.filename}
                    {c.page != null ? ` — page ${c.page}` : ''}
                    {c.row_number != null ? ` — row ${c.row_number}` : ''}
                    {' — '}
                    score {c.score.toFixed(2)}
                    {c.used ? ' (used)' : ' (filtered)'}
                  </div>
                  <p className="mt-1 text-[var(--muted)]">{c.preview}</p>
                </li>
              ))}
              {!debug.retrieved_chunks.length ? (
                <li className="text-xs text-[var(--muted)]">No chunks retrieved.</li>
              ) : null}
            </ol>
          </div>
          <div className="flex flex-wrap gap-4 text-xs text-[var(--muted)]">
            <span>
              Relevance threshold: <strong className="text-[var(--text)]">{debug.relevance_threshold}</strong>
            </span>
            <span>
              Chunks used: <strong className="text-[var(--text)]">{debug.chunks_used}</strong>
            </span>
            <span>
              Top K: <strong className="text-[var(--text)]">{debug.top_k}</strong>
            </span>
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default function App() {
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return false
    return (
      localStorage.getItem('theme') === 'dark' ||
      (!localStorage.getItem('theme') &&
        window.matchMedia('(prefers-color-scheme: dark)').matches)
    )
  })
  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [uploading, setUploading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [latestDebug, setLatestDebug] = useState<ChatDebug | undefined>()
  const bottomRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  const refresh = useCallback(async () => {
    try {
      const [d, h] = await Promise.all([listDocuments(), getHealth()])
      setDocs(Array.isArray(d) ? d : [])
      setHealth(h)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load')
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, asking])

  const readyCount = useMemo(
    () => (Array.isArray(docs) ? docs.filter((d) => d.status === 'ready').length : 0),
    [docs],
  )

  async function onUploadFiles(files: FileList | File[]) {
    const list = Array.from(files)
    if (!list.length) return
    setUploading(true)
    setError(null)
    try {
      for (const file of list) {
        await uploadDocument(file)
      }
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  async function onDelete(id: string) {
    try {
      await deleteDocument(id)
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed')
    }
  }

  async function onAsk(e?: React.FormEvent) {
    e?.preventDefault()
    const question = input.trim()
    if (!question || asking) return
    setInput('')
    setAsking(true)
    setError(null)
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: question,
    }
    setMessages((m) => [...m, userMsg])
    try {
      const res: ChatResponse = await chat(question)
      setLatestDebug(res.debug)
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: res.answer,
          sources: res.sources,
          debug: res.debug,
          abstained: res.abstained,
        },
      ])
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Chat failed'
      setError(msg)
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: msg,
          error: true,
        },
      ])
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-6 md:px-6">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--accent)]">
            Retrieval-Augmented Generation
          </p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight md:text-3xl">
            RAG Document Assistant
          </h1>
          <p className="mt-1 text-sm text-[var(--muted)]">
            Upload PDF, TXT, or CSV — ask grounded questions with citations and relevance gating.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {health ? (
            <div className="hidden rounded-lg border border-[var(--border)] bg-[var(--panel)] px-3 py-2 text-xs text-[var(--muted)] sm:block">
              threshold {health.relevance_threshold} · top_k {health.top_k} ·{' '}
              {health.openai_configured ? 'OpenAI on' : 'offline embeddings'}
              {health.vector_backend ? ` · ${health.vector_backend}` : ''}
            </div>
          ) : null}
          <button
            type="button"
            onClick={() => setDark((v) => !v)}
            className="rounded-lg border border-[var(--border)] bg-[var(--panel)] p-2 shadow-sm"
            aria-label="Toggle theme"
          >
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </header>

      {error ? (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-200">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      ) : null}

      <div className="grid flex-1 gap-5 lg:grid-cols-[340px_minmax(0,1fr)]">
        {/* Documents panel */}
        <aside className="flex flex-col gap-4">
          <section className="rounded-xl border border-[var(--border)] bg-[var(--panel)] p-4 shadow-[var(--shadow)]">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold">Documents</h2>
              <span className="text-xs text-[var(--muted)]">{readyCount} ready</span>
            </div>

            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDragOver(true)
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragOver(false)
                void onUploadFiles(e.dataTransfer.files)
              }}
              className={`rounded-xl border-2 border-dashed px-4 py-8 text-center transition ${
                dragOver
                  ? 'border-[var(--accent)] bg-[var(--accent-soft)]'
                  : 'border-[var(--border)]'
              }`}
            >
              <Upload className="mx-auto h-8 w-8 text-[var(--accent)]" />
              <p className="mt-2 text-sm font-medium">Drag & drop files</p>
              <p className="mt-1 text-xs text-[var(--muted)]">PDF, TXT, CSV</p>
              <button
                type="button"
                disabled={uploading}
                onClick={() => fileRef.current?.click()}
                className="mt-4 inline-flex items-center gap-2 rounded-lg bg-[var(--accent)] px-3 py-2 text-sm font-medium text-white disabled:opacity-60"
              >
                {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                Upload Document
              </button>
              <input
                ref={fileRef}
                type="file"
                accept=".pdf,.txt,.csv"
                multiple
                className="hidden"
                onChange={(e) => {
                  if (e.target.files) void onUploadFiles(e.target.files)
                  e.target.value = ''
                }}
              />
            </div>

            <ul className="mt-4 max-h-[420px] space-y-2 overflow-y-auto">
              {docs.map((doc) => (
                <li
                  key={doc.id}
                  className="flex items-start gap-2 rounded-lg border border-[var(--border)] px-3 py-2"
                >
                  <div className="mt-0.5 text-[var(--accent)]">{fileIcon(doc.file_type)}</div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium">{doc.filename}</span>
                      {typeBadge(doc.file_type)}
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-[var(--muted)]">
                      {statusBadge(doc.status)}
                      <span>{doc.chunk_count} chunks</span>
                    </div>
                    {doc.error_message ? (
                      <p className="mt-1 text-xs text-[var(--danger)]">{doc.error_message}</p>
                    ) : null}
                  </div>
                  <button
                    type="button"
                    onClick={() => void onDelete(doc.id)}
                    className="rounded p-1 text-[var(--muted)] hover:bg-rose-50 hover:text-[var(--danger)] dark:hover:bg-rose-950/40"
                    aria-label={`Delete ${doc.filename}`}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </li>
              ))}
              {!docs.length ? (
                <li className="py-6 text-center text-sm text-[var(--muted)]">
                  No documents yet. Upload a policy PDF or CSV to begin.
                </li>
              ) : null}
            </ul>
          </section>

          <RagDebugPanel debug={latestDebug} />
        </aside>

        {/* Chat panel */}
        <section className="flex min-h-[640px] flex-col rounded-xl border border-[var(--border)] bg-[var(--panel)] shadow-[var(--shadow)]">
          <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-3">
            <h2 className="text-sm font-semibold">Chat</h2>
            <button
              type="button"
              onClick={() => {
                setMessages([])
                setLatestDebug(undefined)
              }}
              className="inline-flex items-center gap-1 rounded-lg border border-[var(--border)] px-2.5 py-1.5 text-xs text-[var(--muted)] hover:text-[var(--text)]"
            >
              <Eraser className="h-3.5 w-3.5" />
              Clear conversation
            </button>
          </div>

          <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
            {!messages.length ? (
              <div className="flex h-full min-h-[320px] flex-col items-center justify-center text-center">
                <p className="text-sm font-medium">Ask a question about your documents</p>
                <p className="mt-1 max-w-md text-xs text-[var(--muted)]">
                  Answers are generated only from retrieved chunks that pass the relevance
                  threshold. Unrelated questions will abstain instead of hallucinating.
                </p>
              </div>
            ) : null}

            {messages.map((m) => (
              <div
                key={m.id}
                className={`max-w-[92%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  m.role === 'user'
                    ? 'ml-auto bg-[var(--accent)] text-white'
                    : m.error
                      ? 'border border-rose-300 bg-rose-50 text-rose-900 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-100'
                      : 'border border-[var(--border)] bg-[var(--bg)]'
                }`}
              >
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide opacity-70">
                  {m.role === 'user' ? 'User' : 'Assistant'}
                </div>
                {m.role === 'assistant' ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p>{m.content}</p>
                )}
                {m.role === 'assistant' && m.sources ? <SourceList sources={m.sources} /> : null}
                {m.role === 'assistant' ? (
                  <WhyAnswer debug={m.debug} abstained={m.abstained} />
                ) : null}
              </div>
            ))}

            {asking ? (
              <div className="inline-flex items-center gap-2 rounded-2xl border border-[var(--border)] bg-[var(--bg)] px-4 py-3 text-sm text-[var(--muted)]">
                <Loader2 className="h-4 w-4 animate-spin" />
                Retrieving & generating…
              </div>
            ) : null}
            <div ref={bottomRef} />
          </div>

          <form
            onSubmit={(e) => void onAsk(e)}
            className="border-t border-[var(--border)] p-3"
          >
            <div className="flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="What is the leave policy?"
                className="flex-1 rounded-xl border border-[var(--border)] bg-[var(--bg)] px-3 py-2.5 text-sm outline-none ring-[var(--accent)] focus:ring-2"
                disabled={asking}
              />
              <button
                type="submit"
                disabled={asking || !input.trim()}
                className="inline-flex items-center gap-2 rounded-xl bg-[var(--accent)] px-4 py-2.5 text-sm font-medium text-white disabled:opacity-50"
              >
                <Send className="h-4 w-4" />
                Send
              </button>
            </div>
          </form>
        </section>
      </div>
    </div>
  )
}
