# Agentic RAG

**Retrieval is a tool the model calls — not a hardcoded graph node.**

Agentic RAG is a document Q&A platform where a ReAct-style controller decides *whether* to search, *what* to search, and *when* it has enough evidence. The loop is self-sustaining:

```
query → rewrite follow-up → greetings? → memory gate
                                      → ReAct agent ⇄ tools (search, list, read page, extract, calc)
                                      → ingest evidence → critic
                                      → synthesizer (citations)
```

Upload PDFs, ask questions in chat or by voice, and get citation-backed answers with a full audit trail of every hop.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-green.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid_Search-DC244C.svg)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project started as a fork of [DocuMind](https://github.com/abhinav-kurup/Documind) and replaced the fixed planner → single-hop / multi-hop graph with a tool-calling agent. Retrieval, page reads, table extraction, and arithmetic are native LangChain tools (`bind_tools` / `tool_calls`), executed by LangGraph `ToolNode` — not JSON scraped out of a prompt.

---

## Why this architecture

Classic RAG always retrieves, then answers. Multi-agent RAG often hard-codes *when* each specialist runs. Here the model is in the loop:

| Problem in static RAG | What Agentic RAG does |
|---|---|
| Every query hits the index, including greetings | A planner routes chitchat away from retrieval |
| One-shot search misses multi-hop facts | The agent can search again with a better query |
| Snippets are truncated | `read_pages` fetches a specific PDF page |
| Working set mixes unrelated follow-ups | Topic-shift clears working memory; evidence is session-scoped |
| The model hallucinates when evidence is thin | A critic grades working memory before synthesis |

**Stop conditions** (any one ends the loop):

- Max hops (`AGENT_MAX_HOPS`, default **5**)
- Two consecutive empty `search_docs` / `read_pages` calls
- Critic recommendation `answer` or `give_up`
- The model emits **no further tool calls**

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
  Streamlit UI ────►│              FastAPI backend                │
  (frontend/app.py) │  /chat  /documents  /voice  /audit  /health │
                    └──────────┬──────────────┬───────────────────┘
                               │              │
                               ▼              ▼
                     LangGraph orchestrator   Redis + RQ worker
                               │              │  (PDF ingest)
                               ▼              ▼
                     Qdrant hybrid index ◄────┘
                     (dense BGE-M3 + sparse BM25, server-side RRF)
```

### Query graph

```
START
  │
  ▼
Contextualizer ── rewrite follow-ups using chat history + rolling summary
  │
  ▼
Planner ── conversational ──────────────────────────────► END (no retrieval)
  │
  └── anything else
        │
        ▼
      Memory gate ── clear evidence on topic shift; maybe compress old turns
        │
        ▼
      ┌────────────────────────────────────────┐
      │  ReAct agent  ──tool_calls──► ToolNode │
      │       ▲                         │      │
      │       │                         ▼      │
      │     Critic ◄──────── Ingest observations│
      └────────────────────────────────────────┘
        │  (no tool calls / hop limit / critic says answer)
        ▼
      Prepare synthesis → Analysis agent (citations, confidence)
        │
        ▼
       END
```

### Tools the agent can call

| Tool | Purpose |
|---|---|
| `search_docs` | Hybrid search + Cohere rerank over uploaded PDFs; optional filename filter |
| `list_documents` | Indexed PDF names |
| `read_pages` | Fetch chunks for one filename + 1-based page when a hit is truncated |
| `extract_tables` | Pull tables/numbers from **working memory** (must search first) |
| `calculator` | Safe arithmetic on retrieved numbers |

### Three-layer memory

1. **Short-term** — LangGraph `MemorySaver` checkpoint per `session_id` (message transcript).
2. **Working** — `evidence_store`: only chunks the agent actually fetched. Deduped, score-ranked, capped (`EVIDENCE_MAX_CHUNKS`, default 10). Survives follow-ups; **cleared on topic shift**.
3. **Long-term** — Qdrant native hybrid collection. A rolling `summary` compresses older turns once the message window overflows (`MEMORY_MESSAGE_WINDOW`, default 16).

---

## Features

- **Agentic retrieval** — Native tool calling; the LLM decides the next hop.
- **Follow-up rewriting** — Pronouns and “that paper” become standalone queries from session history.
- **Hybrid search** — Dense `BAAI/bge-m3` + sparse `Qdrant/bm25` fused with Reciprocal Rank Fusion **inside Qdrant** (no local BM25 index).
- **Cohere rerank** — `rerank-v4.0-fast` re-scores candidates, then diversity selection (page/document penalties).
- **Layout-aware ingest** — LlamaParse or PyMuPDF; optional Gemini vision captions for figures (`ENABLE_IMAGE_PROCESSING`).
- **Async ingestion** — Upload returns a job id; a Redis/RQ worker chunks and indexes in the background.
- **Voice I/O** — Whisper STT (`faster-whisper`) and Piper TTS (local ONNX), plus a spoken intent router.
- **Citation-backed answers** — Analysis agent returns confidence and source pages.
- **Audit trail** — Every node writes JSONL (route, hops, tool names, critic verdict).
- **LangSmith** — `@traceable` on agent steps when tracing is enabled.
- **RAGAS evaluation** — Answer correctness, faithfulness, context recall, relevancy, plus retrieval IR metrics.

---

## Tech stack

| Layer | Choice |
|---|---|
| Workflow | LangGraph (`StateGraph` + `ToolNode` + `MemorySaver`) |
| Agent LLM | Configurable; default `groq/llama-3.3-70b-versatile` |
| Planner / voice router | Groq `llama-3.1-8b-instant` (defaults) |
| Extraction | Gemini `gemini-2.0-flash-lite` |
| Analysis | Gemini `gemini-2.0-flash` |
| Embeddings | `BAAI/bge-m3` (sentence-transformers / HuggingFace) |
| Sparse vectors | FastEmbed `Qdrant/bm25` |
| Reranker | Cohere `rerank-v4.0-fast` |
| Vector DB | Qdrant (HTTP, default `localhost:6333`) |
| Job queue | Redis + RQ |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| STT / TTS | faster-whisper / Piper |
| Observability | LangSmith |
| Evaluation | RAGAS |

LLM identifiers use a `provider/model` prefix: `groq/...`, `gemini/...`, or `ollama/...`. Unprefixed names go to Ollama.

---

## Prerequisites

- **Python 3.10+**
- **Docker** (recommended) *or* local **Qdrant** + **Redis**
- API keys:
  - [Groq](https://console.groq.com/)
  - [Google AI Studio](https://aistudio.google.com/) (Gemini)
  - [Cohere](https://dashboard.cohere.com/api-keys) (rerank)
- Optional: [LlamaCloud](https://cloud.llamaindex.ai/) if `PARSER_TYPE=llama_parse`
- Optional: [LangSmith](https://smith.langchain.com/) for traces
- Optional: NVIDIA GPU + CUDA 12.6 (the pinned `torch` wheel in `requirements.txt` is `cu126`; change it for CPU-only)

---

## Quick start (Docker Compose)

This is the easiest path: Qdrant, Redis, API, ingest worker, and UI come up together.

```bash
git clone https://github.com/abhinav-kurup/Agentic_RAG.git
cd Agentic_RAG

copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux

# fill in GROQ_API_KEY, GOOGLE_API_KEY, COHERE_API_KEY
docker compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| OpenAPI docs | http://localhost:8000/docs |
| Qdrant | http://localhost:6333 |
| Redis | localhost:6379 |

Compose injects `QDRANT_URL=http://qdrant:6333` and `REDIS_URL=redis://redis:6379/0` into the backend and worker.

---

## Local development (no Compose for the app)

You still need Qdrant and Redis running. Either start only those containers:

```bash
docker compose up qdrant redis
```

Or install them yourself. Then:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # then edit keys
```

**Default Redis URL in code is `redis://localhost:6380/0`.** If Redis is on 6379 (Compose and most installs), set:

```env
REDIS_URL=redis://localhost:6379/0
```

Run three processes from the repo root:

```bash
# 1. API
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 2. Ingest worker (required for PDF upload)
python workers/run_worker.py

# 3. UI
streamlit run frontend/app.py
# or: python app.py
```

Windows uses RQ `SimpleWorker` (no `os.fork`). Keep the worker in its own terminal.

---

## Environment

Copy [`.env.example`](.env.example) to `.env`. Important variables:

| Variable | Default | Role |
|---|---|---|
| `GROQ_API_KEY` | — | Planner, agent, some analysis paths |
| `GOOGLE_API_KEY` | — | Gemini extraction / analysis / optional vision |
| `COHERE_API_KEY` | — | Rerank |
| `PARSER_TYPE` | `pymupdf` | `pymupdf` or `llama_parse` |
| `LLAMA_CLOUD_API_KEY` | — | Required for LlamaParse |
| `QDRANT_URL` | `http://localhost:6333` | Vector DB |
| `QDRANT_COLLECTION_NAME` | `documind_collection` | Collection name |
| `REDIS_URL` | `redis://localhost:6380/0` | Job queue (override to `:6379` if needed) |
| `AGENT_MODEL` | `groq/llama-3.3-70b-versatile` | ReAct controller |
| `AGENT_MAX_HOPS` | `5` | Hard stop on the tool loop |
| `EVIDENCE_MAX_CHUNKS` | `10` | Working-memory cap |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Dense encoder |
| `COHERE_RERANK_MODEL` | `rerank-v4.0-fast` | Reranker |
| `ENABLE_VOICE` / `ENABLE_TTS` | `true` | Speech I/O |
| `ENABLE_IMAGE_PROCESSING` | `false` | Gemini captions for figures (slow / costly) |
| `DOCUMIND_API_URL` | `http://localhost:8000` | Frontend → API |
| `LANGCHAIN_TRACING_V2` | — | Set `true` + `LANGCHAIN_API_KEY` for LangSmith |

---

## Usage

### Documents

1. Sidebar → upload one or more **PDFs**.
2. Processing is **asynchronous**. Watch job progress in the UI (or `GET /documents/jobs/{job_id}`).
3. Already-indexed files are skipped. Delete a document from the sidebar before re-uploading to re-ingest.

### Questions

```
"What is the SAC algorithm used for in this paper?"     ← agent searches, maybe once
"Compare the architecture of model A and model B"       ← multiple search hops
"Extract the results table from chapter 4"              ← search + extract_tables
"What was the accuracy on page 12?"                     ← search, then read_pages if needed
"Hello, how are you?"                                   ← conversational, no retrieval
```

### UI tabs

- **Chat** — Answers, citations, voice orb.
- **Audit Logs** — Per-query steps: hops, tool names, critic verdict, retrieved count.
- **Metrics** — RAGAS / IR scores from `evaluation/results/my_metrics.json`.

---

## HTTP API

Base URL: `http://localhost:8000` (interactive docs at `/docs`).

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Qdrant + Redis status |
| `GET` | `/config/features` | Voice flags, embedding model |
| `POST` | `/chat/query` | `{ "query", "session_id" }` → answer + citations + audit |
| `POST` | `/documents/upload` | Multipart PDFs → `{ job_id }` |
| `GET` | `/documents` | Indexed filenames |
| `GET` | `/documents/jobs/{job_id}` | Ingest progress |
| `DELETE` | `/documents/{source}` | Remove one PDF from the index |
| `POST` | `/documents/clear` | Wipe collection |
| `POST` | `/transcribe` | Audio → transcript |
| `POST` | `/interpret` | Audio → intent + reconstructed query |
| `POST` | `/synthesize` | Text → speech |
| `GET` | `/audit/logs` | Recent audit JSONL |
| `GET` | `/metrics` | Evaluation JSON |

Keep `session_id` stable across turns so the checkpointer and working memory apply.

---

## Project structure

```
Agentic-RAG/
├── agents/
│   ├── react_agent.py      # AgenticController: memory gate, think, ingest, critic
│   ├── tools.py            # search_docs, list_documents, read_pages, extract_tables, calculator
│   ├── memory.py           # evidence merge, message window, rolling summary
│   ├── planner.py          # conversational vs document questions
│   ├── query_condenser.py  # follow-up → standalone query
│   ├── analysis.py         # final answer + citations
│   ├── retrieval_base.py   # hybrid retrieve, rerank, diversity select
│   ├── prompt.py           # versioned prompts
│   ├── stt.py / tts.py     # Whisper / Piper
│   └── voice_intent.py     # spoken command routing
├── backend/
│   ├── main.py             # FastAPI app
│   ├── api/routes/         # chat, documents, voice, audit, health
│   ├── services/           # query, ingestion, jobs, voice
│   └── queue/              # Redis + RQ enqueue / job store
├── core/
│   ├── orchestrator.py     # LangGraph wiring
│   ├── state.py            # AgentState (evidence_store, hop_count, critic, …)
│   ├── conditions.py       # route_query / after_agent
│   ├── config.py           # environment
│   └── llm.py              # Groq / Gemini / Ollama factory
├── document_processing/    # loader, layout parser, type-aware chunking
├── vectorstore/qdrant.py   # native hybrid collection
├── workers/                # RQ ingest worker
├── frontend/               # Streamlit UI + API client + voice orb
├── evaluation/             # benchmarks, RAGAS runner, saved metrics
├── audit/logger.py         # JSONL audit writer
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

Runtime data (`data/documents`, `data/logs`, `data/jobs`, Piper voices, audio cache) is gitignored.

---

## Evaluation

1. Produce predictions (see `evaluation/generate_predictions.py`).
2. Score against a JSONL benchmark:

```bash
python evaluation/runner.py `
  --benchmark evaluation/benchmarks.jsonl `
  --predictions path/to/predictions.jsonl `
  --output evaluation/results/metrics.json
```

Metrics include RAGAS (correctness, faithfulness, context recall, relevancy) and retrieval stats (hit rate, MRR, nDCG). Latest saved scores are shown in the **Metrics** tab.

---

## Design notes (interview-friendly)

- **Tool calling is native.** LangChain `bind_tools` + LangGraph `ToolNode`. The stop signal is “no `tool_calls` on the last AI message,” not a hand-rolled parser.
- **Working memory ≠ the vector store.** Qdrant holds everything; `evidence_store` holds only what this session’s agent fetched. Follow-ups reuse it; a topic shift wipes it.
- **Critic is structured output** (`sufficient`, `missing`, `recommendation` ∈ `answer | search_again | read_pages | give_up`), with a heuristic fallback if the structured call fails.
- **Ingest is out of the request path.** Uploads enqueue RQ jobs so chat stays responsive while PDFs are parsed and embedded.

---

## License

MIT — see [LICENSE](LICENSE).
