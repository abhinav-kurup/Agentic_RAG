# 📄 DocuMind AI

**Intelligent Document Analysis Platform**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![LangSmith](https://img.shields.io/badge/LangSmith-Traced-orange.svg)](https://smith.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

DocuMind AI is a multi-agent RAG (Retrieval-Augmented Generation) platform for intelligent conversation with PDF documents. It classifies every query, routes it through a specialised retrieval pipeline, and returns citation-backed answers with a full audit trail.

---

## Architecture

```
START
  │
  ▼
Planner  ──────────────────────────────► Conversational
  │
  ├── single_hop ──► SingleHopRetrievalAgent
  │                        │
  └── multi_hop  ──► MultiHopRetrievalAgent
                           │
                           ▼
                       Extraction
                           │
                           ▼
                        Analysis ◄──── Tools (calculator)
                           │
                          END
```

### Retrieval Pipeline

**Single-hop:** One round of hybrid search → cross-encoder reranking → diversity selection.

**Multi-hop:** Query decomposed into subqueries → parallel hybrid search per subquery → Reciprocal Rank Fusion (RRF) → cross-encoder reranking → diversity selection + coverage validation.

---

## Features

- **Query planning** — Planner classifies queries as `conversational`, `single_hop`, or `multi_hop` and decomposes multi-hop queries into independent subqueries.
- **Hybrid retrieval** — BM25 keyword search fused with dense vector search (BAAI/bge-m3 embeddings).
- **Cross-encoder reranking** — BAAI/bge-reranker-v2-m3 re-scores candidates against the original query.
- **Diversity selection** — Greedy chunk selection with page/document repetition penalties and subquery coverage bonuses.
- **Coverage validation** — Guarantees every subquery has at least one supporting chunk before passing context to the LLM.
- **PydanticAI analysis** — Structured answer generation with confidence score and citations.
- **Extraction agent** — On-demand structured extraction for queries containing "extract" or "table".
- **Voice I/O** — Speech-to-text via Whisper (`faster-whisper`), text-to-speech via SpeechT5 (local inference).
- **LangSmith tracing** — Every agent step decorated with `@traceable` for full pipeline observability.
- **RAGAS evaluation** — Automated benchmark evaluation with answer correctness, faithfulness, context recall, and relevancy metrics.
- **Audit logs** — Per-query step trail (agent, status, subqueries, retrieved count) stored as JSONL.

---

## Tech Stack

| Component | Technology |
|---|---|
| Workflow engine | LangGraph |
| LLM — Planner | Groq `llama-3.1-8b-instant` |
| LLM — Retrieval decomposer | Groq `llama-3.1-8b-instant` |
| LLM — Extraction | Gemini `gemini-2.0-flash-lite` |
| LLM — Analysis | Groq `llama-3.3-70b-versatile` (PydanticAI) |
| Embeddings | `BAAI/bge-m3` (sentence-transformers) |
| Reranker | `BAAI/bge-reranker-v2-m3` (CrossEncoder) |
| Vector store | ChromaDB |
| Keyword search | BM25 (rank-bm25) |
| STT | Whisper via faster-whisper |
| TTS | microsoft/speecht5_tts + speecht5_hifigan |
| Observability | LangSmith |
| Evaluation | RAGAS |
| UI | Streamlit |

---

## Quick Start

### Prerequisites

- Python 3.9+
- Groq API key ([get one free](https://console.groq.com/))
- Google API key for Gemini ([AI Studio](https://aistudio.google.com/))

### Installation

```bash
git clone https://github.com/yourusername/Documind.git
cd Documind
pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env` and fill in:

```env
# LLM providers
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

# Per-agent model overrides (optional)
PLANNER_MODEL=groq/llama-3.1-8b-instant
RETRIEVAL_MODEL=groq/llama-3.1-8b-instant
EXTRACTION_MODEL=gemini/gemini-2.0-flash-lite
ANALYSIS_MODEL=groq/llama-3.3-70b-versatile

# Retrieval
EMBEDDING_MODEL=BAAI/bge-m3
CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3
USE_CROSS_ENCODER=true

# Vector store
CHROMA_DB_DIR=data/chroma

# Voice
ENABLE_VOICE=true
ENABLE_TTS=true
WHISPER_MODEL=base
TTS_SPEAKER_INDEX=7306

# LangSmith (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=documind
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Usage

### Uploading Documents

1. Open the sidebar → **Upload PDFs**
2. Select one or more PDF files
3. Click **Process Documents**

Already-indexed files are skipped. Delete a document from the sidebar list before re-uploading to re-ingest it.

### Asking Questions

```
"What is the SAC algorithm used for in this paper?"        ← single_hop
"Compare the architecture of model A and model B"          ← multi_hop
"Extract the results table from chapter 4"                 ← triggers Extraction agent
"Hello, how are you?"                                      ← conversational (no retrieval)
```

### Tabs

- **💬 Chat** — Main conversation interface with citations and voice I/O.
- **📊 Audit Logs** — Per-query execution trail showing every agent step, route, subqueries, and retrieved count.
- **📈 Metrics** — RAGAS benchmark scores loaded from `evaluation/results/my_metrics.json`.

---

## Project Structure

```
Documind/
├── agents/
│   ├── planner.py          # Query classification + subquery decomposition
│   ├── retrieval.py        # BaseRetrievalAgent, SingleHopRetrievalAgent, MultiHopRetrievalAgent
│   ├── extraction.py       # Structured data extraction (on-demand)
│   ├── analysis2.py        # PydanticAI answer synthesis with citations
│   ├── prompt.py           # Centralised prompt definitions
│   ├── stt.py              # Speech-to-text (Whisper)
│   └── tts.py              # Text-to-speech (SpeechT5)
├── audit/
│   └── logger.py           # JSONL audit trail writer
├── core/
│   ├── config.py           # Environment-based config
│   ├── conditions.py       # LangGraph routing conditions
│   ├── llm.py              # LLM factory (Groq, Gemini, Ollama)
│   ├── orchestrator.py     # LangGraph graph definition
│   └── state.py            # AgentState TypedDict
├── document_processing/
│   ├── loader.py           # PDF text extraction (PyMuPDF)
│   └── chunking.py         # Document chunking
├── evaluation/
│   ├── benchmarks.jsonl    # Ground-truth QA pairs
│   ├── generate_predictions.py
│   ├── metrics.py          # RAGAS metric wrappers
│   ├── runner.py           # Evaluation orchestration
│   └── results/            # Saved metric outputs
├── utils/
│   └── helpers.py          # Shared helpers (log_agent_step, dump_agent_state)
├── vectorstore/
│   └── chroma.py           # ChromaDB + BM25 hybrid search
├── data/
│   ├── chroma/             # Vector database (auto-created)
│   ├── documents/          # Uploaded PDFs (auto-created)
│   └── logs/               # Audit JSONL files (auto-created)
├── app.py                  # Streamlit application
└── requirements.txt
```

---

## Evaluation

Run the RAGAS benchmark suite:

```bash
python evaluation/runner.py
```

Results are saved to `evaluation/results/my_metrics.json` and displayed in the **📈 Metrics** tab.

---

## License

MIT License — see [LICENSE](LICENSE).
