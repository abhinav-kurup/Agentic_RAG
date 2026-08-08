# Ponytail Agent Rules (Lazy Senior Developer Mindset)

You are operating under the **Ponytail** engineering philosophy. Your primary objective is writing minimal, robust, maintainable code without unnecessary bloat or over-engineering.

---

## The Decision Ladder

Before writing or modifying any code in this codebase, run your proposed solution through the following 7 steps in sequence:

1. **YAGNI (You Aren't Gonna Need It):** Does this feature, abstraction, or helper function actually need to exist? If not, skip it.
2. **Codebase Reuse:** Check existing modules (`agents/`, `core/`, `utils/`, `document_processing/`, `vectorstore/`) to see if a function, class, or data model already exists before creating a new one.
3. **Standard Library First:** Prefer standard library capabilities (`pathlib`, `json`, `typing`, `functools`, `itertools`, `asyncio`, `math`, `dataclasses`) over custom logic or new packages.
4. **Native Framework Primitives:** Use native features of installed frameworks (e.g. LangGraph state management, Streamlit state and widgets, Pydantic field validators) instead of building custom wrappers.
5. **Existing Installed Dependencies:** Leverage existing packages in `requirements.txt` (`pydantic`, `chromadb`, `sentence_transformers`, `rank_bm25`, `langgraph`, `streamlit`, `faster_whisper`) before suggesting or introducing any new third-party dependency.
6. **Simplicity & One-Liners:** Keep functions small, focused, legible, and simple. Avoid deep nesting and premature abstraction layers.
7. **Minimum Code Execution:** Write only the exact code required to fulfill the user's intent. Never sacrifice security, input validation, error handling, or data safety.

---

## Project-Specific Directives (`Documind`)

- **Agent Pipeline (`agents/`):** Avoid adding redundant agent steps or complex node chains to the LangGraph workflow unless specifically required.
- **Data Models:** Standardize on existing Pydantic schemas across agents rather than redefining parallel structures.
- **RAG & Retrieval:** Reuse the existing hybrid retrieval pipeline (`SingleHopRetrievalAgent` / `MultiHopRetrievalAgent`) and vectorstore helpers instead of writing ad-hoc retrieval code.
- **LLM Prompts:** Keep system prompts clean, concise, and focused to save token usage and decrease response latency.
