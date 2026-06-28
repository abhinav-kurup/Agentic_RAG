import streamlit as st
import os
import uuid
import hashlib
import logging
from typing import Optional
from langsmith import traceable, tracing_context



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    from document_processing.loader import PDFLoader
    from document_processing.chunking import DocumentChunker
    from vectorstore.chroma import VectorStoreManager
    from core.orchestrator import Orchestrator
    from core.config import Config
    from audit.logger import AuditLogger
except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure you are running from the correct directory.")
    st.stop()


@st.cache_resource
def get_stt_engine():
    from agents.stt import STTEngine

    return STTEngine()


@st.cache_resource
def get_tts_engine():
    from agents.tts import TTSEngine

    return TTSEngine()


def start_background_tts(query_id: str, text: str):
    import threading
    from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
    ctx = get_script_run_ctx()
    thread = threading.Thread(
        target=run_tts_thread,
        args=(query_id, text, ctx)
    )
    if ctx is not None:
        add_script_run_ctx(thread, ctx)
    thread.daemon = True
    thread.start()


def run_tts_thread(query_id: str, text: str, ctx):
    from streamlit.runtime.scriptrunner import RerunData
    try:
        logger.info("Background TTS started for query %s", query_id)
        audio_bytes = get_tts_engine().synthesize_to_wav_bytes(text)
        
        # Save to the specific message matching the query_id
        for msg in st.session_state.messages:
            if msg.get("query_id") == query_id:
                msg["audio"] = audio_bytes
                msg["audio_status"] = "ready"
                logger.info("Background TTS successfully completed for query %s", query_id)
                break
                
        # Trigger rerun on the script run context
        if ctx is not None:
            from streamlit.runtime import Runtime
            runtime = Runtime.instance()
            session_info = runtime._session_mgr.get_session_info(ctx.session_id)
            if session_info is not None:
                session_info.session.request_rerun(None)
    except Exception as e:
        logger.error("Background TTS failed for query %s: %s", query_id, e)
        for msg in st.session_state.messages:
            if msg.get("query_id") == query_id:
                msg["audio_status"] = "failed"
                break
        if ctx is not None:
            from streamlit.runtime import Runtime
            runtime = Runtime.instance()
            session_info = runtime._session_mgr.get_session_info(ctx.session_id)
            if session_info is not None:
                session_info.session.request_rerun(None)


def run_documind_query(
    prompt: str
) -> tuple[str, dict, str]:
    query_id = str(uuid.uuid4())
    try:
        result_state = st.session_state.orchestrator.run(
            prompt,
            query_id=query_id,
            audit_logger=st.session_state.audit_logger,
        )
    except Exception:
        logger.exception("Orchestrator failed for query: %s", prompt[:120])
        raise
    response = result_state.get("final_response") or (
        "I apologize, but I could not generate an answer. "
        "Please check the Audit Logs for more details."
    )
    st.session_state.audit_logger.log_query(query_id, result_state)
    return response, result_state, query_id


def handle_user_query(prompt: str, tts_enabled: bool = True):
    prompt = (prompt or "").strip()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            response, result_state, query_id = run_documind_query(prompt)
            
            # Start background TTS if enabled
            audio_status = "none"
            if tts_enabled and Config.ENABLE_TTS:
                audio_status = "processing"
                start_background_tts(query_id, response)
                
            # Append assistant message with processing status
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "citations": result_state.get("citations"),
                "query_id": query_id,
                "audio_status": audio_status,
                "audio": None
            })
            
            st.session_state.last_sources = {
                "audit_log": result_state.get("audit_log", []),
                "retrieved_docs": result_state.get("retrieved_docs", []),
            }
            st.rerun()
        except Exception as e:
            logger.exception("Query failed")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"An error occurred: {e}",
                "citations": None,
                "query_id": None,
                "audio_status": "none",
                "audio": None
            })
            st.rerun()


@traceable(name="Ingestion Pipeline")
def ingest_documents(uploaded_files, loader, chunker, vector_store, replace_existing, temp_dir):
    all_chunks = []
    existing_sources = vector_store.get_processed_documents()

    for uploaded_file in uploaded_files:
        st.write(f"Processing {uploaded_file.name}...")

        if uploaded_file.name in existing_sources:
            if replace_existing:
                removed = vector_store.delete_by_source(uploaded_file.name)
                st.write(
                    f"Replaced existing index for {uploaded_file.name} "
                    f"({removed} chunks removed)."
                )
            else:
                st.warning(
                    f"Skipped {uploaded_file.name}: already indexed. "
                    "Enable 'Replace existing documents' to re-ingest."
                )
                continue

        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            pages = loader.load(file_path)
            st.write(f"Loaded {len(pages)} pages from {uploaded_file.name}")

            doc_id = hashlib.sha256(
                uploaded_file.name.encode()
            ).hexdigest()[:32]
            chunks = chunker.split_documents(
                pages, doc_id, source=uploaded_file.name
            )
            for c in chunks:
                c["metadata"]["source"] = uploaded_file.name

            all_chunks.extend(chunks)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    if all_chunks:
        st.write(f"Embedding {len(all_chunks)} chunks...")
        vector_store.add_chunks(all_chunks)
        return True
    return False


st.set_page_config(page_title="DocuMind AI", layout="wide", page_icon="📄")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_voice_hash" not in st.session_state:
    st.session_state.last_voice_hash = None


if "orchestrator" not in st.session_state:
    try:
        st.session_state.vector_store = VectorStoreManager()
        st.session_state.orchestrator = Orchestrator(
            vector_store=st.session_state.vector_store
        )
        st.session_state.audit_logger = AuditLogger()
        st.session_state.loader = PDFLoader()
        st.session_state.chunker = DocumentChunker()
        logger.info("Orchestrator initialized")
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")

st.title("📄 DocuMind AI")
st.markdown("### Intelligent Document Analysis Platform")


tab_chat, tab_audit, tab_metrics = st.tabs(["💬 Chat", "📊 Audit Logs", "📈 Metrics"])


with st.sidebar:
    st.header("📤 Document Upload")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 1

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    replace_existing = st.checkbox(
        "Replace existing documents on re-upload",
        value=False,
        help="If the same filename is already indexed, delete the old version before ingesting.",
    )

    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            with st.status("Processing documents...", expanded=True) as status:
                temp_dir = "data/documents"
                os.makedirs(temp_dir, exist_ok=True)
                
                with tracing_context(parent=False):
                    success = ingest_documents(
                        uploaded_files=uploaded_files,
                        loader=st.session_state.loader,
                        chunker=st.session_state.chunker,
                        vector_store=st.session_state.vector_store,
                        replace_existing=replace_existing,
                        temp_dir=temp_dir
                    )
                
                if success:
                    status.update(label="Processing Complete!", state="complete", expanded=False)
                    st.success("Successfully processed documents into Knowledge Base.")
                    import time
                    time.sleep(1.2)
                    st.session_state.uploader_key += 1
                    st.rerun()
                else:
                    status.update(label="Processing Failed", state="error")

    st.divider()
    st.markdown("### Processed Documents")

    if "vector_store" in st.session_state:
        docs = st.session_state.vector_store.get_processed_documents()
        if docs:
            for doc in docs:
                st.markdown(f"- 📄 {doc}")
        else:
            st.caption("No documents processed yet.")

    st.divider()
    st.markdown("### Voice")

    # if Config.ENABLE_VOICE:
    #     st.checkbox(
    #         "Voice input (microphone)",
    #         value=True,
    #         key="use_voice_input",
    #         help="Record a question; it is transcribed and sent to DocuMind.",
    #     )
    # else:
    #     st.caption("Voice input disabled (ENABLE_VOICE=false).")

    if Config.ENABLE_TTS:
        st.checkbox(
            "Read answers aloud",
            value=True,
            key="use_tts_output",
            help="Synthesize assistant replies as audio in the browser.",
        )
    else:
        st.caption("TTS disabled (ENABLE_TTS=false).")

    st.divider()
    st.markdown("### ⚙️ System Controls")

    if st.button(
        "🗑️ Clear Database",
        help="Delete all documents from vector store",
        type="secondary",
        use_container_width=True,
    ):
        try:
            st.session_state.vector_store.clear_database()
            st.success("Database cleared successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear database: {e}")

    if st.button(
        "🔧 Repair BM25 Index",
        help="Rebuild keyword index from Chroma if indexes drifted",
        use_container_width=True,
    ):
        try:
            count = st.session_state.vector_store.rebuild_bm25_from_chroma()
            st.success(f"BM25 index repaired ({count} chunks).")
        except Exception as e:
            st.error(f"Failed to repair BM25 index: {e}")


with tab_chat:
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # show citations directly under assistant responses (if present)
                if message.get("role") == "assistant" and message.get("citations"):
                    try:
                        st.markdown("**Citations:**")
                        for c in message.get("citations", []):
                            st.markdown(f"- {c}")
                    except Exception:
                        # fall back to a simple representation if citations have unexpected structure
                        st.write(message.get("citations"))
                if message.get("role") == "assistant":
                    if message.get("audio_status") == "processing":
                        st.caption("🔊 *Synthesizing voice...*")
                    elif message.get("audio"):
                        st.audio(message["audio"], format="audio/wav")

    with chat_container:
        if "last_sources" in st.session_state and st.session_state.last_sources:
            with st.expander("📋 View Sources & Reasoning"):
                st.json(st.session_state.last_sources.get("audit_log", []))

                docs = st.session_state.last_sources.get("retrieved_docs", [])
                if docs:
                    st.write("**Retrieved Documents:**")
                    for i, doc in enumerate(docs):
                        filename = doc["metadata"].get("source", f"Document {i+1}")
                        page = doc["metadata"].get("page_number", "?")
                        st.caption(f"{filename} - Page {page}")
                        st.text(doc.get("content", "")[:500] + "...")

    voice_enabled = Config.ENABLE_VOICE 
    tts_enabled = Config.ENABLE_TTS and st.session_state.get("use_tts_output", True)

    if voice_enabled:
        voice_audio = st.audio_input("Ask by voice")
        if voice_audio is not None:
            audio_bytes = voice_audio.getvalue()
            audio_hash = hashlib.md5(audio_bytes).hexdigest()
            if audio_hash != st.session_state.last_voice_hash:
                st.session_state.last_voice_hash = audio_hash
                with st.spinner("Transcribing..."):
                    try:
                        transcript = get_stt_engine().transcribe_bytes(audio_bytes)
                    except Exception as e:
                        st.error(f"Speech recognition failed: {e}")
                        transcript = ""
                if transcript:
                    st.caption(f"Heard: _{transcript}_")
                    handle_user_query(transcript, tts_enabled=tts_enabled)
                else:
                    st.warning("Could not transcribe audio. Try again.")

    prompt = st.chat_input("Ask a question about your documents...")

    if prompt:
        handle_user_query(prompt, tts_enabled=tts_enabled)


with tab_audit:
    st.header("System Audit Trail")
    if st.button("Refresh Logs"):
        logs = st.session_state.audit_logger.get_logs()

        import json

        if not logs:
            st.info("No logs available.")
        else:
            st.success(f"Loaded {len(logs)} log entries")
            # Render each log as an expander with structured tables
            # Sort logs newest-first by timestamp (fallback to original order)
            def _log_key(l):
                return l.get("timestamp") or l.get("created_at") or ""

            sorted_logs = sorted(logs, key=_log_key, reverse=True)

            for log in sorted_logs:
                query_id = log.get("query_id") or log.get("id") or "-"
                timestamp = log.get("timestamp", log.get("created_at", "-"))
                query_text = log.get("query", "-")
                final_response = log.get("final_response")

                # Show timestamp first so the newest entries are easy to scan
                with st.expander(f"{timestamp} — {query_id}"):
                    st.markdown(f"**Query:** {query_text}")
                    st.markdown(f"**Final response:** {final_response}")

                    # Audit trail (list of step dicts)
                    audit_trail = log.get("audit_trail", [])
                    # If audit_trail was serialized to a string, try to parse it back
                    if isinstance(audit_trail, str):
                        try:
                            audit_trail_parsed = json.loads(audit_trail)
                        except Exception:
                            audit_trail_parsed = audit_trail
                    else:
                        audit_trail_parsed = audit_trail

                    if audit_trail_parsed:
                        st.markdown("**Audit Steps**")
                        # If it's a list of dicts, show as a table
                        if isinstance(audit_trail_parsed, list) and all(isinstance(x, dict) for x in audit_trail_parsed):
                            # normalize dicts to have the same columns
                            cols = []
                            for item in audit_trail_parsed:
                                for k in item.keys():
                                    # exclude 'query' because the full query is shown in the expander header
                                    if k == "query":
                                        continue
                                    if k not in cols:
                                        cols.append(k)

                            # Desired column order for readability (remove 'query' to avoid repetition;
                            # the full query is already shown above each expander)
                            desired_cols = ["step", "status", "retrieved_count", "reason", "error", "timestamp"]
                            # Build ordered column list: desired first (if present), then remaining
                            ordered_cols = [c for c in desired_cols if c in cols] + [c for c in cols if c not in desired_cols]

                            rows = []
                            for item in audit_trail_parsed:
                                row = {}
                                for c in ordered_cols:
                                    val = item.get(c, "")
                                    row[c] = "" if val is None else str(val)
                                if "error" in row and len(row["error"]) > 100:
                                    row["error"] = row["error"][:100] + "..."
                                rows.append(row)
                            st.table(rows)
                        else:
                            # Fallback to JSON view
                            st.json(audit_trail_parsed)
                    else:
                        st.caption("No audit steps recorded for this entry.")

                    # Retrieved documents (if present)
                    retrieved = log.get("retrieved_docs", [])
                    if retrieved:
                        st.markdown("**Retrieved Documents**")
                        doc_rows = []
                        for d in retrieved:
                            meta = d.get("metadata", {}) if isinstance(d, dict) else {}
                            doc_rows.append({
                                "source": meta.get("source", "-"),
                                "page": meta.get("page_number", "-"),
                                "score": d.get("score", "-") if isinstance(d, dict) else "-",
                                "chunk_index": meta.get("chunk_index", "-"),
                                "snippet": (d.get("content") or d.get("text") or "")[:200] if isinstance(d, dict) else "",
                            })
                        st.table(doc_rows)


with tab_metrics:
    st.header("RAG Evaluation Metrics")
    
    metrics_path = os.path.join("evaluation", "results", "my_metrics.json")
    if not os.path.exists(metrics_path):
        st.info(f"Metrics file not found at `{metrics_path}`. Please run evaluations first.")
    else:
        try:
            import json
            import pandas as pd
            
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # 1. Final Average Metrics
            st.subheader("Final Averages")
            overall = data.get("overall", {})
            if overall:
                # Filter out ndc metrics
                filtered_overall = {k: v for k, v in overall.items() if "ndc" not in k.lower()}
                
                # Show key metrics in metric cards
                col1, col2, col3, col4 = st.columns(4)
                cols = [col1, col2, col3, col4]
                col_idx = 0
                
                key_metrics = ["answer_correctness", "answer_relevancy", "faithfulness", "context_recall"]
                for km in key_metrics:
                    if km in filtered_overall:
                        val = filtered_overall[km]
                        display_name = km.replace("_", " ").title()
                        val_str = f"{val * 100:.2f}%" if isinstance(val, (int, float)) else str(val)
                        cols[col_idx % 4].metric(display_name, val_str)
                        col_idx += 1
                
                # List all averages in a detailed table
                avg_rows = []
                for m_name, m_val in filtered_overall.items():
                    display_name = m_name.replace("_", " ").title()
                    val_str = f"{m_val:.4f}" if isinstance(m_val, float) else str(m_val)
                    avg_rows.append({"Metric": display_name, "Average Score": val_str})
                    
                df_avg = pd.DataFrame(avg_rows)
                st.markdown("<br>", unsafe_allow_html=True)
                st.table(df_avg)
            else:
                st.write("No overall averages available.")
                
            st.markdown("---")
            
            # 2. Benchmark Query Results (per-item scores)
            st.subheader("Benchmark Query Results")
            per_item = data.get("per_item", [])
            if per_item:
                query_rows = []
                for item in per_item:
                    row = {
                        "Query ID": item.get("id"),
                    }
                    # Add metrics scores, filtering out ndc
                    scores = item.get("scores", {})
                    for m_name, m_val in scores.items():
                        if "ndc" in m_name.lower():
                            continue
                        col_name = m_name.replace("_", " ").title()
                        row[col_name] = round(m_val, 4) if isinstance(m_val, float) else m_val
                    query_rows.append(row)
                
                df_queries = pd.DataFrame(query_rows)
                st.dataframe(df_queries, width="stretch")
            else:
                st.write("No individual query scores available.")
                
        except Exception as e:
            st.error(f"Error loading metrics: {e}")

