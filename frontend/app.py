import json
import logging
import sys
import time
import uuid
from pathlib import Path

# Streamlit adds frontend/ to sys.path; repo root is needed for `core` / `frontend`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from core.config import Config
from frontend import api_client
from frontend.api_client import APIError
from frontend.voice_orb import voice_orb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Agentic RAG", layout="wide", page_icon="📄")

st.markdown(
    """
    <style>
    iframe[title="documind_voice_orb"],
    iframe[title*="documind_voice_orb"] {
      position: fixed !important;
      bottom: 0.85rem !important;
      right: 0.45rem !important;
      width: 240px !important;
      height: 300px !important;
      z-index: 1000 !important;
      background: transparent !important;
      border: none !important;
      overflow: visible !important;
    }
    div[data-testid="stIFrame"]:has(iframe[title*="voice_orb"]),
    div[data-testid="stElementContainer"]:has(iframe[title*="voice_orb"]),
    div[data-testid="element-container"]:has(iframe[title*="voice_orb"]),
    .stCustomComponentV1:has(iframe[title*="voice_orb"]) {
      position: fixed !important;
      bottom: 0.85rem !important;
      right: 0.45rem !important;
      width: 240px !important;
      height: 300px !important;
      z-index: 1000 !important;
      background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def ensure_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def check_backend():
    now = time.time()
    cached = st.session_state.get("_health_cache")
    if cached and now - cached["t"] < 20:
        return cached["ok"], cached["info"]
    try:
        health = api_client.health_check()
        ok = health.get("status") == "ok"
        st.session_state._health_cache = {"t": now, "ok": ok, "info": health}
        return ok, health
    except Exception as e:
        info = {"error": str(e)}
        st.session_state._health_cache = {"t": now, "ok": False, "info": info}
        return False, info


def cached_documents(force: bool = False):
    now = time.time()
    cached = st.session_state.get("_docs_cache")
    if not force and cached and now - cached["t"] < 15:
        return cached["docs"]
    docs = api_client.list_documents()
    st.session_state._docs_cache = {"t": now, "docs": docs}
    return docs


def _file_status_icon(status: str) -> str:
    return {
        "pending": "⏳",
        "processing": "🔄",
        "done": "✅",
        "failed": "❌",
        "skipped": "⏭️",
    }.get(status, "•")


def render_ingestion_progress(job: dict) -> None:
    """Render structured per-PDF progress from job progress_state."""
    state = job.get("progress_state") or {}
    status = job.get("status", "")
    percent = state.get("percent", 0)
    stage_label = state.get("stage_label") or status
    current_file = state.get("current_file")
    total = state.get("total_files", 0)
    completed = state.get("completed_files", 0)

    st.progress(percent / 100.0 if percent else 0.0, text=f"{percent}% — {stage_label}")

    if total:
        st.caption(f"Documents: {completed}/{total} complete")
    if current_file and status in ("queued", "processing", "pending"):
        st.markdown(f"**Current file:** `{current_file}`")
    elif status == "queued":
        st.caption("Waiting for ingest worker to pick up the job...")

    files = state.get("files") or []
    if files:
        st.markdown("**Files**")
        for f in files:
            icon = _file_status_icon(f.get("status", "pending"))
            name = f.get("name", "?")
            stage = f.get("stage", "pending")
            label = STAGE_LABELS.get(stage, stage) if stage else ""
            chunks = f.get("chunks")
            extra = f" — {chunks} chunks" if chunks is not None else ""
            err = f.get("error")
            err_txt = f" — _{err}_" if err else ""
            st.markdown(f"{icon} `{name}` · {label}{extra}{err_txt}")

    log = job.get("progress") or []
    if log:
        with st.expander("Detailed log", expanded=False):
            for line in log[-20:]:
                st.text(line)


STAGE_LABELS = {
    "pending": "Waiting",
    "replacing": "Replacing index",
    "parsing": "LlamaParse",
    "parsed": "Parsed",
    "chunking": "Chunking",
    "chunked": "Chunked",
    "embedding": "Embedding",
    "indexed": "Done",
    "failed": "Failed",
    "skipped": "Skipped",
    "queued": "Queued",
    "processing": "Processing",
    "completed": "Completed",
}


def poll_active_ingestion_job() -> None:
    job_id = st.session_state.get("active_ingest_job_id")
    if not job_id:
        return
    _ingest_poll_fragment()


@st.fragment(run_every=2)
def _ingest_poll_fragment() -> None:
    job_id = st.session_state.get("active_ingest_job_id")
    if not job_id:
        return

    try:
        job = api_client.get_document_job(job_id)
    except APIError as e:
        st.error(f"Could not fetch job status: {e}")
        st.session_state.active_ingest_job_id = None
        return

    job_status = job.get("status", "")

    with st.status("Processing documents...", expanded=True) as status_widget:
        render_ingestion_progress(job)

        if job_status == "completed":
            result = job.get("result") or {}
            st.success(
                f"Processed {len(result.get('processed', []))} file(s), "
                f"{result.get('chunks_added', 0)} chunks added."
            )
            for s in result.get("replaced") or []:
                st.info(f"Re-indexed: {s}")
            for err in result.get("errors") or []:
                st.error(f"{err['file']}: {err['error']}")
            status_widget.update(label="Processing complete!", state="complete", expanded=False)
            st.session_state.active_ingest_job_id = None
            st.session_state.uploader_key += 1
            st.session_state.pop("_docs_cache", None)
            st.rerun()
            return

        if job_status == "failed":
            st.error(job.get("error", "Ingestion failed"))
            status_widget.update(label="Processing failed", state="error")
            st.session_state.active_ingest_job_id = None
            return


def _trim_stored_audio(keep: int = 3) -> None:
    msgs = st.session_state.messages
    with_audio = [i for i, m in enumerate(msgs) if m.get("audio")]
    for i in with_audio[:-keep]:
        msgs[i]["audio"] = None


def poll_tts_for_message(query_id: str, job_id: str):
    try:
        job = api_client.get_tts_job(job_id)
        if job.get("status") == "completed":
            audio = api_client.download_tts_audio(job_id)
            for msg in st.session_state.messages:
                if msg.get("tts_job_id") == job_id:
                    msg["audio"] = audio
                    msg["audio_status"] = "ready"
                    break
            _trim_stored_audio()
        elif job.get("status") == "failed":
            for msg in st.session_state.messages:
                if msg.get("tts_job_id") == job_id:
                    msg["audio_status"] = "failed"
                    break
    except Exception as e:
        logger.error("TTS poll failed: %s", e)


def last_assistant_message():
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
            return msg
    return None


def _render_answer_audio(message: dict) -> None:
    autoplay = bool(message.pop("autoplay_once", False))
    try:
        st.audio(message["audio"], format="audio/wav", autoplay=autoplay)
    except TypeError:
        st.audio(message["audio"], format="audio/wav")


def start_tts_for_message(message: dict) -> None:
    if not Config.ENABLE_TTS:
        return
    text = (message.get("content") or "").strip()
    if not text:
        return
    try:
        tts_job = api_client.start_tts(text, query_id=message.get("query_id"))
        message["tts_job_id"] = tts_job.get("job_id")
        message["audio_status"] = "processing"
        message["audio"] = None
        message["autoplay_once"] = True
    except Exception as e:
        logger.error("Failed to start TTS: %s", e)
        message["audio_status"] = "failed"


def handle_user_query(prompt: str, tts_enabled: bool = False):
    prompt = (prompt or "").strip()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_query = prompt
    st.session_state.pending_tts = tts_enabled
    st.rerun()


def process_pending_query() -> None:
    prompt = st.session_state.pop("pending_query", None)
    if not prompt:
        return
    tts_enabled = st.session_state.pop("pending_tts", False)

    with st.spinner("Thinking..."):
        try:
            result = api_client.chat_query(prompt, st.session_state.session_id)
            response = result.get("response", "")
            query_id = result.get("query_id")

            audio_status = "none"
            tts_job_id = None
            if tts_enabled and Config.ENABLE_TTS:
                try:
                    tts_job = api_client.start_tts(response, query_id=query_id)
                    tts_job_id = tts_job.get("job_id")
                    audio_status = "processing"
                except Exception as e:
                    logger.error("Failed to start TTS: %s", e)

            assistant_msg = {
                "role": "assistant",
                "content": response,
                "citations": result.get("citations"),
                "query_id": query_id,
                "audio_status": audio_status,
                "tts_job_id": tts_job_id,
                "audio": None,
            }
            if audio_status == "processing":
                assistant_msg["autoplay_once"] = True
            st.session_state.messages.append(assistant_msg)

            st.session_state.last_sources = {
                "audit_log": result.get("audit_log", []),
                "retrieved_docs": result.get("retrieved_docs", []),
            }
            st.session_state.voice_pending_read = response
        except APIError as e:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"An error occurred: {e}",
                    "citations": None,
                    "query_id": None,
                    "audio_status": "none",
                    "audio": None,
                }
            )
    st.rerun()


def consume_orb_event(event) -> bool:
    if not event or not isinstance(event, dict):
        return False
    eid = str(event.get("id") or "")
    if not eid or eid == st.session_state.get("last_orb_id"):
        return False
    st.session_state.last_orb_id = eid
    etype = event.get("type")

    if etype == "error":
        st.session_state.voice_error = event.get("message") or "Voice failed"
        return True

    if etype in ("confirm_done", "quiet", "hangup"):
        st.session_state.voice_pending_read = ""
        st.session_state.voice_error = ""
        return True

    if etype != "turn":
        return False

    reconstructed = (event.get("reconstructed_query") or event.get("transcript") or "").strip()
    if not reconstructed:
        return False
    user_msg = {"role": "user", "content": reconstructed}
    heard = (event.get("transcript") or "").strip()
    if heard and heard != reconstructed:
        user_msg["heard"] = heard
    st.session_state.messages.append(user_msg)

    response = event.get("response") or ""
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "citations": event.get("citations") or [],
            "query_id": event.get("query_id"),
            "audio_status": "none",
            "tts_job_id": None,
            "audio": None,
        }
    )
    st.session_state.last_sources = {
        "audit_log": event.get("audit_log") or [],
        "retrieved_docs": event.get("retrieved_docs") or [],
    }
    st.session_state.voice_pending_read = response
    st.session_state.voice_error = ""
    return True


@st.fragment(run_every=1)
def tts_poller() -> None:
    pending = [
        m
        for m in st.session_state.messages
        if m.get("audio_status") == "processing" and m.get("tts_job_id")
    ]
    if not pending:
        return
    changed = False
    for message in pending:
        before = message.get("audio_status")
        poll_tts_for_message(message.get("query_id"), message["tts_job_id"])
        if message.get("audio_status") != before:
            changed = True
    if changed:
        st.rerun()


ensure_session_id()
ensure_messages()

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 1

if "active_ingest_job_id" not in st.session_state:
    st.session_state.active_ingest_job_id = None

if "last_orb_id" not in st.session_state:
    st.session_state.last_orb_id = None

if "voice_pending_read" not in st.session_state:
    st.session_state.voice_pending_read = ""

if "voice_error" not in st.session_state:
    st.session_state.voice_error = ""

st.title("Agentic RAG")
st.markdown("### Intelligent Document Analysis Platform")

backend_ok, health_info = check_backend()
if not backend_ok:
    st.error(
        f"Backend unavailable at `{Config.DOCUMIND_API_URL}`. "
        f"Start it with: `uvicorn backend.main:app --reload --port 8000`"
    )
    st.json(health_info)
    st.stop()

if Config.ENABLE_VOICE:
    last = last_assistant_message()
    orb_event = voice_orb(
        api_url=Config.DOCUMIND_API_URL,
        session_id=st.session_state.session_id,
        last_answer=(last or {}).get("content") or "",
        pending_read=st.session_state.get("voice_pending_read") or "",
        enable_tts=Config.ENABLE_TTS,
        key="talk_orb",
    )
    if consume_orb_event(orb_event):
        st.rerun()

tab_chat, tab_audit, tab_metrics = st.tabs(["💬 Chat", "📊 Audit Logs", "📈 Metrics"])

with st.sidebar:
    st.header("📤 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        disabled=bool(st.session_state.active_ingest_job_id),
    )

    if st.button(
        "Process Documents",
        type="primary",
        disabled=bool(st.session_state.active_ingest_job_id),
    ):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            try:
                with st.spinner(f"Uploading {len(uploaded_files)} file(s)..."):
                    files = [(f.name, f.getbuffer().tobytes()) for f in uploaded_files]
                    upload_result = api_client.upload_documents(files)
                st.session_state.active_ingest_job_id = upload_result["job_id"]
                st.rerun()
            except APIError as e:
                st.error(str(e))

    poll_active_ingestion_job()

    st.divider()
    st.markdown("### Processed Documents")
    try:
        docs = cached_documents()
        if docs:
            for doc in docs:
                st.markdown(f"- 📄 {doc}")
        else:
            st.caption("No documents processed yet.")
    except APIError as e:
        st.error(str(e))

    st.divider()
    st.markdown("### Voice")
    if Config.ENABLE_VOICE:
        st.caption("Use the talk orb in the bottom-right. Click once to start a voice session; click again to hang up. Incomplete questions are repeated back to you.")
        if not Config.ENABLE_TTS:
            st.caption("TTS disabled (ENABLE_TTS=false).")
        if st.session_state.get("voice_error"):
            st.error(st.session_state.voice_error)
    else:
        st.caption("Voice input is disabled (ENABLE_VOICE=false).")

    st.divider()
    st.markdown("### ⚙️ System Controls")
    if st.button(
        "🗑️ Clear Database",
        help="Delete all documents from vector store",
        type="secondary",
        use_container_width=True,
    ):
        try:
            api_client.clear_database()
            st.success("Database cleared successfully!")
            st.rerun()
        except APIError as e:
            st.error(str(e))

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

with tab_chat:
    chat_container = st.container()

    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("heard"):
                    st.caption(f"Heard: {message['heard']}")
                if message.get("role") == "assistant" and message.get("citations"):
                    st.markdown("**Citations:**")
                    for c in message.get("citations", []):
                        st.markdown(f"- {c}")

                if message.get("role") == "assistant":
                    audio_status = message.get("audio_status")
                    if message.get("audio"):
                        _render_answer_audio(message)
                    elif audio_status == "processing":
                        st.caption("Synthesizing voice...")
                    elif audio_status == "failed":
                        st.caption("Voice synthesis failed.")

                    if (
                        Config.ENABLE_TTS
                        and not message.get("audio")
                        and message.get("audio_status") != "processing"
                    ):
                        play_key = f"play_{i}_{message.get('query_id') or 'msg'}"
                        if st.button("Play answer", key=play_key):
                            start_tts_for_message(message)
                            st.rerun()

        process_pending_query()

        if any(
            m.get("audio_status") == "processing" and m.get("tts_job_id")
            for m in st.session_state.messages
        ):
            tts_poller()

    with chat_container:
        if st.session_state.get("last_sources"):
            with st.expander("📋 View Sources & Reasoning"):
                st.json(st.session_state.last_sources.get("audit_log", []))
                docs = st.session_state.last_sources.get("retrieved_docs", [])
                if docs:
                    st.write("**Retrieved Documents:**")
                    for i, doc in enumerate(docs):
                        filename = doc.get("metadata", {}).get("source", f"Document {i+1}")
                        page = doc.get("metadata", {}).get("page_number", "?")
                        st.caption(f"{filename} - Page {page}")
                        st.text((doc.get("content") or "")[:500] + "...")

    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        handle_user_query(prompt, tts_enabled=False)

with tab_audit:
    st.header("System Audit Trail")
    if st.button("Refresh Logs"):
        try:
            logs = api_client.get_audit_logs()
            if not logs:
                st.info("No logs available.")
            else:
                st.success(f"Loaded {len(logs)} log entries")
                for log in logs:
                    query_id = log.get("query_id") or "-"
                    timestamp = log.get("timestamp", "-")
                    with st.expander(f"{timestamp} — {query_id}"):
                        st.markdown(f"**Query:** {log.get('query', '-')}")
                        st.markdown(f"**Final response:** {log.get('final_response')}")
                        audit_trail = log.get("audit_trail", [])
                        if isinstance(audit_trail, str):
                            try:
                                audit_trail = json.loads(audit_trail)
                            except Exception:
                                pass
                        if audit_trail and isinstance(audit_trail, list):
                            st.markdown("**Audit Steps**")
                            desired_cols = [
                                "step", "status", "route", "strategy", "subqueries",
                                "retrieved_count", "reason", "error", "timestamp",
                            ]
                            cols = []
                            for item in audit_trail:
                                if isinstance(item, dict):
                                    for k in item:
                                        if k != "query" and k not in cols:
                                            cols.append(k)
                            ordered = [c for c in desired_cols if c in cols] + [
                                c for c in cols if c not in desired_cols
                            ]
                            rows = []
                            for item in audit_trail:
                                if isinstance(item, dict):
                                    row = {c: str(item.get(c, "")) for c in ordered}
                                    if "error" in row and len(row["error"]) > 100:
                                        row["error"] = row["error"][:100] + "..."
                                    rows.append(row)
                            if rows:
                                st.table(rows)
                            else:
                                st.json(audit_trail)
        except APIError as e:
            st.error(str(e))

with tab_metrics:
    st.header("RAG Evaluation Metrics")
    try:
        data = api_client.get_metrics()
        overall = data.get("overall", {})
        if overall:
            st.subheader("Final Averages")
            filtered = {k: v for k, v in overall.items() if "ndc" not in k.lower()}
            col1, col2, col3, col4 = st.columns(4)
            cols = [col1, col2, col3, col4]
            key_metrics = ["answer_correctness", "answer_relevancy", "faithfulness", "context_recall"]
            for i, km in enumerate(key_metrics):
                if km in filtered:
                    val = filtered[km]
                    val_str = f"{val * 100:.2f}%" if isinstance(val, (int, float)) else str(val)
                    cols[i % 4].metric(km.replace("_", " ").title(), val_str)
            avg_rows = [
                {
                    "Metric": k.replace("_", " ").title(),
                    "Average Score": f"{v:.4f}" if isinstance(v, float) else str(v),
                }
                for k, v in filtered.items()
            ]
            st.table(pd.DataFrame(avg_rows))

        st.markdown("---")
        st.subheader("Benchmark Query Results")
        per_item = data.get("per_item", [])
        if per_item:
            query_rows = []
            for item in per_item:
                row = {"Query ID": item.get("id")}
                for m_name, m_val in item.get("scores", {}).items():
                    if "ndc" not in m_name.lower():
                        row[m_name.replace("_", " ").title()] = (
                            round(m_val, 4) if isinstance(m_val, float) else m_val
                        )
                query_rows.append(row)
            st.dataframe(pd.DataFrame(query_rows), width="stretch")
    except APIError as e:
        st.info(str(e))
