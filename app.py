import streamlit as st
import os
import sys
import uuid
import hashlib
import logging


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


st.set_page_config(page_title="DocuMind AI", layout="wide", page_icon="📄")


if "messages" not in st.session_state:
    st.session_state.messages = []


if "orchestrator" not in st.session_state:
    try:
        st.session_state.vector_store = VectorStoreManager()
        st.session_state.orchestrator = Orchestrator(vector_store=st.session_state.vector_store)
        st.session_state.audit_logger = AuditLogger()
        st.session_state.loader = PDFLoader()
        st.session_state.chunker = DocumentChunker()
        logger.info("Orchestrator initialized")
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")

st.title("📄 DocuMind AI")
st.markdown("### Intelligent Document Analysis Platform")


tab_chat, tab_audit = st.tabs(["💬 Chat", "📊 Audit Logs"])


with st.sidebar:
    st.header("📤 Document Upload")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 1

    uploaded_files = st.file_uploader(
        "Upload PDFs", 
        type=['pdf'], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
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
                all_chunks = []
                temp_dir = "data/documents"
                os.makedirs(temp_dir, exist_ok=True)
                existing_sources = st.session_state.vector_store.get_processed_documents()

                for uploaded_file in uploaded_files:
                    st.write(f"Processing {uploaded_file.name}...")

                    if uploaded_file.name in existing_sources:
                        if replace_existing:
                            removed = st.session_state.vector_store.delete_by_source(
                                uploaded_file.name
                            )
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
                        pages = st.session_state.loader.load(file_path)
                        st.write(f"✅ Loaded {len(pages)} pages from {uploaded_file.name}")

                        doc_id = hashlib.sha256(
                            uploaded_file.name.encode()
                        ).hexdigest()[:32]
                        chunks = st.session_state.chunker.split_documents(
                            pages, doc_id, source=uploaded_file.name
                        )
                        for c in chunks:
                            c["metadata"]["source"] = uploaded_file.name

                        all_chunks.extend(chunks)

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                if all_chunks:
                    st.write(f"Embedding {len(all_chunks)} chunks...")
                    st.session_state.vector_store.add_chunks(all_chunks)
                    status.update(label="Processing Complete!", state="complete", expanded=False)
                    st.success(f"Successfully processed {len(all_chunks)} chunks into Knowledge Base.")
                    
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
    st.markdown("### ⚙️ System Controls")
    
    if st.button("🗑️ Clear Database", help="Delete all documents from vector store", type="secondary", use_container_width=True):
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
    
    with chat_container:
        if "last_sources" in st.session_state and st.session_state.last_sources:
            with st.expander("📋 View Sources & Reasoning"):
                st.json(st.session_state.last_sources.get("audit_log", []))
                
                docs = st.session_state.last_sources.get("retrieved_docs", [])
                if docs:
                    st.write("**Retrieved Documents:**")
                    for i, doc in enumerate(docs):
                        filename = doc['metadata'].get('source', f'Document {i+1}')
                        page = doc['metadata'].get('page_number', '?')
                        st.caption(f"{filename} - Page {page}")
                        st.text(doc.get('content', '')[:500] + "...")

    prompt = st.chat_input("Ask a question about your documents...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        query_id = str(uuid.uuid4())
                        result_state = st.session_state.orchestrator.run(
                            prompt, 
                            query_id=query_id, 
                            audit_logger=st.session_state.audit_logger
                        )
                        
                        response = result_state.get("final_response") or "I apologize, but I could not generate an answer. Please check the Audit Logs for more details."
                        
                        st.session_state.audit_logger.log_query(query_id, result_state)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        st.session_state.last_sources = {
                            "audit_log": result_state.get("audit_log", []),
                            "retrieved_docs": result_state.get("retrieved_docs", [])
                        }
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = f"An error occurred: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})






with tab_audit:
    st.header("System Audit Trail")
    if st.button("Refresh Logs"):
        logs = st.session_state.audit_logger.get_logs()
        
        import json
        for log in logs:
            if "audit_trail" in log:
                log["audit_trail"] = json.dumps(log["audit_trail"], indent=2)
                
        st.dataframe(logs)
