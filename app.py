import os
import tempfile
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud secrets -> environment variables (works even for nested/odd configs)
try:
    for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "GEMINI_MODEL", "EMBEDDING_MODEL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # no secrets file locally
    pass

from src.rag_pipeline import MAX_PAGES, RAGPipeline  # noqa: E402

MAX_FILE_MB = 25

st.set_page_config(page_title="RAG Document QA", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Connecting to the vector database...")
def get_pipeline(code_version=RAGPipeline.VERSION):
    # code_version is part of the cache key, so a redeploy never reuses an old cached pipeline object
    return RAGPipeline()


def show_sources(sources):
    with st.expander("Sources (Page Citations)"):
        for name, page in sorted({(s["source"], s["page"]) for s in sources}):
            st.markdown(f"- {name} - Page {page}")


def main():
    st.title("📄 Multimodal RAG-Based Intelligent Document QA System")
    st.markdown("Upload PDFs and ask questions. Answers come only from your documents, with page citations.")

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        st.warning("Please set GEMINI_API_KEY in your secrets / .env file to proceed.")
        st.stop()

    try:
        pipeline = get_pipeline()
        if time.time() - st.session_state.get("_last_cleanup", 0) > 3600:  # hourly housekeeping
            pipeline.cleanup_expired()
            st.session_state["_last_cleanup"] = time.time()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not start the pipeline: {e}")
        st.stop()

    if pipeline.storage_mode == "memory":
        if pipeline.connection_error:
            st.warning("Could not connect to Qdrant Cloud, so temporary in-memory storage is being used "
                       f"(documents are lost on restart). Details: {pipeline.connection_error}")
        else:
            st.info("QDRANT_URL / QDRANT_API_KEY not set - using temporary in-memory storage.")

    # Every browser session gets its own private slice of the vector store.
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("documents", {})
    sid = st.session_state.session_id

    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(f"Upload PDF Documents (Max {MAX_FILE_MB}MB each)",
                                          type=["pdf"], accept_multiple_files=True,
                                          help=f"Text-based PDFs only, up to {MAX_PAGES} pages each.")
        if st.button("Process Documents", type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Ingesting documents and generating embeddings..."):
                    for f in uploaded_files:
                        if f.size > MAX_FILE_MB * 1024 * 1024:
                            st.error(f"{f.name} exceeds the {MAX_FILE_MB}MB limit.")
                            continue
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(f.getvalue())
                            path = tmp.name
                        try:
                            n = pipeline.ingest_document(path, display_name=f.name, session_id=sid)
                            st.session_state.documents[f.name] = n
                            st.success(f"{f.name}: {n} chunks stored.")
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Failed to process {f.name}: {e}")
                        finally:
                            if os.path.exists(path):
                                os.remove(path)

        if st.session_state.documents:
            st.subheader("Loaded documents")
            for name, n in st.session_state.documents.items():
                st.caption(f"📄 {name} ({n} chunks)")
            if st.button("Clear documents & chat"):
                try:
                    pipeline.clear_session(sid)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not clear documents: {e}")
                st.session_state.documents = {}
                st.session_state.messages = []
                st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    if prompt := st.chat_input("Ask a question about the uploaded documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if not st.session_state.documents:
                st.info("Please upload and process at least one PDF first.")
                return
            try:
                stream, docs = pipeline.answer_query(prompt, session_id=sid)
                answer = st.write_stream(stream)
                sources = [{"source": d.metadata.get("source", "document"),
                            "page": d.metadata.get("page", 0) + 1} for d in docs]
                if sources:
                    show_sources(sources)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as e:  # noqa: BLE001
                st.error(f"An error occurred while answering: {e}")


if __name__ == "__main__":
    main()
