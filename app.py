import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Document QA", page_icon="📄", layout="wide")

@st.cache_resource
def get_pipeline():
    return RAGPipeline()

def main():
    st.title("📄 Multimodal RAG-Based Intelligent Document QA System")
    st.markdown("Upload PDFs and ask questions. Powered by Gemini 3.6 Flash and Qdrant.")
    
    # Check for GEMINI_API_KEY
    if not os.environ.get("GEMINI_API_KEY"):
        st.warning("Please set GEMINI_API_KEY in your .env file or environment variables to proceed.")
        st.stop()

    pipeline = get_pipeline()
    if pipeline.storage_mode == "memory":
        if pipeline.connection_error:
            st.warning("Could not connect to Qdrant Cloud, so temporary in-memory storage is being used "
                       "(uploaded documents will be lost when the app restarts). "
                       "Check QDRANT_URL / QDRANT_API_KEY in your secrets and that the cluster is running. "
                       f"Details: {pipeline.connection_error}")
        else:
            st.info("QDRANT_URL / QDRANT_API_KEY not set - using temporary in-memory storage.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader("Upload PDF Documents (Max 25MB each)", type=['pdf'], accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Ingesting documents and generating embeddings..."):
                    total_chunks = 0
                    for uploaded_file in uploaded_files:
                        # Ensure file size is within limits (25MB)
                        if uploaded_file.size > 25 * 1024 * 1024:
                            st.error(f"File {uploaded_file.name} exceeds 25MB limit.")
                            continue
                            
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Ingest the document
                            chunks_count = pipeline.ingest_document(tmp_file_path)
                            total_chunks += chunks_count
                        except Exception as e:
                            st.error(f"Failed to process {uploaded_file.name}: {e}")
                        finally:
                            # Purge temp file immediately after upsert to Vector DB
                            if os.path.exists(tmp_file_path):
                                os.remove(tmp_file_path)
                            
                    if total_chunks > 0:
                        st.success(f"Success! Vector embeddings for {len(uploaded_files)} document(s) ({total_chunks} chunks) stored in Qdrant.")
            else:
                st.error("Please upload at least one PDF.")
                
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources (Page Citations)"):
                    unique_pages = sorted(list(set(s["page"] for s in message["sources"])))
                    for page in unique_pages:
                        st.markdown(f"- Page {page}")
                        
    # Chat input
    if prompt := st.chat_input("Ask a question about the uploaded documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            try:
                response_stream, retrieved_docs = pipeline.answer_query(prompt)
                
                # Stream the response
                full_response = st.write_stream(response_stream)
                
                # Extract sources for citation
                sources = []
                for doc in retrieved_docs:
                    page = doc.metadata.get('page', 0) + 1
                    sources.append({"page": page})
                
                # Display sources
                unique_pages = sorted(list(set(s["page"] for s in sources)))
                
                if unique_pages:
                    with st.expander("Sources (Page Citations)"):
                        for page in unique_pages:
                            st.markdown(f"- Page {page}")
                            
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
