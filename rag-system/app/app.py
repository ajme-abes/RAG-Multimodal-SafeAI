import streamlit as st
import os
from tempfile import NamedTemporaryFile

# Import pipeline dependencies
from loader import extract_clean_text_pdf
from chunker import chunk_clean_text
from embedding import get_embedding_model
from vector_store import create_vector_store
from qa_pipeline import execute_query_pipeline

st.set_page_config(page_title="Production RAG System", page_icon="⚙️", layout="wide")
st.title("⚙️ Enterprise RAG Environment")
st.subheader("Production Workspace with Guarded Exception Boundaries")

@st.cache_resource
def initialize_embedding_layer():
    try:
        return get_embedding_model()
    except Exception as e:
        st.error(f"🚨 Failed to load embedding layer model weights: {e}")
        return None

embedding_model = initialize_embedding_layer()

# Track running session state attributes
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = ""

# Sidebar Ingestion Layout Panel
with st.sidebar:
    st.header("📂 Document Control Center")
    uploaded_file = st.file_uploader("Upload Target System Manual (PDF)", type=["pdf"])
    
    if uploaded_file:
        if st.session_state.processed_filename != uploaded_file.name:
            with st.spinner("Executing document parsing, text normalization, and vectorization layers..."):
                
                # Check for zero-byte empty uploads
                if uploaded_file.size == 0:
                    st.error("❌ Ingestion Rejected: The uploaded file is completely empty (0 bytes).")
                else:
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_filepath = tmp_file.name
                    
                    try:
                        pages_data, total_length = extract_clean_text_pdf(tmp_filepath)
                        
                        if pages_data and total_length > 0:
                            chunks, metadatas = chunk_clean_text(pages_data, uploaded_file.name)
                            
                            if chunks:
                                st.session_state.vector_store = create_vector_store(
                                    chunks, metadatas, embedding_model
                                )
                                st.session_state.processed_filename = uploaded_file.name
                                st.success(f"Successfully processed {len(chunks)} chunks!")
                            else:
                                st.error("❌ Segmentation Error: No valid string segments could be broken out from the document text.")
                        else:
                            st.error("❌ Loader Failure: The uploaded file contains zero digital text elements. It might be a scanned image or restricted by security passwords.")
                    except Exception as pipeline_err:
                        st.error(f"🚨 Pipeline Ingestion Exception: An unexpected error occurred while parsing: {pipeline_err}")
                    finally:
                        if os.path.exists(tmp_filepath):
                            os.remove(tmp_filepath)

    if st.button("Reset App Core Tables"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.processed_filename = ""
        st.rerun()

# Conversational Interface Section
for interaction in st.session_state.chat_history:
    with st.chat_message(interaction["role"]):
        st.markdown(interaction["content"])
        if "citations" in interaction and interaction["citations"]:
            st.markdown("### 📄 Sources:")
            for cit in interaction["citations"]:
                st.markdown(f"- **File:** `{cit['source']}` | **Page:** `{cit['page']}` *(Distance Score: {cit['score']})*")
                with st.expander(f"➔ View Snippet from Page {cit['page']}"):
                    st.info(f'"{cit["text"]}"')

if user_input := st.chat_input("Ask a question about your knowledge base..."):
    
    # Block empty query submission strings before hitting any resources
    if not user_input.strip():
        st.warning("⚠️ Cannot process an empty input string. Please enter a valid question.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("assistant"):
            # Check for empty state / missing database errors
            if st.session_state.vector_store is None:
                err_msg = "⚠️ Operational Block: No active reference knowledge database detected. Please upload a valid PDF manual in the sidebar menu before querying."
                st.warning(err_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": err_msg, "citations": []})
            else:
                with st.spinner("🔍 Querying Vector Database & Generating Structured Answer..."):
                    try:
                        answer, citations = execute_query_pipeline(
                            st.session_state.vector_store,
                            user_input,
                            chat_history=st.session_state.chat_history)
                        
                        st.markdown("### 📑 Answer:")
                        st.markdown(answer)
                        
                        if citations:
                            st.markdown("### 📄 Sources:")
                            for cit in citations:
                                st.markdown(f"- **File:** `{cit['source']}` | **Page:** `{cit['page']}` *(Distance Score: {cit['score']})*")
                                with st.expander(f"➔ View Snippet from Page {cit['page']}"):
                                    st.info(f'"{cit["text"]}"')
                                    
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations
                        })
                    except Exception as inference_crash:
                        fatal_msg = f"🚨 Execution Failure: An unrecoverable runtime error occurred inside the generation pipeline: {inference_crash}"
                        st.error(fatal_msg)