# app.py
import streamlit as st
import os
from tempfile import NamedTemporaryFile

# Import your customized pipeline modules
from loader import extract_clean_text_pdf
from chunker import chunk_clean_text
from embeding import get_embedding_model
from vectore_store import create_vector_store
from qa_pipeline import execute_query_pipeline

# Configure Web Interface Styles
st.set_page_config(page_title="Modular RAG System", page_icon="💡", layout="wide")
st.title("💡 Production Modular RAG Application")
st.subheader("Brought together via isolated functional script files")

# Initialize Shared Resources inside Streamlit's Cached Resources
@st.cache_resource
def initialize_embedding_layer():
    return get_embedding_model()

embedding_model = initialize_embedding_layer()

# Track running session state attributes
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = ""

# Sidebar - Document Ingestion controls
with st.sidebar:
    st.header("📂 Document Control Center")
    uploaded_file = st.file_uploader("Upload a document manual (PDF)", type=["pdf"])
    
    if uploaded_file:
        # Run ingestion only if this is a brand new file
        if st.session_state.processed_filename != uploaded_file.name:
            with st.spinner("Processing document architecture elements..."):
                
                # Write to a temporary file to give a clear string path to PyPDF
                with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_filepath = tmp_file.name
                
                try:
                    # 1. Load and Clean Pages
                    pages_data, total_length = extract_clean_text_pdf(tmp_filepath)
                    
                    if pages_data:
                        # 2. Generate Recursive Chunks with Source Page Metadata
                        chunks, metadatas = chunk_clean_text(pages_data, uploaded_file.name)
                        
                        # 3. Initialize and Index Vector Database Store
                        st.session_state.vector_store = create_vector_store(
                            chunks, metadatas, embedding_model
                        )
                        st.session_state.processed_filename = uploaded_file.name
                        st.success(f"Successfully indexed {len(chunks)} text chunks!")
                    else:
                        st.error("Could not parse meaningful text characters from this PDF.")
                finally:
                    # Clean up file path tracking
                    if os.path.exists(tmp_filepath):
                        os.remove(tmp_filepath)

    if st.button("Reset App & Chat History"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.processed_filename = ""
        st.rerun()

# Main Chat Display Logic Area
# Render previous historical chat dialog bubbles
for interaction in st.session_state.chat_history:
    with st.chat_message(interaction["role"]):
        st.markdown(interaction["content"])
        if "citations" in interaction and interaction["citations"]:
            with st.expander("🔍 View Grounded Citations"):
                for cit in interaction["citations"]:
                    st.caption(f"**Source:** {cit['source']} | **Page:** {cit['page']}")
                    st.info(f"\"{cit['text']}\"")

# Accept incoming runtime chat inquiries
if user_input := st.chat_input("Ask a question about your knowledge base..."):
    
    # Print user bubble instantly
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Process retrieval generation pipeline execution
    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            warning_msg = "Please upload and process a technical PDF manual on the sidebar layout before initiating questions."
            st.warning(warning_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": warning_msg, "citations": []})
        else:
            with st.spinner("Executing pipeline query loop..."):
                # Call query pipeline which implements the fallback sequence models
                answer, citations = execute_query_pipeline(st.session_state.vector_store, user_input)
                
                # Show generated response answers
                st.markdown(answer)
                
                # Display structural source block tracking components if they exist
                if citations:
                    with st.expander("🔍 View Grounded Citations"):
                        for cit in citations:
                            st.caption(f"**Source:** {cit['source']} | **Page:** {cit['page']}")
                            st.info(f"\"{cit['text']}\"")
                
                # Commit response state data frames back into conversation memory history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })