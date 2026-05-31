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
if "uploaded_documents_list" not in st.session_state:
    st.session_state.uploaded_documents_list = []

# Sidebar Ingestion Layout Panel
with st.sidebar:
    st.header("📂 Multi-Doc Knowledge base")
    
    # Clear visual status check of active knowledge repositories
    if st.session_state.uploaded_documents_list:
        st.markdown("### 📚 Active Indexed Files:")
        for doc_name in st.session_state.uploaded_documents_list:
            st.caption(f"✅ `{doc_name}`")
        st.markdown("---")
    else:
        st.info("No documents uploaded yet. Database workspace is empty.")

    uploaded_file = st.file_uploader("Upload a System Manual (PDF)", type=["pdf"])
    
    if uploaded_file:
        # Move the name check inside the action execution block
        if uploaded_file.name not in st.session_state.uploaded_documents_list:
            with st.spinner(f"Vectorizing and appending `{uploaded_file.name}`..."):
                if uploaded_file.size == 0:
                    st.error("❌ Ingestion Rejected: Empty file.")
                else:
                    from tempfile import NamedTemporaryFile
                    import os
                    
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_filepath = tmp_file.name
                    
                    try:
                        pages_data, total_length = extract_clean_text_pdf(tmp_filepath)
                        
                        if pages_data and total_length > 0:
                            chunks, metadatas = chunk_clean_text(pages_data, uploaded_file.name)
                            
                            if chunks:
                                # Append to the continuous vector index matrix
                                st.session_state.vector_store = create_vector_store(
                                    chunks, metadatas, embedding_model
                                )
                                # Save the name to the memory list to update the visual sidebar checklist
                                st.session_state.uploaded_documents_list.append(uploaded_file.name)
                                st.success(f"Appended {len(chunks)} chunks successfully!")
                                st.rerun() # Refresh layout to cleanly transition state
                            else:
                                st.error("❌ Segmentation Error: No clean chunks extracted.")
                        else:
                            st.error("❌ Loader Failure: Document text unreadable.")
                    except Exception as pipeline_err:
                        st.error(f"🚨 Pipeline Ingestion Exception: {pipeline_err}")
                    finally:
                        if os.path.exists(tmp_filepath):
                            os.remove(tmp_filepath)
        else:
            # Show a clean, non-blocking informational caption inside the sidebar instead of a harsh warning banner
            st.caption(f"ℹ️ `{uploaded_file.name}` is active in the repository vector space.")

    if st.button("Purge Entire Knowledge Base Storage"):
        import shutil
        import os
        import gc
        
        # 1. CRITICAL: Kill the active memory client binds to release file locks
        if "vector_store" in st.session_state and st.session_state.vector_store is not None:
            try:
                # Tell Chroma to close connections and clear its internal systems
                st.session_state.vector_store._client.reset() 
            except Exception:
                pass
        
        # Completely nullify the memory tracking variables
        st.session_state.vector_store = None
        st.session_state.uploaded_documents_list = []
        st.session_state.chat_history = []
        
        # Run Python garbage collection to forcefully flush the dead file hooks out of RAM
        gc.collect()
        
        # 2. Now it is completely safe to delete the physical directory without lock exceptions
        if os.path.exists("./chroma_db"):
            try:
                shutil.rmtree("./chroma_db")
                print("🗑️ Database directory wiped cleanly.")
            except Exception as e:
                print(f"Error deleting database files: {e}")
                
        st.success("Database and session hooks fully wiped!")
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
                st.markdown("### 📑 Answer:")
                answer_container = st.empty()
                
                with st.spinner("🔍 Retrieving context chunks..."):
                    pipeline_output, citations, confidence = execute_query_pipeline(
                        st.session_state.vector_store, 
                        user_input, 
                        chat_history=st.session_state.chat_history
                    )
                
                # 2. Resilient Stream Extraction Loop (Handles Gemini, OpenAI, and pure string generators)
                if hasattr(pipeline_output, "__iter__") and not isinstance(pipeline_output, (str, list)):
                    
                    def unified_text_chunk_extractor():
                        for chunk in pipeline_output:
                            # Condition A: It's a Gemini Stream Chunk
                            if hasattr(chunk, "text"):
                                if chunk.text:
                                    yield chunk.text
                                    
                            # Condition B: It's an OpenAI Stream Chunk
                            elif hasattr(chunk, "choices"):
                                if chunk.choices and chunk.choices[0].delta.content:
                                    yield chunk.choices[0].delta.content
                                    
                            # Condition C: It's a raw string generator token (Ollama / Hugging Face / Custom)
                            elif isinstance(chunk, str):
                                yield chunk

                    # Animate the text writing to the screen dynamically in real-time
                    final_answer = answer_container.write_stream(unified_text_chunk_extractor())
                else:
                    # Fallback rendering if a normal static text string was passed back
                    final_answer = pipeline_output
                    answer_container.markdown(final_answer)
                
                # 3. Render clean enterprise source mapping blocks right beneath the stream
                if citations:
                    st.markdown("### 📄 Sources:")
                    for cit in citations:
                        st.markdown(f"- **File:** `{cit['source']}` | **Page:** `{cit['page']}` *(Distance Score: {cit['score']})*")
                        with st.expander(f"➔ View Snippet from Page {cit['page']}"):
                            st.info(f'"{cit["text"]}"')
                            
                # 4. Commit the fully generated answer text string to your chat history logs
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_answer,
                    "citations": citations
                })