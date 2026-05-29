from langchain_community.vectorstores import Chroma
import chromadb

def create_vector_store(chunk_list, metadatas, embedding_model):
    persistent_client = chromadb.PersistentClient(path="./chroma_db")
    
    # CRITICAL FIX: Clear old collections to prevent state corruption
    try:
        persistent_client.delete_collection("rag_chunks")
        print("🧹 DEBUG: Cleared old vector tables successfully.")
    except Exception:
        pass

    vector_store = Chroma.from_texts(
        texts=chunk_list,
        metadatas=metadatas,
        embedding=embedding_model,
        collection_name="rag_chunks",
        client=persistent_client
    )
    return vector_store