# vector.py
from langchain_community.vectorstores import Chroma
import chromadb

def create_vector_store(chunk_list, metadatas, embedding_model):
    # Initializes a local storage client 
    persistent_client = chromadb.PersistentClient(path="./chroma_db")

    # Builds the vector store along with text snippets and source metadata
    vector_store = Chroma.from_texts(
        texts=chunk_list,
        metadatas=metadatas,
        embedding=embedding_model,
        collection_name="rag_chunks",
        client=persistent_client
    )
    return vector_store