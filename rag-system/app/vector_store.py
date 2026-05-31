from langchain_community.vectorstores import Chroma
import chromadb
import uuid

def create_vector_store(chunk_list, metadatas, embedding_model):
    persistent_client = chromadb.PersistentClient(path="./chroma_db")
    
    
    collection = persistent_client.get_or_create_collection("rag_chunks")
    
    unique_ids = [str(uuid.uuid4()) for _ in range(len(chunk_list))]

    vector_store = Chroma.from_texts(
        texts=chunk_list,
        metadatas=metadatas,
        embedding=embedding_model,
        collection_name="rag_chunks",
        client=persistent_client,
        ids=unique_ids

    )
    return vector_store