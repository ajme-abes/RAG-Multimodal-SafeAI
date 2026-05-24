from langchain_community.vectorstores import Chroma
import chromadb

def create_vector_store(chunk_list, embedding_model):
    perssisten_client = chromadb.PersistentClient("./chroma_db")

    vectore_store = Chroma.from_texts(
        texts = chunk_list,
        embedding = embedding_model,
        collection_name = "rag_chunks",
        client = perssisten_client
    )
    return vectore_store