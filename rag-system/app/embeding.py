from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    # Initializes and returns the all-MiniLM-L6-v2 embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings