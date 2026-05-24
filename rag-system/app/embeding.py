from langchain_community.embeddings import HuggingFaceEmbeddings

def create_embedding():
    return HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")
