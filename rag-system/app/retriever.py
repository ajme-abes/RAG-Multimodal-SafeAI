from vectore_store import create_vector_store

def create_retriever(vector_store, query):
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k": 3}
    )
    similar_docs = retriever.invoke(query)

    return similar_docs


