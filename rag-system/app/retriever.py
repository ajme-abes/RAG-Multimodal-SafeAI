# retriever.py

def create_retriever(vector_store, query):
    # Setup MMR search to maximize both relevance and chunk diversity
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )
    similar_docs = retriever.invoke(query)
    return similar_docs