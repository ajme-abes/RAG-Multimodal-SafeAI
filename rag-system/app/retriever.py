from vectore_store import create_vector_store

def create_retriever(vector_store):
    query = input("enter your query")
    retriver = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k": 3}
    )
    similar_docs = retriver.invoke(query)

    print("top 3: similar chunks recived from vector store")

    for docs in similar_docs:
        print(f"-- {docs.page_content[:100]}")
