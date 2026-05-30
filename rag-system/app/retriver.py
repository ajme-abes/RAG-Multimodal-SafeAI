
def create_retriever(vector_store, query, distance_threshold=0.88, k=3):
    
    raw_text = vector_store.similarity_search_with_score(query, k=k)
    filtered_doc = []

    for doc, score in raw_text:
        if score <= distance_threshold:
            doc.metadata["score"] = round(float(score), 4)
            filtered_doc.append(doc)

    return filtered_doc