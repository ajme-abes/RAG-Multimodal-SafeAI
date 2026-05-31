
def create_retriever(vector_store, query, distance_threshold=0.75, k=4):
    
    raw_text = vector_store.similarity_search_with_score(query, k=k)
    filtered_doc = []

    print("\n🔍 --- CHROMA DATABASE SEARCH LOGS ---")
    for idx, (doc, score) in enumerate(raw_text):
        print(f"   -> Match #{idx+1} | Distance Score: {score:.4f} | Preview: {doc.page_content[:50]}...")
        
        if score <= distance_threshold:
            # Save the score into metadata so qa_pipeline can display it in sources
            doc.metadata["distance_score"] = round(float(score), 4)
            filtered_doc.append(doc)
        else:
            print(f"      ❌ DROPPED: Score {score:.4f} exceeds threshold {distance_threshold}")
            
    print("---------------------------------------\n")
    return filtered_doc