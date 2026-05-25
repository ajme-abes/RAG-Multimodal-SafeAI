import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.chunker import chunk_clean_text
from app.loader import extract_clean_text_pdf
from app.embeding import create_embedding
from langchain_community.vectorstores import Chroma
import chromadb
def run_semantic_search_test(pdf_path):
    print("=== STARTING RAG SEMANTIC SEARCH TEST ===")
    
    # Load and clean the PDF
    print(f"\nStep 1: Reading and cleaning text from: {pdf_path}...")
    full_text, total_length = extract_clean_text_pdf(pdf_path)
    
    if not full_text:
        print("Test stopped: Could not read any text from the PDF.")
        return
        
    print(f"Success! Cleaned {total_length} characters from the document.")

    # Chunk the Cleaned Text
    print(f"\nStep 2: Splitting text into chunks (Size: 800, Overlap: 150)...")
    chunks = chunk_clean_text(full_text, chunk_size=800, chunk_overlap=150)
    
    print(f"Success! Created {len(chunks)} total chunks.")

    # create embedding
    print(f"\n step3: craete embedding for chunks...")
    embeddings = create_embedding()
    print("sucess! embedding craeted")

    perssisten_client = chromadb.PersistentClient("./chroma_db")

    vectore_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        client = perssisten_client,
        collection_name= "pdf_chunks_test"

    )

    test_query = [
        "How does the Claude Platform Workflow process BibTeX files?",
        "What role does prompt engineering play in LLM-based literature analysis?"
    ]

    retriver = vectore_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k": 3}
    )

    for query in test_query:
        print("\n" + "="*50)
        print(f"🔍 TESTING QUERY: '{query}'")
        print("="*50)
        similar_docs = retriver.invoke(query)

        if not similar_docs:
            print("Test failed: No similar documents retrieved.")
            continue

        print(f"found {len(similar_docs)} similar chunks for the query.")
        for idx, doc in enumerate(similar_docs):

            print(f"[match {idx+1}] previwe")

            clean_preview = doc.page_content.replace("\n", " ").strip()
            print(f"  \"{clean_preview[:200]}...\"")
            print(f"-"*50)
if __name__ == "__main__":
    target_pdf = r"C:/Users/ajmel/desktop/internship-projects/rag-system/data/ArtificiaL_.pdf"
    run_semantic_search_test(target_pdf)
