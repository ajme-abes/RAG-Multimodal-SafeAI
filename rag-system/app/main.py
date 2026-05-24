# 1. Bring in the tools from your other files
from loader import extract_clean_text_pdf
from chunker import chunk_clean_text
from embeding import create_embedding
from vectore_store import create_vector_store
from retriever import create_retriever
def run_pipeline(pdf_path):
    print("=== STARTING RAG DATA PIPELINE ===")
    
    # 2. Load and Clean the PDF
    print(f"\nStep 1: Reading and cleaning text from: {pdf_path}...")
    full_text, total_length = extract_clean_text_pdf(pdf_path)
    
    # Safety check if the PDF failed to load
    if not full_text:
        print("Pipeline stopped: Could not read any text from the PDF.")
        return
        
    print(f"Success! Cleaned {total_length} characters from the document.")

    # 3. Chunk the Cleaned Text
    print(f"\nStep 2: Splitting text into chunks (Size: 800, Overlap: 150)...")
    chunks = chunk_clean_text(full_text, chunk_size=800, chunk_overlap=150)
    
    print(f"Success! Created {len(chunks)} total chunks.")

    # create embedding
    print(f"\n step3: craete embedding for chunks...")
    embeddings = create_embedding()
    print("sucess! embedding craeted")

    #4. stor vectore in db
    print(f"creating vectore store")
    vectore_store = create_vector_store(chunks, embeddings)
    print("success! vectore store created and stored in chroma db")

    #5 retrive similar chunks
    print("retrive similar chunks from vectore store")
    retriver = create_retriever(vectore_store)
    print("sucessfully retrived similar chunks from vectore store")


   


if __name__ == "__main__":
    # The path to your actual PDF file
    target_pdf = r"C:/Users/ajmel/desktop/internship-projects/rag-system/data/ArtificiaL_.pdf"
    
    # Run the connected pipeline
    run_pipeline(target_pdf)
