# 1. Bring in the tools from your other files
from loader import extract_and_clean_pdf
from chunker import chunk_clean_text

def run_pipeline(pdf_path):
    print("=== STARTING RAG DATA PIPELINE ===")
    
    # 2. Load and Clean the PDF
    print(f"\nStep 1: Reading and cleaning text from: {pdf_path}...")
    full_text, total_length = extract_and_clean_pdf(pdf_path)
    
    # Safety check if the PDF failed to load
    if not full_text:
        print("Pipeline stopped: Could not read any text from the PDF.")
        return
        
    print(f"Success! Cleaned {total_length} characters from the document.")

    # 3. Chunk the Cleaned Text
    print(f"\nStep 2: Splitting text into chunks (Size: 500, Overlap: 100)...")
    chunks = chunk_clean_text(full_text, chunk_size=500, chunk_overlap=100)
    
    print(f"Success! Created {len(chunks)} total chunks.")

    # 4. Preview the results
    print(f"\n=========================================")
    print(f"PIPELINE OUTPUT PREVIEW")
    print(f"=========================================")
    
    # Show the first 2 chunks as a test sample
    for index, chunk in enumerate(chunks[:2]):
        print(f"\n--- CHUNK {index + 1} (Length: {len(chunk)} characters) ---")
        print(chunk)
        print("-" * 40)

if __name__ == "__main__":
    # The path to your actual PDF file
    target_pdf = r"C:/Users/ajmel/desktop/internship-projects/rag-system/data/ArtificiaL_.pdf"
    
    # Run the connected pipeline
    run_pipeline(target_pdf)
