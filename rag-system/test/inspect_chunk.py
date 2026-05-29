import os, sys

# Ensure the parent `rag-system` directory is on sys.path so `app` package imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.loader import extract_clean_text_pdf
from app.chunker import chunk_clean_text

def inspect_chunk(pdf_path):

    print("--------starting debuging chunking quality--------------")

    cleaned_text, total_length = extract_clean_text_pdf(pdf_path)
    chunks, metadata = chunk_clean_text(cleaned_text, file_name=pdf_path, chunk_size=800, chunk_overlap=150)

    too_short_count = 0
    broken_sentence_count = 0
    total_chars = 0


    min_len_treshold = 40
    valid_endings = (".", "!", "?", '"', "'", "”", "’")

    for idx, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        total_chars += chunk_len

        is_too_short = chunk_len < min_len_treshold

        is_broken_sentence = not chunk[-1].endswith(valid_endings)

        if is_too_short or is_broken_sentence:
            print(f"issues found in chunk {idx + 1}: Length={chunk_len}, Ends with valid punctuation: {not is_broken_sentence}")

            if is_too_short:
                print(f"[Critical]: chunks fells below the minimum semanic length threshold")
            if is_broken_sentence:
                print(f"[Critical]: chunk ends with a broken sentence, which may cause loss of context for the LLM")
            
            if is_too_short: too_short_count += 1
            if is_broken_sentence: broken_sentence_count += 1

    avg_chunk = total_chars / len(chunks) if chunks else 0
    broken_ratio = (broken_sentence_count / len(chunks)) * 100 if chunks else 0

    print("\n--------chunking quality report--------------")
    print(f"Total chunks: {len(chunks)}")
    print(f"Too short chunks: {too_short_count}")
    print(f"Broken sentence chunks: {broken_sentence_count}")
    print(f"Average chunk length: {avg_chunk:.2f}")
    print(f"Broken sentence ratio: {broken_ratio:.2f}%")

    #
    print("\n architectural recommendations:")

    if broken_ratio > 30:
        print(f"[Critical]: A high percentage of chunks end with broken sentences. Consider adjusting chunking parameters or implementing smarter sentence-aware chunking to preserve context for the LLM.")
    elif avg_chunk < 150:
        print(f"[Warning]: The average chunk length is quite low, which may lead to inefficient use of the LLM's context window. Consider increasing the chunk size or reducing overlap to create more meaningful chunks.")
    else:
        print(f"[Success]: The chunking quality appears to be good with a reasonable average length and low broken sentence ratio. You can proceed with this configuration for your RAG system.")
if __name__ == "__main__":
    pdf_path = r"C:/Users/ajmel/desktop/internship-projects/rag-system/data/ArtificiaL_.pdf"
    inspect_chunk(pdf_path)