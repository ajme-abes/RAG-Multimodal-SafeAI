import os
from langchain_text_splitters import NLTKTextSplitter
import nltk

# CRITICAL FIX: Explicitly download and verify the sentence splitter dataset
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def chunk_clean_text(pages_data, file_name, chunk_size=800, chunk_overlap=150):
    text_splitter = NLTKTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    final_chunks = []
    final_metadatas = []

    for page in pages_data:
        page_chunks = text_splitter.split_text(page["text"])
        for chunk in page_chunks:
            if len(chunk.strip()) > 10:
                final_chunks.append(chunk)
                final_metadatas.append({
                    "source_file": file_name,
                    "page_number": page["page_number"]
                })

    # Diagnostic printout so you can see exactly how many chunks are built in your terminal
    print(f"📊 DEBUG: Built {len(final_chunks)} chunks for file: {file_name}")
    return final_chunks, final_metadatas