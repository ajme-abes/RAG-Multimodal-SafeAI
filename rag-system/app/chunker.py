# chunker.py
import os
from langchain_text_splitters import NLTKTextSplitter
import nltk

# Ensure NLTK tokenizers are downloaded automatically
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def chunk_clean_text(pages_data, file_name, chunk_size=800, chunk_overlap=150):
    text_splitter = NLTKTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    final_chunks = []
    final_metadatas = []

    for page in pages_data:
        # Split text *within* this specific page
        page_chunks = text_splitter.split_text(page["text"])
        
        for chunk in page_chunks:
            if len(chunk.strip()) > 10: # Filter out empty artifacts
                final_chunks.append(chunk)
                # Map metadata to the chunk for future citations
                final_metadatas.append({
                    "source_file": file_name,
                    "page_number": page["page_number"]
                })

    return final_chunks, final_metadatas