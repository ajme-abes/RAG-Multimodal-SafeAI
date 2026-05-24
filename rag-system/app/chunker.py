from langchain_text_splitters import NLTKTextSplitter

def chunk_clean_text(text, chunk_size=800, chunk_overlap=150):
    text_splitter = NLTKTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunk_list = text_splitter.split_text(text)

    return chunk_list