from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_clean_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunk_list = text_splitter.split_text(text)

    return chunk_list