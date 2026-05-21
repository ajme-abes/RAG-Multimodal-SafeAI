import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()
# load PDF
pdf_path = r"C:/Users/ajmel/desktop/internship-projects/rag-system/data/ArtificiaL_.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

#create embeding
embeding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persistence_directory = "./chroma_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeding_model,
    collection_name="pdf_chunks",
    persist_directory=persistence_directory
)
vectorstore.persist()

print(f"Successfully loaded and embedded {len(chunks)} chunks into ChromaDB!")

