import chromadb

# sample data
raw_document = (
    "The domestic cat (Felis catus) is a small carnivorous mammal. It is the only "
    "domesticated species in the family Felidae. Cats are known for their agility, "
    "exceptional hunting skills, and flexible bodies. On the other hand, the "
    "domestic dog (Canis lupus familiaris) is a domesticated descendant of the wolf. "
    "Dogs were the first species to be domesticated by humans over 15,000 years ago, "
    "evolving to become uniquely attuned to human behavior."
)

# chunking
def text_chunking(text, chunk_size = 20, overlap = 5):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)

        i += chunk_size - overlap

    return chunks
chunk_text = text_chunking(raw_document, chunk_size=20, overlap=5)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="animal_knowladge")
unique_ids = [f"chunks_{i}" for i in range(len(chunk_text))]
collection.add(
    documents=chunk_text,
    ids=unique_ids
)

query = "Tell me about feline traits"

search_result = collection.query(
    query_texts= [query],
    n_results=1 
)

collection.get(ids=["chunks_0"])

print(f"______rag pipeline______")
print(f"user query: {query}\n")
print(f"Retrived Context Chunk:")
print(search_result["documents"][0][0])
print(f"\nDistance Score (Lower means closer match): {search_result['distances'][0][0]}")
