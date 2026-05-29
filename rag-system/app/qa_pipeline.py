import subprocess
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from retriever import create_retriever

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def run_ollama(prompt, model="llama3"):
    try:
        # Optimization: communicate using standard subprocess pattern cleanly
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )
        stdout, _ = process.communicate(input=prompt, timeout=60)
        return stdout.strip()
    except Exception as e:
        print(f"Error running Ollama: {e}")
        return None

def run_huggingface(prompt):
    if not HF_API_KEY:
        return None
    try:
        API_URL = "https://api-inference.huggingface.co/models/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=30)
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        return None
    except Exception as e:
        print(f"Hugging Face error: {e}")
        return None

def run_openai(prompt, model="gpt-4o-mini"):
    if not OPENAI_API_KEY:
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None
    
def run_google_genai(prompt, model="gemini-2.5-flash"):
    if not GOOGLE_API_KEY:
        return None
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Google GenAI error: {e}")
        return None

def execute_query_pipeline(vector_store, user_query):

    if not user_query or user_query.strip():
        return "⚠️ Error: Received an empty or invalid search query. Please type a meaningful question.", []

    if vector_store is None: 
        return "⚠️ Error: The vector store is not initialized. Please upload and process a PDF document first.", []
    
    print(f"\n[1] Processing User Incoming  Question: {user_query}")
    
    # 1. Retrieve raw similar document chunks
    try:
       search_result = create_retriever(vector_store, user_query, distance_threshold=0.80)
    except Exception as e:
        print(f"Error occurred while retrieving search results: {e}")
        return "An error occurred while processing your query.", []

    if not search_result:
        print("No relevant chunks retrieved from vector store. Returning fallback response.")
        return "I cannot find the answer within the provided documentation.", []


    # 2. Extract and format all chunks into a unified string context block
    context_pieces = []
    citations = []

    if search_result:
        for idx, doc in enumerate(search_result):
            context_pieces.append(f"--- Context Chunk {idx+1} ---\n{doc.page_content}")
            citations.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source_file", "Unknown Document"),
            "page": doc.metadata.get("page_number", "Unknown Page"),
            "score": doc.metadata.get("score", 0.0)

            })

        unified_context = "\n\n".join(context_pieces)
        print(f"[2] Retrieval successful. Found {len(search_result)} relevant references.")
    else:
        # Gracefully handle total retrieval failures / off-topic queries
        print("ℹ️ Zero chunks passed the distance threshold. Proceeding with empty context fallback.")
        unified_context = "NO RELEVANT CONTENT FOUND IN KNOWLEDGE BASE."
    
    

    # 3. Formulate the Strict Grounded Prompt 
    grounded_prompt = f"""
    You are a highly precise enterprise documentation clerk. Your sole objective is to answer the final user question using ONLY the facts explicitly provided within the Reference Context block below.

    === REFERENCE CONTEXT ===
    {unified_context}
    =========================

    FINAL USER QUESTION: {user_query}

    STRICT INSTRUCTIONS:
    1. Base your answer entirely on the provided context. If the answer is not explicitly written there, respond with exactly: "I cannot find the relevant information inside the uploaded documentation."
    2. DO NOT use conversational filler (e.g., "Based on the context provided...", "Sure, here is the answer..."). Start immediately with the direct facts.
    3. You MUST format your answer using clean, professional Markdown. Use bold headers or bullet points if explaining a multi-step process or a list of specifications.
    """

    # 4. Fallback Execution Loop Chain
    print(f"[3] Dispatching Grounded Prompt to Model Stack...")
    response_text = run_ollama(grounded_prompt)
    
    if not response_text or "Error" in response_text:
        print("--> Ollama failed/missing. Attempting HuggingFace Fallback...")
        response_text = run_huggingface(grounded_prompt)

    if not response_text:
        print("--> HuggingFace failed/missing. Attempting OpenAI Fallback...")
        response_text = run_openai(grounded_prompt)

    if not response_text:
        print("--> OpenAI failed/missing. Attempting Google GenAI Fallback...")
        response_text = run_google_genai(grounded_prompt)

    if not response_text:
        response_text = "All model generation APIs failed to respond. Please check your connectivity and API keys."

    print(f"\n[4] Final Generated Response Synthesized.")
    return response_text, citations