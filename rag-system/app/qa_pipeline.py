import subprocess
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from retriver import create_retriever

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
    
def condense_user_query(user_query, chat_history):
    if not chat_history:
        return user_query
    
    formatted_memory = ""
    recent_chat_history = chat_history[-3:]
    for interaction in recent_chat_history:
        roled_label = "User" if interaction["role"] == "user" else "Assistant"
        formatted_memory += f"{roled_label}: {interaction['content']}\n"
    condensing_prompt = f"""
    Given the following conversation history and a follow-up question, rewrite the follow-up question into a standalone, explicit question that contains all necessary context. Do NOT use pronouns like "it", "this", or "they". Make it a clean keyword search string for a vector database.

    === CHAT HISTORY ===
    {formatted_memory}
    ====================

    FOLLOW-UP QUESTION TO REWRITE: {user_query}

    STANDALONE QUESTION OUTPUT:"""  

    rewrettin_query = run_ollama(condensing_prompt)
    if not rewrettin_query or "Error" in rewrettin_query:
        rewrettin_query = run_huggingface(condensing_prompt)
    if not rewrettin_query or "Error" in rewrettin_query:
        rewrettin_query = run_openai(condensing_prompt)
    if not rewrettin_query or "Error" in rewrettin_query:
        rewrettin_query = run_google_genai(condensing_prompt)

    if rewrettin_query:
        print(f"[Query Rewritter ]: Transformed {user_query} --> {rewrettin_query.strip()}")
        return rewrettin_query.strip()
    return user_query
    


def execute_query_pipeline(vector_store, user_query, chat_history=None, distance_threshold=0.88):

    if not user_query or not user_query.strip():
        return "⚠️ Error: Received an empty or invalid search query. Please type a meaningful question.", []

    search_query = condense_user_query(user_query, chat_history)
    print(f"\n[1] Processing User Incoming  Question: {user_query}")

    # 1. Retrieve raw similar document chunks
    try:
       search_result = create_retriever(vector_store, user_query, distance_threshold=distance_threshold)
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

    formatted_memory = ""
    if chat_history:
        for interaction in chat_history[-4:]:
            role_label = "User" if interaction["role"] == "user" else "Assistant"
            formatted_memory += f"{role_label}: {interaction['content']}\n"
    
    

    # 3. Formulate the Strict Grounded Prompt 
    grounded_prompt = f"""

    You are a highly precise enterprise knowledge assistant. Your goal is to answer the final User Question.
    You must maintain situational continuity by analyzing the prior chat conversation context window.

    === SHORT TERM CHAT MEMORY HISTORY ===
    {formatted_memory if formatted_memory else "No history"}
    =======================================

    === CRITICAL REFERENCE CONTEXT DOCUMENTATION ===
    {unified_context}
    ================================================

    FINAL USER QUESTION: {user_query}

   STRICT COMPLIANCE RULES:
   1. Base your answer primarily on the provided Reference Context documentation. 
   2. If the answer cannot be found or logically derived from the Reference Context, but can be answered using the Short Term Chat Memory History above, you may use the memory to provide continuity.
   3. If neither the context nor the memory contains the answer, reply exactly with: "I cannot find the relevant information inside the uploaded documentation." Do not invent facts.
   4. Keep your output direct, professional, and objective. Do not add casual introductory remarks.
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