# qa_pipeline.py
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
    print(f"\n[1] User query received: {user_query}")
    
    # 1. Retrieve raw similar document chunks
    search_results = create_retriever(vector_store, user_query)
    
    if not search_results:
        return "I cannot find any text context matching your query.", []

    # 2. Extract and format all chunks into a unified string context block
    context_pieces = []
    citations = []
    
    for idx, doc in enumerate(search_results):
        context_pieces.append(f"--- Context Chunk {idx+1} ---\n{doc.page_content}")
        citations.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source_file", "Unknown Document"),
            "page": doc.metadata.get("page_number", "Unknown Page")
        })
        
    unified_context = "\n\n".join(context_pieces)
    print(f"[2] Retrieved {len(search_results)} relevant chunks.")

    # 3. Formulate the Strict Grounded Prompt 
    grounded_prompt = f"""
You are a strict technical Q&A clerk. Your job is to answer the User Question using ONLY the factual statements provided in the Reference Context block below.

=== REFERENCE CONTEXT ===
{unified_context}
=========================

USER QUESTION: {user_query}

STRICT INSTRUCTIONS:
1. Base your answer entirely on the Reference Context. 
2. If the answer cannot be explicitly found within the context, you must reply exactly with: "I cannot find the answer within the provided documentation." Do not guess or extrapolate.
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