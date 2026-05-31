# qa_pipeline.py
import subprocess
import os
import requests
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

load_dotenv()

# ==========================================
# PERSISTENT CLIENT INITIALIZATION (Connection Continuity Guard)
# ==========================================
google_client = None
if os.getenv("GOOGLE_API_KEY"):
    try:
        google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception as e:
        print(f"Failed to initialize global Google GenAI client: {e}")

openai_client = None
if os.getenv("OPENAI_API_KEY"):
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print(f"Failed to initialize global OpenAI client: {e}")

# ==========================================
# 1. STATIC INFERENCE ENGINES (For Background Query Condensing)
# ==========================================
def run_google_genai_static(prompt):
    if not google_client:
        return None
    try:
        response = google_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Google GenAI static error: {e}")
        return None

def run_openai_static(prompt):
    if not openai_client:
        return None
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI static error: {e}")
        return None

def run_ollama_static(prompt, model="llama3"):
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )
        stdout, _ = process.communicate(input=prompt, timeout=12)
        return stdout.strip()
    except Exception:
        return None

def run_huggingface_static(prompt):
    token = os.getenv("HF_API_KEY")
    if not token:
        return None
    try:
        API_URL = "https://api-inference.huggingface.co/models/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=12)
        if response.status_code == 200:
            return response.json()[0]['generated_text'].strip()
    except Exception:
        return None

# ==========================================
# 2. STREAMING INFERENCE ENGINES (For Live UI Animatics)
# ==========================================
def run_openai_stream(prompt):
    if not openai_client:
        return None
    try:
        return openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            timeout=15
        )
    except Exception as e:
        print(f"OpenAI streaming failure: {e}")
        return None

def run_google_genai_stream(prompt):
    if not google_client:
        return None
    try:
        return google_client.models.generate_content_stream(model="gemini-2.5-flash", contents=prompt)
    except Exception as e:
        print(f"Google GenAI streaming failure: {e}")
        return None

def run_ollama_stream(prompt, model="llama3"):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": True}
        response = requests.post(url, json=payload, stream=True, timeout=10)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded_line = json.loads(line.decode('utf-8'))
                    yield decoded_line.get("response", "")
    except Exception as e:
        print(f"Ollama stream exception: {e}")
        return

def run_huggingface_stream(prompt):
    token = os.getenv("HF_API_KEY")
    if not token:
        return
    try:
        API_URL = "https://api-inference.huggingface.co/models/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 512, "return_full_text": False}
        }
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            full_text = response.json()[0]['generated_text']
            for word in full_text.split(" "):
                yield word + " "
                time.sleep(0.02) # Smoothes word stream rendering cadence
    except Exception as e:
        print(f"Hugging Face stream exception: {e}")
        return

# ==========================================
# 3. CONVERSATIONAL QUERY CONDENSER LAYER
# ==========================================
def condense_user_query(user_query, chat_history):
    if not chat_history:
        return user_query  

    formatted_memory = ""
    for interaction in chat_history[-3:]:
        role_label = "User" if interaction["role"] == "user" else "Assistant"
        formatted_memory += f"{role_label}: {interaction['content']}\n"

    condensing_prompt = f"""
Analyze the following conversation history and the latest follow-up question. 
Extract and output ONLY the primary subject keywords needed to search a database for the answer. 
Do NOT write a full sentence. Do NOT include words like "what", "is", "of", "the", "it", or "this".

=== CHAT HISTORY ===
{formatted_memory}
====================

FOLLOW-UP QUESTION: {user_query}

CRITICAL KEYWORDS OUTPUT:"""

    # Multi-vendor static check loop for background keywords expansion
    rewritten_query = run_google_genai_static(condensing_prompt)
    if not rewritten_query or "Error" in rewritten_query:
        rewritten_query = run_openai_static(condensing_prompt)
    if not rewritten_query or "Error" in rewritten_query:
        rewritten_query = run_ollama_static(condensing_prompt)
    if not rewritten_query or "Error" in rewritten_query:
        rewritten_query = run_huggingface_static(condensing_prompt)
        
    if rewritten_query:
        clean_keywords = rewritten_query.strip().replace('"', '').replace("'", "")
        print(f"🔄 [Query Rewriter]: Transformed '{user_query}' -> '{clean_keywords}'")
        return clean_keywords
    
    return user_query

# ==========================================
# 4. RUNTIME PIPELINE ORCHESTRATION WITH ANCHORED SCORE CALCULATIONS
# ==========================================
def execute_query_pipeline(vector_store, user_query, chat_history=None, distance_threshold=0.85):
    if not user_query or not user_query.strip():
        return "⚠️ Error: Received an empty query.", [], "0%"

    # 1. Run history text condensation
    search_query = condense_user_query(user_query, chat_history)
    print(f"\n[1] Executing vector database search for: '{search_query}'")
    
    # 2. Run vector index matching search
    try:
        from retriver import create_retriever
        search_results = create_retriever(vector_store, search_query, distance_threshold=distance_threshold)
    except Exception as e:
        print(f"🚨 Vector store retrieval crashed: {e}")
        return "🚨 System error occurred while reading database index.", [], "0%"

    context_pieces = []
    citations = []
    total_score = 0.0
    
    # 3. Context consolidation & Feature 4 Confidence Scoring Evaluation
    if search_results:
        for idx, doc in enumerate(search_results):
            context_pieces.append(f"--- Context Chunk {idx+1} ---\n{doc.page_content}")
            score = doc.metadata.get("distance_score", 0.0)
            total_score += score
            citations.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source_file", "Unknown Document"),
                "page": doc.metadata.get("page_number", "Unknown Page"),
                "score": score
            })
        unified_context = "\n\n".join(context_pieces)
        
        # Translate raw Euclidean/Cosine distance metrics into user-facing percentage values
        avg_distance = total_score / len(search_results)
        confidence_percentage = max(0, min(100, int((1.0 - avg_distance) * 100)))
        confidence_display = f"{confidence_percentage}%"
    else:
        print("ℹ️ Zero chunks passed threshold. Guard triggered: Returning fallback instruction.")
        unified_context = "NO RELEVANT CONTENT FOUND IN KNOWLEDGE BASE."
        confidence_display = "0% (No Context Match)"

    # 4. Formulate memory strings for your target prompt layout
    formatted_memory = ""
    if chat_history:
        for interaction in chat_history[-4:]:
            role_label = "User" if interaction["role"] == "user" else "Assistant"
            formatted_memory += f"{role_label}: {interaction['content']}\n"

    # Your exact requested grounded system prompt architecture
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

    print(f"[3] Routing prompt payload to multi-provider streaming stack...")
    
    # 5. Cascading Multi-Vendor Stream Check Loop
    # Vendor A: OpenAI Cloud Stream
    if openai_client:
        stream_generator = run_openai_stream(grounded_prompt)
        if stream_generator:
            print("➔ Initialized stream via OpenAI gpt-4o-mini.")
            return stream_generator, citations, confidence_display

    # Vendor B: Google GenAI Cloud Stream
    if google_client:
        stream_generator = run_google_genai_stream(grounded_prompt)
        if stream_generator:
            print("➔ Initialized stream via Google Gemini-2.5-Flash.")
            return stream_generator, citations, confidence_display

    # Vendor C: Local Ollama Server Process Stream
    try:
        check_res = requests.get("http://localhost:11434/", timeout=2)
        if check_res.status_code == 200:
            stream_generator = run_ollama_stream(grounded_prompt)
            if stream_generator:
                print("➔ Initialized stream via local Ollama engine.")
                return stream_generator, citations, confidence_display
    except Exception:
        pass

    # Vendor D: Hugging Face Public Inference Hub Stream
    if os.getenv("HF_API_KEY"):
        stream_generator = run_huggingface_stream(grounded_prompt)
        if stream_generator:
            print("➔ Initialized stream via Hugging Face Inference API.")
            return stream_generator, citations, confidence_display

    # Terminal Pipeline Error Fallback Exception
    error_msg = "🚨 Generation Operational Error: No live API streams or local servers responded."
    return error_msg, citations, "0%"