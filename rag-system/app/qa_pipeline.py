# qa_pipeline.py
import subprocess
import os
import requests
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import errors

load_dotenv()

# ==========================================
# PERSISTENT CLIENT INITIALIZATION
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
# 1. STATIC INFERENCE ENGINES (For Background Tasks)
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
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=60)
        if response.status_code == 200:
            return response.json()[0]['generated_text'].strip()
    except Exception:
        return None

# ==========================================
# 2. RESILIENT STREAM EVALUATORS (Prevents 429 Runtime Crashes)
# ==========================================
def run_openai_stream(prompt):
    if not openai_client:
        return None
    try:
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            timeout=15
        )
        
        # Helper generator to test the first chunk immediately
        def stream_guard():
            iterator = iter(stream)
            try:
                first_chunk = next(iterator)
                yield first_chunk
                for chunk in iterator:
                    yield chunk
            except Exception as e:
                print(f"OpenAI mid-stream or initial quota exception: {e}")
                raise e
                
        return stream_guard()
    except Exception as e:
        print(f"OpenAI pipeline connection rejected: {e}")
        return None

def run_google_genai_stream(prompt):
    if not google_client:
        return None
    try:
        raw_stream = google_client.models.generate_content_stream(model="gemini-2.5-flash", contents=prompt)
        
        # CRITICAL FIX: Peek inside the first token chunk before returning to app.py
        def stream_guard():
            iterator = iter(raw_stream)
            try:
                # If this next() fails due to a 429 quota error, it jumps straight to the except block!
                first_chunk = next(iterator)
                yield first_chunk
                for chunk in iterator:
                    yield chunk
            except errors.ClientError as ce:
                print(f"⚠️ Google Quota Exhausted (429) detected via stream guard logic.")
                raise ce
            except Exception as e:
                print(f"Google unexpected streaming error: {e}")
                raise e
                
        # Return our protected generator proxy
        return stream_guard()
    except Exception as e:
        print(f"Google GenAI connection rejected: {e}")
        return None

def run_ollama_stream(prompt, model="llama3"):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": True}
        response = requests.post(url, json=payload, stream=True, timeout=120)
        
        if response.status_code == 200:
            def token_yield_loop():
                for line in response.iter_lines():
                    if line:
                        decoded_line = json.loads(line.decode('utf-8'))
                        yield decoded_line.get("response", "")
            return token_yield_loop()
    except Exception as e:
        print(f"Ollama stream exception: {e}")
    return None

def run_huggingface_stream(prompt):
    token = os.getenv("HF_API_KEY")
    if not token:
        return None
    try:
        url = "https://api-inference.huggingface.co/models/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "return_full_text": False}}
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            def mock_stream():
                full_text = response.json()[0]['generated_text']
                for word in full_text.split(" "):
                    yield word + " "
                    time.sleep(0.02)
            return mock_stream()
    except Exception as e:
        print(f"Hugging Face stream exception: {e}")
    return None

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
# 4. RUNTIME PIPELINE ORCHESTRATION WITH ACTIVE RECOVERY
# ==========================================
def execute_query_pipeline(vector_store, user_query, chat_history=None, distance_threshold=0.85):
    if not user_query or not user_query.strip():
        return "⚠️ Error: Received an empty query.", [], "0%"

    search_query = condense_user_query(user_query, chat_history)
    print(f"\n[1] Executing vector database search for: '{search_query}'")
    
    try:
        from retriver import create_retriever
        search_results = create_retriever(vector_store, search_query, distance_threshold=distance_threshold)
    except Exception as e:
        print(f"🚨 Vector store retrieval crashed: {e}")
        return "🚨 System error occurred while reading database index.", [], "0%"

    context_pieces = []
    citations = []
    total_score = 0.0
    
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
        avg_distance = total_score / len(search_results)
        confidence_percentage = max(0, min(100, int((1.0 - avg_distance) * 100)))
        confidence_display = f"{confidence_percentage}%"
    else:
        print("ℹ️ Zero chunks passed threshold. Returning fallback instruction.")
        unified_context = "NO RELEVANT CONTENT FOUND IN KNOWLEDGE BASE."
        confidence_display = "0%"

    formatted_memory = ""
    if chat_history:
        for interaction in chat_history[-4:]:
            role_label = "User" if interaction["role"] == "user" else "Assistant"
            formatted_memory += f"{role_label}: {interaction['content']}\n"

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

    print(f"[3] Evaluating resilient fallback streaming channels...")
    
    print(f"[3] Evaluating resilient fallback streaming channels (Priority: Local -> HF -> OpenAI -> Gemini)...")
    
    # Cascade Level 1: Try Local Ollama Instance
    try:
        check_res = requests.get("http://localhost:11434/", timeout=60)
        if check_res.status_code == 200:
            stream = run_ollama_stream(grounded_prompt)
            if stream:
                try:
                    # Probe the local stream to verify the server is actively generating tokens
                    peek_stream = iter(stream)
                    first_token = next(peek_stream)
                    
                    def reassembled_stream():
                        yield first_token
                        for chunk in peek_stream:
                            yield chunk
                            
                    print("➔ Success: Active stream routed via Local Ollama.")
                    return reassembled_stream(), citations, confidence_display
                except Exception:
                    print("➔ Local Ollama generation failed to initialize. Cascading down...")
    except Exception:
        print("➔ Local Ollama server offline. Cascading to Hugging Face...")

    # Cascade Level 2: Try Hugging Face Hub
    if os.getenv("HF_API_KEY"):
        stream = run_huggingface_stream(grounded_prompt)
        if stream:
            try:
                # Probe the Hugging Face API to ensure it isn't loading or rate-limited
                peek_stream = iter(stream)
                first_token = next(peek_stream)
                
                def reassembled_stream():
                    yield first_token
                    for chunk in peek_stream:
                        yield chunk
                        
                print("➔ Success: Active stream routed via Hugging Face API.")
                return reassembled_stream(), citations, confidence_display
            except Exception:
                print("➔ Hugging Face Inference endpoint unavailable. Cascading to OpenAI...")

    # Cascade Level 3: Try OpenAI Stream
    if openai_client:
        stream = run_openai_stream(grounded_prompt)
        if stream:
            try:
                # Probe the stream to ensure it doesn't fail on quota/billing checks
                peek_stream = iter(stream)
                first_token = next(peek_stream)
                
                def reassembled_stream():
                    yield first_token
                    for chunk in peek_stream:
                        yield chunk
                        
                print("➔ Success: Active stream routed via OpenAI.")
                return reassembled_stream(), citations, confidence_display
            except Exception:
                print("➔ OpenAI stream validation failed (Quota/Token limits). Cascading to Gemini...")

    # Cascade Level 4: Try Google GenAI Stream
    if google_client:
        stream = run_google_genai_stream(grounded_prompt)
        if stream:
            try:
                # Probe the stream to intercept 429 RESOURCE_EXHAUSTED conditions instantly
                peek_stream = iter(stream)
                first_token = next(peek_stream)
                
                def reassembled_stream():
                    yield first_token
                    for chunk in peek_stream:
                        yield chunk
                        
                print("➔ Success: Active stream routed via Google Gemini.")
                return reassembled_stream(), citations, confidence_display
            except Exception:
                print("➔ Gemini stream validation failed (Quota Exhausted or API error).")

    # Final Catch-All System Error
    return "🚨 Generation Operational Error: All local servers and API cloud streaming pipelines are currently unavailable or rate-limited.", citations, "0%"