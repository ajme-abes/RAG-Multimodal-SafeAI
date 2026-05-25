from retriever import create_retriever
import subprocess
def run_ollama(prompt, model="llama3"):
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def excute_query_pipeline(vector_store, user_query):
    print(f"\n[1] user query recived: {user_query}")
    # 1. Create a retriever
    search_result = create_retriever(vector_store, user_query)

    retrieved_results = search_result[0].page_content
    print(f"\n[2] retrieved relevant chunks: {retrieved_results}")

    grounded_prompt = f"""
    You are a strict technical Q&A clerk. Your job is to answer the User Question using ONLY the factual statements provided in the Reference Context block below.

    === REFERENCE CONTEXT ===
    {retrieved_results}
    =========================

    USER QUESTION: {user_query}

    STRICT INSTRUCTIONS:
    1. Base your answer entirely on the Reference Context. 
    2. If the answer cannot be explicitly found within the context, you must reply exactly with: "I cannot find the answer within the provided documentation." Do not guess.
    """

    # 2. Generate a response using the grounded prompt

    # 2. Generate a response using Ollama
    print(f"[3] Dispatching Grounded Prompt to Ollama...")
    response_text = run_ollama(grounded_prompt, model="llama3")

    print(f"\n[4] Final generated response")
    print("-" * 20)
    print(response_text)
    print("-" * 20)

    

