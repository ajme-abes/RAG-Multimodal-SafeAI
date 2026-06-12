import time
import random
from google.genai.errors import APIError

def retry_with_backoff(func, max_retries=5, initial_delay=2, backoff_factor=2):
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except APIError as e:
            if e.code in [429, 503] and attempt < max_retries:
                jitter = random.uniform(0, 1)
                sleep_time = delay + jitter
                print(f"⚠️ Gemini Server Overloaded ({e.code}). Retry {attempt}/{max_retries} in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                delay *= backoff_factor  
            else:
                print(f"🚨 Max Retries hit or Unrecoverable API Error code: {e.code}")
                raise e
        except Exception as e:
            print(f"🚨 Non-API unexpected execution breakdown: {str(e)}")
            raise e