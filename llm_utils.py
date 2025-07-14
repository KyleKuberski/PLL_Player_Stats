import hashlib
import os
import json
import ollama

def get_cached_llama_response(prompt: str, cache_dir="llm_cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{prompt_hash}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["response"]

    # Query Ollama locally (or switch to OpenAI if hosted)
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )["message"]["content"]

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "response": response}, f)

    return response
