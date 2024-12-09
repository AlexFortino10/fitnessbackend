import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
import re
from tenacity import retry, wait_fixed, stop_after_attempt, RetryError
from cachetools import LRUCache

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# Risposte predefinite
PREDEFINED_RESPONSES = {
    "ciao": "Ciao! Come posso aiutarti oggi?",
    "come stai?": "Sto bene, grazie! E tu?",
    "allenamento": "Inizia con 10 minuti di stretching per scaldarti bene.",
}

# Cache per risposte recenti
CACHE = LRUCache(maxsize=100)

# Configurazione Hugging Face
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
HTTP_CLIENT = httpx.AsyncClient(timeout=40)  # Timeout di 10 secondi
FALLBACK_RESPONSE = "Non riesco a rispondere in questo momento, ma possiamo riprovare!"

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def fetch_from_huggingface(prompt: str):
    if not HUGGINGFACE_TOKEN:
        raise ValueError("Errore: Token Hugging Face mancante.")
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 15,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "do_sample": True,
        },
    }
    response = await HTTP_CLIENT.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def clean_text(prompt: str, text: str) -> str:
    text = re.sub(r"[\n\r*]", " ", text)
    text = re.sub(r"\s+", " ", text)
    escaped_prompt = re.escape(prompt.strip())
    pattern = rf"^{escaped_prompt}\W*"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip()
    print(f"Ricevuto prompt: '{prompt}'")
    if prompt.lower() in PREDEFINED_RESPONSES:
        return {"response": PREDEFINED_RESPONSES[prompt.lower()]}

    if prompt.lower() in CACHE:
        return {"response": CACHE[prompt.lower()]}

    try:
        result = await fetch_from_huggingface(prompt)
        if isinstance(result, list) and len(result) > 0:
            generated_text = clean_text(prompt, result[0].get("generated_text", ""))
            CACHE[prompt.lower()] = generated_text
            return {"response": generated_text}
        return {"response": FALLBACK_RESPONSE}
    except RetryError:
        return {"response": FALLBACK_RESPONSE}
    except Exception as e:
        print(f"Errore: {e}")
        return {"response": FALLBACK_RESPONSE}

@app.on_event("startup")
async def warm_up_model():
    try:
        await fetch_from_huggingface("Ciao")
        print("Modello riscaldato correttamente.")
    except Exception as e:
        print(f"Errore durante il warm-up: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    await HTTP_CLIENT.aclose()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
