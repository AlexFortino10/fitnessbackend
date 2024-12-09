import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
import re
from tenacity import retry, wait_fixed, stop_after_attempt, RetryError

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# Risposte predefinite
PREDEFINED_RESPONSES = {
    "ciao": "Ciao! Come posso aiutarti oggi?",
    "come stai?": "Sto bene, grazie! E tu?",
    "allenamento": "Inizia con 10 minuti di stretching per scaldarti bene.",
}

# Configurazione Hugging Face
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
HTTP_CLIENT = httpx.AsyncClient(timeout=40)  # Timeout di 40 secondi
FALLBACK_RESPONSE = "Non riesco a rispondere in questo momento, ma possiamo riprovare!"

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def fetch_from_huggingface(prompt: str):
    if not HUGGINGFACE_TOKEN:
        raise ValueError("Errore: Token Hugging Face mancante.")
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 50,  # Aumento la lunghezza massima
            "temperature": 0.6,
            "top_k": 40,
            "top_p": 0.9,
            "do_sample": True,
        },
    }
    try:
        response = await HTTP_CLIENT.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Solleva un'eccezione se la risposta è di errore
        return response.json()
    except httpx.RequestError as e:
        print(f"Errore nella richiesta HTTP: {e}")
        raise
    except httpx.HTTPStatusError as e:
        print(f"Errore HTTP {e.response.status_code}: {e.response.text}")
        raise
    except Exception as e:
        print(f"Errore durante la chiamata a Hugging Face: {e}")
        raise

def clean_text(prompt: str, text: str) -> str:
    # Rimuove newline, caratteri speciali e spazi multipli
    text = re.sub(r"[\n\r*#\\]", " ", text)  # Rimuove i caratteri speciali
    text = re.sub(r"\s+", " ", text)  # Sostituisce spazi multipli con uno singolo
    escaped_prompt = re.escape(prompt.strip())  # Escape del prompt per evitare conflitti
    pattern = rf"^{escaped_prompt}\W*"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip()
    print(f"Ricevuto prompt: '{prompt}'")

    # 1. Controllo risposte predefinite
    if prompt.lower() in PREDEFINED_RESPONSES:
        return PREDEFINED_RESPONSES[prompt.lower()]  # Restituisce solo il testo senza "response"

    try:
        # 2. Chiamata asincrona a Hugging Face
        print("Invio richiesta a Hugging Face...")
        result = await fetch_from_huggingface(prompt)
        if isinstance(result, list) and len(result) > 0:
            generated_text = clean_text(prompt, result[0].get("generated_text", ""))
            return generated_text  # Restituisce direttamente il testo generato, senza "response"
        return FALLBACK_RESPONSE
    except RetryError:
        print("Numero massimo di tentativi superato durante il recupero del modello.")
        return FALLBACK_RESPONSE
    except Exception as e:
        print(f"Errore: {e}")
        return FALLBACK_RESPONSE

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
