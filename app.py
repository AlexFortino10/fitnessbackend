import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
import threading

app = FastAPI()  # Definizione dell'app FastAPI

class PromptRequest(BaseModel):
    prompt: str

PREDEFINED_RESPONSES = {
    "ciao": "Ciao! Come posso aiutarti oggi?",
    "come stai?": "Sto bene, grazie! E tu?",
    "allenamento": "Inizia con 10 minuti di stretching per scaldarti bene."
}

CACHE = {}  # Cache per memorizzare risposte frequenti
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
TIMEOUT = 30  # Timeout configurabile (secondi)

def clean_text(text: str) -> str:
    """
    Rimuove caratteri indesiderati come '\n' o altri caratteri non leggibili.
    """
    return " ".join(text.splitlines()).strip()

def preload_cache():
    """
    Pre-carica risposte per prompt comuni.
    """
    common_prompts = ["ciao", "come stai?", "allenamento"]
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    for prompt in common_prompts:
        payload = {"inputs": prompt}
        try:
            response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                CACHE[prompt] = clean_text(result[0].get("generated_text", ""))
            else:
                CACHE[prompt] = "Risposta non disponibile."
        except Exception as e:
            CACHE[prompt] = f"Errore nel pre-caricamento: {e}"

@app.on_event("startup")
async def on_startup():
    """
    Funzione di avvio per pre-caricare la cache.
    """
    threading.Thread(target=preload_cache).start()  # Usa threading per non bloccare il server

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip().lower()
    print(f"Ricevuto prompt: '{prompt}'")

    # Controllo nella cache
    if prompt in CACHE:
        print("Risposta trovata nella cache.")
        return CACHE[prompt]

    # Risposta predefinita
    if prompt in PREDEFINED_RESPONSES:
        response = clean_text(PREDEFINED_RESPONSES[prompt])
        CACHE[prompt] = response  # Salva nella cache
        return response

    # Controllo token
    if not HUGGINGFACE_TOKEN:
        return clean_text("Errore: Token Hugging Face mancante. Controlla la configurazione.")

    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 10,
            "temperature": 0.1,
            "top_k": 30,
            "top_p": 0.9,
            "do_sample": True
        }
    }

    try:
        print("Invio della richiesta al servizio Hugging Face...")
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            generated_text = clean_text(result[0].get("generated_text", ""))
            CACHE[prompt] = generated_text  # Salva nella cache
            return generated_text

        return clean_text("Errore nel generare la risposta dal modello esterno.")
    except requests.exceptions.ReadTimeout:
        print("Errore: Timeout raggiunto.")
        return clean_text("Il server è occupato. Riprova più tardi.")
    except requests.exceptions.RequestException as e:
        print(f"Errore di richiesta: {e}")
        return clean_text(f"Errore imprevisto: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
