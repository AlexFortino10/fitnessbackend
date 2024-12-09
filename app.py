import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# Risposte predefinite per prompt frequenti
PREDEFINED_RESPONSES = {
    "ciao": "Ciao! Come posso aiutarti oggi?",
    "come stai?": "Sto bene, grazie! E tu?",
    "allenamento": "Inizia con 10 minuti di stretching per scaldarti bene.",
}

# Cache per risposte recenti
CACHE = {}

# Configurazione API Hugging Face
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"

def clean_text(text: str) -> str:
    """
    Rimuove caratteri indesiderati e formatta il testo.
    """
    return " ".join(text.splitlines()).strip()

def preload_cache():
    """
    Pre-carica la cache per prompt comuni e pre-riscalda il modello.
    """
    print("Pre-caricamento della cache e pre-riscaldamento del modello...")
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    common_prompts = ["ciao", "come stai?", "allenamento"]

    for prompt in common_prompts:
        payload = {"inputs": prompt, "parameters": {"max_length": 50}}
        try:
            # Richiesta di pre-riscaldamento
            response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()

            # Estrarre e salvare nella cache
            if isinstance(result, list) and len(result) > 0:
                CACHE[prompt] = clean_text(result[0].get("generated_text", ""))
                print(f"Risposta pre-caricata per '{prompt}': {CACHE[prompt]}")
            else:
                CACHE[prompt] = "Risposta non disponibile."
        except Exception as e:
            CACHE[prompt] = f"Errore durante il pre-riscaldamento: {e}"
            print(f"Errore durante il pre-caricamento per '{prompt}': {e}")

@app.on_event("startup")
async def on_startup():
    """
    Funzione di avvio per pre-caricare la cache e pre-riscaldare il modello.
    """
    preload_cache()

@app.post("/generate")
async def generate_text(request: PromptRequest):
    """
    Gestisce il prompt e restituisce una risposta generata o predefinita.
    """
    prompt = request.prompt.strip().lower()
    print(f"Ricevuto prompt: '{prompt}'")

    # 1. Controllo risposte predefinite
    if prompt in PREDEFINED_RESPONSES:
        print(f"Risposta predefinita trovata per il prompt: '{prompt}'")
        return PREDEFINED_RESPONSES[prompt]

    # 2. Controllo nella cache
    if prompt in CACHE:
        print(f"Risposta trovata nella cache per il prompt: '{prompt}'")
        return CACHE[prompt]

    # 3. Richiesta al modello Hugging Face
    if not HUGGINGFACE_TOKEN:
        return "Errore: Token Hugging Face mancante."

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

    try:
        print("Invio della richiesta al servizio Hugging Face...")
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()

        # 4. Estrarre e salvare nella cache
        if isinstance(result, list) and len(result) > 0:
            generated_text = clean_text(result[0].get("generated_text", ""))
            CACHE[prompt] = generated_text  # Salvataggio nella cache
            return generated_text

        return "Errore nel generare la risposta dal modello esterno."
    except requests.exceptions.RequestException as e:
        print(f"Errore di richiesta al modello: {e}")
        return "Il server è occupato o lento. Riprova più tardi."

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
