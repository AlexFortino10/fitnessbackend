import os
import httpx  # Usato per richieste asincrone
from fastapi import FastAPI
from pydantic import BaseModel
import time
import re
import asyncio
from tenacity import retry, wait_fixed, stop_after_attempt, RetryError

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

# Configurazione client HTTP
HTTP_CLIENT = httpx.AsyncClient(timeout=10)

def clean_text(prompt: str, response: str) -> str:
    """
    Rimuove il prompt e qualsiasi sua variante dalla risposta.
    """
    response = " ".join(response.splitlines()).strip()  # Rimuove caratteri indesiderati

    # Creare un pattern robusto per individuare il prompt
    escaped_prompt = re.escape(prompt.strip())
    pattern = rf"^{escaped_prompt}\W*"  # Cerca il prompt all'inizio della risposta con spazi o punteggiatura

    # Rimuove il prompt dalla risposta
    cleaned_response = re.sub(pattern, "", response, flags=re.IGNORECASE).strip()

    return cleaned_response

# Funzione di retry con tenacity
@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def fetch_from_huggingface(prompt: str):
    """
    Funzione asincrona che invia una richiesta al servizio Hugging Face.
    Usa tenacity per gestire i retry in caso di errore temporaneo.
    """
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
    response.raise_for_status()  # Solleva un'eccezione se la risposta è di errore
    return response.json()

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
        return {"response": PREDEFINED_RESPONSES[prompt]}

    # 2. Controllo nella cache
    if prompt in CACHE:
        print(f"Risposta trovata nella cache per il prompt: '{prompt}'")
        return {"response": CACHE[prompt]}

    try:
        # 3. Chiamata asincrona al modello Hugging Face
        print("Invio della richiesta al servizio Hugging Face...")
        result = await fetch_from_huggingface(prompt)

        if isinstance(result, list) and len(result) > 0:
            generated_text = clean_text(prompt, result[0].get("generated_text", ""))
            CACHE[prompt] = generated_text  # Salvataggio nella cache
            return {"response": generated_text}

        return {"response": "Errore nel generare la risposta dal modello esterno."}
    
    except RetryError:
        return {"response": "Il server è temporaneamente occupato. Riprova più tardi."}

    except Exception as e:
        print(f"Errore: {e}")
        return {"response": "Errore nel comunicare con il server. Riprova più tardi."}

@app.on_event("shutdown")
async def shutdown_event():
    """
    Chiude il client HTTP quando il server viene spento.
    """
    await HTTP_CLIENT.aclose()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
