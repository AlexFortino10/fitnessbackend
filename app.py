import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import re
import asyncio
import httpx  # Usato per richieste asincrone
import time  # Per i ritardi tra i tentativi

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# Risposte predefinite per prompt frequenti
PREDEFINED_RESPONSES = {
    "ciao": "Ciao! Come posso aiutarti oggi?",
    "come stai?": "Sto bene, grazie! E tu?",
    "allenamento": "Inizia con 10 minuti di stretching per scaldarti bene.",
}

# Configurazione API Hugging Face
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"

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

@app.on_event("startup")
async def on_startup():
    """
    Pre-riscalda il modello per ridurre i tempi di risposta iniziali.
    """
    if HUGGINGFACE_TOKEN:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
        payload = {"inputs": "ciao", "parameters": {"max_length": 10}}
        try:
            print("Pre-riscaldamento del modello...")
            async with httpx.AsyncClient() as client:
                await client.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=15)
            print("Modello pre-riscaldato con successo!")
        except Exception as e:
            print(f"Errore durante il pre-riscaldamento: {e}")

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

    # 2. Richiesta al modello Hugging Face
    if not HUGGINGFACE_TOKEN:
        return "Errore: Token Hugging Face mancante."

    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 10,  # Ridotto per velocizzare la risposta
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "do_sample": True,
        },
    }

    retry_attempts = 3  # Numero di tentativi in caso di errore
    for attempt in range(retry_attempts):
        try:
            # Invia la richiesta asincrona
            async with httpx.AsyncClient() as client:
                response = await client.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=15)
                response.raise_for_status()
                result = response.json()

                # 3. Estrarre e pulire il testo generato
                if isinstance(result, list) and len(result) > 0:
                    generated_text = clean_text(prompt, result[0].get("generated_text", ""))
                    return generated_text

            return "Errore nel generare la risposta dal modello esterno."

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                # Gestione dell'errore 503, tentativo di ritentare dopo una breve pausa
                print(f"Errore 503 ricevuto. Tentativo {attempt + 1} di {retry_attempts}.")
                time.sleep(5)  # Aspetta 5 secondi prima di ritentare
            else:
                print(f"Errore di stato HTTP: {e.response.status_code}")
                return f"Errore nel comunicare con il server di Hugging Face: {e.response.status_code}"
        except httpx.RequestError as e:
            print(f"Errore di richiesta al modello: {e}")
            return "Il server è occupato o lento. Riprova più tardi."

    # Se tutti i tentativi falliscono, restituisci un errore
    return "Errore nel generare la risposta dal modello esterno dopo diversi tentativi."

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
