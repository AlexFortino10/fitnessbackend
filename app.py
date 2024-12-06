import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Classe per la richiesta
class PromptRequest(BaseModel):
    prompt: str

# Risposte predefinite
PREDEFINED_RESPONSES = {
    "ciao": "Ciao! Come posso aiutarti oggi?",
    "come stai?": "Sto bene, grazie! E tu?",
    "allenamento": "Inizia con 10 minuti di stretching per scaldarti bene."
}

# Configurazione dell'Inference API
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Assicurati che il token sia impostato come variabile d'ambiente

def query_huggingface_api(prompt):
    """Funzione per inviare una richiesta all'Inference API di Hugging Face."""
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 30,
            "temperature": 0.7,
            "top_k": 30,
            "top_p": 0.9,
            "do_sample": True
        },
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Solleva un errore se la risposta non è OK
        result = response.json()
        
        # Controlla se il modello ha generato un testo
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return "Errore: Nessun testo generato dal modello."
    except requests.exceptions.RequestException as e:
        return f"Errore nella richiesta al modello: {str(e)}"

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip().lower()
    print(f"Ricevuto prompt: '{prompt}'")

    # Controlla se esiste una risposta predefinita
    if prompt in PREDEFINED_RESPONSES:
        print(f"Risposta predefinita trovata per il prompt: '{prompt}'")
        return {"response": PREDEFINED_RESPONSES[prompt]}

    # Verifica che il token sia presente
    if not HUGGINGFACE_TOKEN:
        print("Errore: Token Hugging Face mancante.")
        return {"response": "Errore: Token Hugging Face mancante. Controlla la configurazione."}

    # Genera il testo usando l'API di Hugging Face
    print("Invio della richiesta al modello Hugging Face...")
    response_text = query_huggingface_api(prompt)
    print(f"Risposta generata: {response_text}")
    return {"response": response_text}

# Gestione dinamica della porta
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Railway assegna una porta tramite la variabile d'ambiente PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
