import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# Risposte predefinite
PREDEFINED_RESPONSES = {
    "ciao": "Ciao! Come posso aiutarti oggi?",
    "come stai?": "Sto bene, grazie! E tu?",
    "allenamento": "Inizia con 10 minuti di stretching per scaldarti bene."
}

# URL del modello e token
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Assicurati che il token sia impostato nelle variabili d'ambiente

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip().lower()
    print(f"Ricevuto prompt: '{prompt}'")

    # Risposta predefinita se disponibile
    if prompt in PREDEFINED_RESPONSES:
        print(f"Risposta predefinita trovata per il prompt: '{prompt}'")
        return {"response": PREDEFINED_RESPONSES[prompt]}

    # Chiamata al servizio esterno per l'inferenza
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
        payload = {"inputs": prompt}

        print("Invio della richiesta al servizio Hugging Face...")
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Genera un'eccezione per HTTP error

        # Estrai il testo generato dalla risposta JSON
        result = response.json()
        generated_text = result.get("generated_text", "Errore: nessun testo generato.")

        print(f"Risposta generata dal servizio Hugging Face: {generated_text}")
        return {"response": generated_text}

    except Exception as e:
        print(f"Errore durante la generazione con il servizio esterno: {e}")
        return {"response": "Errore nel generare la risposta."}

# Gestione dinamica della porta
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Railway assegna una porta tramite la variabile d'ambiente PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
