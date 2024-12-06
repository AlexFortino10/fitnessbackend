import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

PREDEFINED_RESPONSES = {
    "ciao": "Ciao! Come posso aiutarti oggi?",
    "come stai?": "Sto bene, grazie! E tu?",
    "allenamento": "Inizia con 10 minuti di stretching per scaldarti bene."
}

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip().lower()
    print(f"Ricevuto prompt: '{prompt}'")

    # Risposta predefinita se disponibile
    if prompt in PREDEFINED_RESPONSES:
        print(f"Risposta predefinita trovata per il prompt: '{prompt}'")
        return {"response": PREDEFINED_RESPONSES[prompt]}

    # Verifica che il token di Hugging Face sia presente
    if not HUGGINGFACE_TOKEN:
        return {"response": "Errore: Token Hugging Face mancante. Controlla la configurazione."}

    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 30,
            "temperature": 0.7,
            "top_k": 30,
            "top_p": 0.9,
            "do_sample": True
        }
    }

    # Retry per gestire il caricamento del modello
    max_retries = 5
    retry_delay = 10  # secondi

    for attempt in range(max_retries):
        try:
            print("Invio della richiesta al servizio Hugging Face...")
            response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
            
            if response.status_code == 503:  # Modello in caricamento
                error_details = response.json()
                estimated_time = error_details.get("estimated_time", retry_delay)
                print(f"Il modello è in caricamento, attendo {estimated_time} secondi prima di ritentare...")
                time.sleep(estimated_time + 2)  # Aggiunge un buffer extra
                continue

            # Se la risposta è positiva, elaborala
            response.raise_for_status()
            result = response.json()

            if "generated_text" in result:
                generated_text = result["generated_text"]
                print(f"Risposta generata dal servizio Hugging Face: {generated_text}")
                return {"response": generated_text}
            else:
                print("Errore: Nessun testo generato dal servizio.")
                return {"response": "Errore nel generare la risposta dal modello esterno."}

        except requests.exceptions.HTTPError as http_err:
            print(f"Errore HTTP: {http_err}")
            return {"response": f"Errore HTTP: {http_err}"}
        except Exception as e:
            print(f"Errore generico: {e}")
            return {"response": f"Errore imprevisto: {e}"}

    # Se esaurisce i tentativi
    print("Il modello non è stato caricato dopo diversi tentativi.")
    return {"response": "Errore: Il modello non è riuscito a caricarsi dopo diversi tentativi."}

# Gestione dinamica della porta
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Railway assegna una porta tramite la variabile d'ambiente PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
