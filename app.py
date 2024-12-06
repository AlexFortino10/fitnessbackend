import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

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

    # Chiamata al servizio Hugging Face per la generazione del testo
    try:
        # Verifica che il token di Hugging Face sia presente
        if not HUGGINGFACE_TOKEN:
            raise ValueError("Hugging Face Token non trovato.")

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

        print("Invio della richiesta al servizio Hugging Face...")

        # Fai la richiesta al servizio Hugging Face
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)

        # Logga la risposta di errore se presente
        if response.status_code != 200:
            print(f"Errore nella risposta: {response.status_code} - {response.text}")
            return {"response": f"Errore nel generare la risposta: {response.text}"}

        result = response.json()
        
        # Verifica la presenza del campo 'generated_text' nella risposta
        if "generated_text" in result:
            generated_text = result["generated_text"]
            print(f"Risposta generata dal servizio Hugging Face: {generated_text}")
            return {"response": generated_text}
        else:
            print("Errore: Nessun testo generato dal servizio.")
            return {"response": "Errore nel generare la risposta dal modello esterno."}

    except requests.exceptions.HTTPError as http_err:
        print(f"Errore HTTP: {http_err}")
        return {"response": f"Errore nel generare la risposta: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        print(f"Errore di richiesta: {req_err}")
        return {"response": "Errore nella richiesta al servizio esterno."}
    except ValueError as val_err:
        print(f"Errore: {val_err}")
        return {"response": f"Errore: {val_err}"}
    except Exception as e:
        print(f"Errore generico: {e}")
        return {"response": "Errore imprevisto nel generare la risposta."}

# Gestione dinamica della porta
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Railway assegna una porta tramite la variabile d'ambiente PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
