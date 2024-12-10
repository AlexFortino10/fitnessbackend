import os
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
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
if not HUGGINGFACE_TOKEN:
    raise ValueError("Errore: Token Hugging Face mancante.")
HUGGINGFACE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HUGGINGFACE_CLIENT = InferenceClient(api_key=HUGGINGFACE_TOKEN)
FALLBACK_RESPONSE = "Non riesco a rispondere in questo momento, ma possiamo riprovare!"

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def fetch_from_huggingface(prompt: str):
    try:
        # Creazione del messaggio di input
        messages = [{"role": "user", "content": prompt}]
        
        # Richiesta al modello
        response = HUGGINGFACE_CLIENT.chat.completions.create(
            model=HUGGINGFACE_MODEL,
            messages=messages,
            max_tokens=500
        )
        
        # Recupero del testo generato
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message["content"].strip()
        else:
            return FALLBACK_RESPONSE
    except Exception as e:
        print(f"Errore durante la chiamata a Hugging Face: {e}")
        raise

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip()
    print(f"Ricevuto prompt: '{prompt}'")

    # 1. Controllo risposte predefinite
    if prompt.lower() in PREDEFINED_RESPONSES:
        return PREDEFINED_RESPONSES[prompt.lower()]  # Restituisce solo il testo senza "response"

    try:
        # 2. Chiamata al modello
        print("Invio richiesta a Hugging Face...")
        generated_text = await fetch_from_huggingface(prompt)
        return generated_text  # Restituisce il testo generato
    except RetryError:
        print("Numero massimo di tentativi superato durante il recupero del modello.")
        return FALLBACK_RESPONSE
    except Exception as e:
        print(f"Errore: {e}")
        return FALLBACK_RESPONSE

@app.on_event("startup")
async def warm_up_model():
    try:
        test_message = "Ciao, come stai?"
        print(f"Warm-up con il messaggio: {test_message}")
        response = await fetch_from_huggingface(test_message)
        print(f"Warm-up completato: {response}")
    except Exception as e:
        print(f"Errore durante il warm-up: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
