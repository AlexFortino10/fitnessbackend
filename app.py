import os
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from tenacity import retry, wait_fixed, stop_after_attempt, RetryError
import re

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
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Facoltativo per GPT-2
HUGGINGFACE_MODEL = "gpt2"
HUGGINGFACE_CLIENT = InferenceClient(model=HUGGINGFACE_MODEL, token=HUGGINGFACE_TOKEN)
FALLBACK_RESPONSE = "Non riesco a rispondere in questo momento, ma possiamo riprovare!"

# Funzione per pulire il testo generato
def clean_text(prompt: str, text: str) -> str:
    text = re.sub(r"[\n\r*#\\]", " ", text)  # Rimuove newline e caratteri speciali
    text = re.sub(r"\s+", " ", text)  # Sostituisce spazi multipli con uno singolo
    escaped_prompt = re.escape(prompt.strip())  
    pattern = rf"^{escaped_prompt}\W*"  # Rimuove il prompt iniziale (case insensitive)
    text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def fetch_from_huggingface(prompt: str):
    try:
        response = HUGGINGFACE_CLIENT.text_generation(
            prompt,
            max_new_tokens=100,  # GPT-2 ha un limite inferiore rispetto ai modelli più grandi
            temperature=0.7,
            top_p=0.9
        )
        return clean_text(prompt, response) if response else FALLBACK_RESPONSE
    except Exception as e:
        print(f"Errore durante la chiamata a Hugging Face: {e}")
        raise

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip()
    print(f"Ricevuto prompt: '{prompt}'")

    # 1. Controllo risposte predefinite
    if prompt.lower() in PREDEFINED_RESPONSES:
        return PREDEFINED_RESPONSES[prompt.lower()]

    try:
        print("Invio richiesta a Hugging Face...")
        generated_text = await fetch_from_huggingface(prompt)
        return generated_text
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
