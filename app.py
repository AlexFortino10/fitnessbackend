import os
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from tenacity import retry, wait_fixed, stop_after_attempt, RetryError
import re

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

# Risposte predefinite per domande comuni
PREDEFINED_RESPONSES = {
    "ciao": "Ciao! Come posso aiutarti oggi?",
    "come stai?": "Sto bene, grazie! E tu?",
    "allenamento": "Inizia con 10 minuti di stretching per scaldarti bene.",
}

# Configurazione Hugging Face
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("Errore: Token Hugging Face mancante.")
    
HUGGINGFACE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HUGGINGFACE_CLIENT = InferenceClient(api_key=HUGGINGFACE_TOKEN)

FALLBACK_RESPONSE = "Non riesco a rispondere in questo momento, ma possiamo riprovare!"

# Pulizia del testo generato
def clean_text(prompt: str, text: str) -> str:
    text = re.sub(r"[\n\r*#\\]", " ", text)  # Rimuove caratteri speciali
    text = re.sub(r"\s+", " ", text).strip()  # Rimuove spazi multipli
    escaped_prompt = re.escape(prompt.strip())
    pattern = rf"^{escaped_prompt}\W*"  # Rimuove il prompt iniziale dalla risposta
    text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text

# Funzione per richiedere il completamento testuale al modello LLaMA
@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def fetch_from_huggingface(prompt: str):
    try:
        response = HUGGINGFACE_CLIENT.text_generation(
            model=HUGGINGFACE_MODEL,
            prompt=prompt,
            max_new_tokens=500,
            temperature=0.7,  # Controlla la creatività
            top_p=0.9,  # Controlla la diversità
        )

        if response:
            return clean_text(prompt, response)
        else:
            return FALLBACK_RESPONSE
    except Exception as e:
        print(f"Errore durante la chiamata a Hugging Face: {e}")
        raise

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip()
    print(f"Ricevuto prompt: '{prompt}'")

    # Controllo risposte predefinite
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
        print(f"Riscaldamento con il messaggio: {test_message}")
        response = await fetch_from_huggingface(test_message)
        print(f"Riscaldamento completato: {response}")
    except Exception as e:
        print(f"Errore durante il riscaldamento: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
