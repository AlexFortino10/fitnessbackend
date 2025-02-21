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
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("Errore: Token Hugging Face mancante.")
HUGGINGFACE_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HUGGINGFACE_CLIENT = InferenceClient(api_key=HUGGINGFACE_TOKEN)
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
        messages = [{"role": "user", "content": prompt}]
        
        response = HUGGINGFACE_CLIENT.chat.completions.create(
            model=HUGGINGFACE_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.1,
            top_p=0.3
        )
        
        if response.choices and len(response.choices) > 0:
            raw_text = response.choices[0].message["content"].strip()
            return clean_text(prompt, raw_text)
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
