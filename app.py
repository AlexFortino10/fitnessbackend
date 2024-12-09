import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
import re
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

# Configurazione OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
HTTP_CLIENT = httpx.AsyncClient(timeout=40)  # Timeout di 40 secondi
FALLBACK_RESPONSE = "Non riesco a rispondere in questo momento, ma possiamo riprovare!"

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def fetch_from_openai(prompt: str):
    if not OPENAI_API_KEY:
        raise ValueError("Errore: API Key di OpenAI mancante.")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": "gpt-4",  # Specifica il modello GPT-4
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,  # Limite di token nella risposta
        "temperature": 0.7,  # Controlla la casualità della risposta
        "top_p": 0.9,       # Nucleus sampling
    }
    try:
        response = await HTTP_CLIENT.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Solleva un'eccezione se la risposta è di errore
        return response.json()
    except httpx.RequestError as e:
        print(f"Errore nella richiesta HTTP: {e}")
        raise
    except httpx.HTTPStatusError as e:
        print(f"Errore HTTP {e.response.status_code}: {e.response.text}")
        raise
    except Exception as e:
        print(f"Errore durante la chiamata a OpenAI: {e}")
        raise

def clean_text(prompt: str, text: str) -> str:
    # Rimuove newline, caratteri speciali e spazi multipli
    text = re.sub(r"[\n\r*#\\]", " ", text)  # Rimuove i caratteri speciali
    text = re.sub(r"\s+", " ", text)  # Sostituisce spazi multipli con uno singolo
    return text.strip()

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip()
    print(f"Ricevuto prompt: '{prompt}'")

    # 1. Controllo risposte predefinite
    if prompt.lower() in PREDEFINED_RESPONSES:
        return PREDEFINED_RESPONSES[prompt.lower()]  # Restituisce solo il testo senza "response"

    try:
        # 2. Chiamata asincrona a OpenAI
        print("Invio richiesta a OpenAI...")
        result = await fetch_from_openai(prompt)
        if "choices" in result and len(result["choices"]) > 0:
            generated_text = clean_text(prompt, result["choices"][0]["message"]["content"])
            return generated_text  # Restituisce direttamente il testo generato
        return FALLBACK_RESPONSE
    except RetryError:
        print("Numero massimo di tentativi superato durante il recupero del modello.")
        return FALLBACK_RESPONSE
    except Exception as e:
        print(f"Errore: {e}")
        return FALLBACK_RESPONSE

@app.on_event("startup")
async def warm_up_model():
    try:
        await fetch_from_openai("Ciao")
        print("Modello riscaldato correttamente.")
    except Exception as e:
        print(f"Errore durante il warm-up: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    await HTTP_CLIENT.aclose()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)