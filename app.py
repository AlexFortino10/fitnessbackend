import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import psutil

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

PREDEFINED_RESPONSES = {
    "ciao": "Ciao! Come posso aiutarti oggi?",
    "come stai?": "Sto bene, grazie! E tu?",
    "allenamento": "Inizia con 10 minuti di stretching per scaldarti bene."
}

# Variabili globali per modello e tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def check_memory_usage(step: str):
    """Funzione per monitorare la memoria disponibile e utilizzata."""
    memory = psutil.virtual_memory()
    print(f"[{step}] Memoria disponibile: {memory.available / (1024 * 1024):.2f} MB")
    print(f"[{step}] Memoria utilizzata: {memory.used / (1024 * 1024):.2f} MB")
    print(f"[{step}] Percentuale di utilizzo memoria: {memory.percent}%")

@app.on_event("startup")
def load_model():
    global model, tokenizer

    try:
        # Ottieni il token di accesso dalle variabili d'ambiente
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        if huggingface_token is None:
            raise ValueError("Token di autenticazione Hugging Face non trovato. Impostare la variabile d'ambiente 'HUGGINGFACE_TOKEN'.")

        print(f"Token Hugging Face trovato: {huggingface_token[:5]}...")  # Mostra solo una parte del token per sicurezza

        # Monitoraggio memoria prima di caricare il modello
        check_memory_usage("Prima del caricamento del tokenizer")

        # Carica il tokenizer con il token di accesso
        print("Caricamento del tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=huggingface_token)
        print("Tokenizer caricato con successo.")

        # Monitoraggio memoria prima di caricare il modello
        check_memory_usage("Prima del caricamento del modello")

        # Carica il modello con distribuzione automatica tra CPU/GPU e precisione ridotta
        print(f"Caricamento del modello su {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            token=huggingface_token
        )
        print("Modello caricato con successo.")

        # Monitoraggio memoria dopo il caricamento del modello
        check_memory_usage("Dopo il caricamento del modello")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip().lower()
    print(f"Ricevuto prompt: '{prompt}'")

    # Risposta predefinita se disponibile
    if prompt in PREDEFINED_RESPONSES:
        print(f"Risposta predefinita trovata per il prompt: '{prompt}'")
        return {"response": PREDEFINED_RESPONSES[prompt]}

    # Usa il modello per generare una risposta
    print("Generazione della risposta con il modello...")
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
            print(f"Input tokenizzato: {inputs}")
            output = model.generate(
                **inputs,
                max_length=30,
                top_k=30,
                top_p=0.9,
                temperature=0.7,
                do_sample=True
            )
        response_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Risposta generata: {response_text}")
        return {"response": response_text}
    except Exception as e:
        print(f"Errore durante la generazione della risposta: {e}")
        return {"response": "Si è verificato un errore durante la generazione della risposta."}

@app.on_event("shutdown")
async def shutdown_event():
    global model, tokenizer
    print("Rilasciando il modello...")
    model = None
    tokenizer = None
    torch.cuda.empty_cache()
    check_memory_usage("Dopo il rilascio del modello")
