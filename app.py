import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

@app.on_event("startup")
def load_model():
    global model, tokenizer

    # Ottieni il token di accesso dalle variabili d'ambiente
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    if huggingface_token is None:
        raise ValueError("Token di autenticazione Hugging Face non trovato. Impostare la variabile d'ambiente 'HUGGINGFACE_TOKEN'.")

    # Carica il tokenizer con il token di accesso
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=huggingface_token)

    # Carica il modello con distribuzione automatica tra CPU/GPU e precisione ridotta
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",  # Distribuzione automatica su CPU/GPU
        use_auth_token=huggingface_token  # Passa il token di autenticazione
    )
    print("Modello caricato con successo.")

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip().lower()

    # Risposta predefinita se disponibile
    if prompt in PREDEFINED_RESPONSES:
        return {"response": PREDEFINED_RESPONSES[prompt]}

    # Usa il modello per generare una risposta
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
        output = model.generate(
            **inputs,
            max_length=30,
            top_k=30,
            top_p=0.9,
            temperature=0.7,
            do_sample=True
        )

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response_text}
