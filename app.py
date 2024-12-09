from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import re

app = FastAPI()

# Configurazione del modello
MODEL_NAME = "google/gemma-2-2b-it"  # Sostituisci con il modello desiderato
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    """
    Carica il modello durante l'avvio dell'applicazione.
    """
    global model, tokenizer
    print("Caricamento del modello in corso...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
        print("Modello caricato con successo!")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        raise HTTPException(status_code=500, detail="Errore durante il caricamento del modello.")

@app.post("/generate")
async def generate_text(prompt: str):
    """
    Genera testo utilizzando il modello locale.
    """
    global model, tokenizer
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Il modello non è stato caricato correttamente.")
    
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Il prompt non può essere vuoto.")

    # Pulizia del prompt
    prompt = prompt.strip()
    
    try:
        # Tokenizzazione
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generazione
        outputs = model.generate(
            inputs["input_ids"],
            max_length=50,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decodifica e pulizia della risposta
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = clean_response(prompt, response)

        return {"response": response}

    except Exception as e:
        print(f"Errore durante la generazione del testo: {e}")
        raise HTTPException(status_code=500, detail="Errore durante la generazione del testo.")

def clean_response(prompt: str, response: str) -> str:
    """
    Pulisce la risposta generata rimuovendo il prompt e caratteri indesiderati.
    """
    # Rimuove il prompt dalla risposta
    escaped_prompt = re.escape(prompt.strip())
    pattern = rf"^{escaped_prompt}\W*"  # Cerca il prompt all'inizio della risposta
    response = re.sub(pattern, "", response, flags=re.IGNORECASE).strip()

    # Rimuove caratteri indesiderati come \n o *
    response = " ".join(response.splitlines()).replace("*", "").strip()
    return response
