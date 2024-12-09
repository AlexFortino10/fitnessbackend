from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Caricamento del modello e del tokenizer
MODEL_NAME = "google/gemma-2-2b-it"  # Modifica con il tuo modello scaricato
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    """
    Carica il modello durante l'avvio dell'applicazione.
    """
    global model, tokenizer
    print("Caricamento del modello in corso...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print("Modello caricato con successo!")

@app.post("/generate")
async def generate_text(prompt: str):
    """
    Genera testo utilizzando il modello locale.
    """
    global model, tokenizer
    if not model or not tokenizer:
        return {"error": "Il modello non è stato caricato correttamente."}
    
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
    )
    
    # Decodifica
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
