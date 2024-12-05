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

device = "cuda" if torch.cuda.is_available() else "cpu"

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip().lower()
    print(f"Ricevuto prompt: '{prompt}'")

    # Risposta predefinita se disponibile
    if prompt in PREDEFINED_RESPONSES:
        print(f"Risposta predefinita trovata per il prompt: '{prompt}'")
        return {"response": PREDEFINED_RESPONSES[prompt]}

    # Caricamento dinamico del modello e tokenizer
    try:
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        print("Caricamento del tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=huggingface_token)
        print("Tokenizer caricato con successo.")

        print(f"Caricamento del modello su {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            token=huggingface_token
        )
        print("Modello caricato con successo.")

        # Usa il modello per generare una risposta
        print("Generazione della risposta con il modello...")
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
        print(f"Risposta generata: {response_text}")
        return {"response": response_text}

    except Exception as e:
        print(f"Errore durante la generazione: {e}")
        return {"response": "Errore nel generare la risposta."}

    finally:
        # Scarica il modello dalla memoria
        print("Scaricamento del modello per liberare memoria.")
        del model
        del tokenizer
        torch.cuda.empty_cache()  # Pulisci la cache se stai usando CUDA
