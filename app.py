def clean_text(text: str) -> str:
    """
    Rimuove caratteri indesiderati come '\n' o altri caratteri non leggibili.
    """
    return " ".join(text.splitlines()).strip()

@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip().lower()
    print(f"Ricevuto prompt: '{prompt}'")

    # Risposta predefinita
    if prompt in PREDEFINED_RESPONSES:
        print(f"Risposta predefinita trovata per il prompt: '{prompt}'")
        return clean_text(PREDEFINED_RESPONSES[prompt])

    # Controllo token
    if not HUGGINGFACE_TOKEN:
        return clean_text("Errore: Token Hugging Face mancante. Controlla la configurazione.")

    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 10,
            "temperature": 0.1,
            "top_k": 30,
            "top_p": 0.9,
            "do_sample": True
        }
    }

    # Retry per gestire errori temporanei
    max_retries = 3
    retry_delay = 5  # secondi

    for attempt in range(max_retries):
        try:
            print("Invio della richiesta al servizio Hugging Face...")
            response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
            response.raise_for_status()  # Lancia un'eccezione se lo stato HTTP è diverso da 200

            result = response.json()
            print(f"Risultato completo: {result}")

            # Estrarre il testo generato
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                print(f"Risposta generata: {generated_text}")
                return clean_text(generated_text)

            print("Errore: Nessun testo generato.")
            return clean_text("Errore nel generare la risposta dal modello esterno.")

        except requests.exceptions.RequestException as e:
            print(f"Errore di richiesta: {e}")
            if attempt < max_retries - 1:
                print(f"Retry tra {retry_delay} secondi...")
                time.sleep(retry_delay)
            else:
                return clean_text(f"Errore imprevisto: {e}")

    # Esauriti i tentativi
    return clean_text("Errore: Non è stato possibile ottenere una risposta dal modello esterno.")
