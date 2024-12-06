@app.post("/generate")
async def generate_text(request: PromptRequest):
    prompt = request.prompt.strip().lower()
    print(f"Ricevuto prompt: '{prompt}'")

    if prompt in PREDEFINED_RESPONSES:
        print(f"Risposta predefinita trovata per il prompt: '{prompt}'")
        return {"response": PREDEFINED_RESPONSES[prompt]}

    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 30,
                "temperature": 0.7,
                "top_k": 30,
                "top_p": 0.9,
                "do_sample": True
            }
        }

        print("Invio della richiesta al servizio Hugging Face...")
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        if "generated_text" in result:
            generated_text = result["generated_text"]
            print(f"Risposta generata dal servizio Hugging Face: {generated_text}")
            return {"response": generated_text}
        else:
            print("Errore: Nessun testo generato dal servizio.")
            return {"response": "Errore nel generare la risposta dal modello esterno."}

    except requests.exceptions.HTTPError as http_err:
        print(f"Errore HTTP: {http_err}")
        return {"response": f"Errore nel generare la risposta: {http_err}"}
    except Exception as e:
        print(f"Errore generico: {e}")
        return {"response": "Errore imprevisto nel generare la risposta."}