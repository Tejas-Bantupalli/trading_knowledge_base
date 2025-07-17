# helpers.py

import os
import fitz  # PyMuPDF
from ollama import AsyncClient

client = AsyncClient(host='http://localhost:11434')  # default Ollama host

def extract_text_from_pdf(pdf_url: str) -> str:
    import requests
    from io import BytesIO

    response = requests.get(pdf_url)
    doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()
    return text

async def analyze_paper_with_ollama(text: str, model: str = "phi3") -> str:
    prompt = f"""
You're an expert quant researcher.

Given the following paper text, identify:

1. The domain of finance (traditional, crypto, derivatives, etc.)
2. Up to 5 key formulas, in LaTeX if possible.
3. A short use case or scenario for each formula.

Respond in JSON with keys: domain, formulas, use_cases.


Paper:
{text}
"""

    response = await client.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content'].strip()

def safe_parse_json(response: str) -> dict:
    import json
    try:
        return json.loads(response)
    except Exception:
        return {"error": "Invalid JSON", "raw_response": response}
