# helpers.py

import os
import requests
import fitz  # PyMuPDF
from io import BytesIO
import google.generativeai as genai
import json 
# --- Configure Gemini ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is not set. Export it in your terminal or .zshrc.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-pro")


# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch PDF from {pdf_url} (status {response.status_code})")
    pdf_data = BytesIO(response.content)
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    return "".join([page.get_text() for page in doc])


# --- Gemini Paper Analysis ---
def analyze_paper_with_gemini(text_chunk: str) -> str:
    prompt = f"""
You're an expert quant researcher.

Given the following paper text, identify:

1. The domain of finance (traditional, crypto, derivatives, etc.)
2. Up to 5 key formulas, in LaTeX if possible.
3. A short use case or scenario for each formula.

Respond in JSON with keys: domain, formulas, use_cases.

Paper text:
\"\"\"
{text_chunk[:4000]}
\"\"\"
"""
    try:
        print("🔍 Sending prompt to Gemini...")
        response = model.generate_content(prompt)
        print("✅ Gemini response received.")
        print("📤 Raw response preview:", response.text[:300])  # peek at first 300 chars
        return response.text
    except Exception as e:
        print("❌ Gemini API call failed:", e)
        raise
def safe_parse_json(text):
    if text.startswith("```json"):
        text = text.strip("```json").strip()
    elif text.startswith("```"):
        text = text.strip("```").strip()
    try:
        return json.loads(text)
    except Exception as e:
        return {
            "error": "Invalid JSON",
            "raw_response": text[:1000]  # log only part of the raw response
        }