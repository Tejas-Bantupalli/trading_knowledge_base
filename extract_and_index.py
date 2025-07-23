import os
import json
import requests
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import re
from io import BytesIO

# Config
INDEX_FILE = 'arxiv_faiss.index'
MAPPING_FILE = 'arxiv_id_mapping.json'
JSON_FILE = 'arxiv_qfin_index.json'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

with open(JSON_FILE, 'r') as f:
    papers = json.load(f)

papers = papers[:10]

def download_pdf_to_memory(pdf_url):
    try:
        r = requests.get(pdf_url + '.pdf', timeout=30)
        if r.status_code == 200:
            return BytesIO(r.content)
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")
    return None

def extract_abstract_from_memory(pdf_bytes):
    try:
        reader = PdfReader(pdf_bytes)
        text = "\n".join(page.extract_text() or '' for page in reader.pages)
        match = re.search(r'(?i)abstract\s*([\s\S]{0,2000}?)(?:\n\s*\n|introduction|1\.|I\.|keywords|JEL|PACS)', text)
        if match:
            abstract = match.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract
    except Exception as e:
        print(f"Failed to extract abstract from PDF: {e}")
    return None

model = SentenceTransformer(EMBEDDING_MODEL)

embeddings = []
id_mapping = []

for paper in tqdm(papers, desc="Processing papers"):
    arxiv_id = paper['id']
    pdf_url = paper['pdf_url']

    pdf_bytes = download_pdf_to_memory(pdf_url)
    if not pdf_bytes:
        continue
    abstract = extract_abstract_from_memory(pdf_bytes)
    if not abstract or len(abstract) < 20:
        continue
    emb = model.encode(abstract)
    embeddings.append(emb)
    id_mapping.append({'id': arxiv_id, 'abstract': abstract})

if embeddings:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, INDEX_FILE)
    with open(MAPPING_FILE, 'w') as f:
        json.dump(id_mapping, f)
    print(f"Indexed {len(embeddings)} papers.")
else:
    print("No embeddings to index.") 