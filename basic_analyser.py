import google.generativeai as genai
import requests
from io import BytesIO
from PyPDF2 import PdfReader
import torch
from sentence_transformers import SentenceTransformer
import json
import faiss
import os

# ========= Gemini API Key Setup =========
os.environ["GOOGLE_API_KEY"] = "AIzaSyA1XsDHyHjIyA9dcE3YRfa23g6PnN5UyFk"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ======== Device Selection =========
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ======== Load embedding model (also includes tokenizer) =========
model_name = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name, device=device)

# ======== Helper Functions =========
def download_pdf_to_memory(pdf_url):
    try:
        r = requests.get(pdf_url + '.pdf', timeout=30)
        if r.status_code == 200:
            return BytesIO(r.content)
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")
    return None

def load_ndjson_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            try:
                obj = json.loads(line)
                mapping[str(idx)] = obj  
            except json.JSONDecodeError as e:
                print(f"Skipping line {idx} due to JSON error: {e}")
    return mapping

def get_relevant_urls(query_embedding):
    index = faiss.read_index('arxiv_faiss.index')
    D, I = index.search(query_embedding, 5)
    mapping = load_ndjson_mapping('arxiv_id_mapping.jsonl')

    relevant_urls = []
    for i in I[0]:
        key = str(i)
        if key in mapping:
            relevant_urls.append(f"https://arxiv.org/pdf/{mapping[key]['id']}")
        else:
            print(f"⚠️ Warning: ID {i} not found in mapping.")
    return relevant_urls


def get_paper_transcript(query_embedding):
    urls = get_relevant_urls(query_embedding)
    all_text = ""
    for url in urls:
        arxiv_id = url.split("/")[-1]  # Extract arXiv ID from the URL
        pdf_file = download_pdf_to_memory(url)
        if not pdf_file:
            continue
        try:
            reader = PdfReader(pdf_file)
            text = "\n".join(page.extract_text() or '' for page in reader.pages)
            # Append citation tag at the end of each paper
            paper_with_citation = f"{text}\n\n[CITATION: {arxiv_id}]\n\n"
            all_text += paper_with_citation
        except Exception as e:
            print(f"Error reading PDF {arxiv_id}: {e}")
    return all_text


# ======== Prompt Template =========
RAG_PROMPT_TEMPLATE = """SYSTEM: You are an expert academic research assistant specializing in Quantitative Finance. Your goal is to provide comprehensive and accurate answers to user questions. Base your answers *strictly* on the provided research paper excerpts. Do not use external knowledge or invent information. If the answer cannot be found in the provided context, clearly state that you do not have enough information from the given papers.

CONTEXT:
---
{retrieved_papers_content}
---

Present your findings in the following structured format:

1.  **Summary:** A concise overview of the answer derived from the papers.
2.  **Key Insights:** A bulleted list of significant findings or contributions. For each insight, cite the source paper(s) using their exact [ARXIV_ID] as found in the context.
3.  **Limitations/Future Work:** Identify any limitations of the research or directions for future work explicitly mentioned in the provided papers, if relevant to the query.

USER QUESTION: {user_query}

ANSWER:"""

# ======== Run Query and Generate Gemini Response =========
query = "How are LLMs used in finance"
query_embedding = embedding_model.encode([query]).astype('float32')
all_papers = get_paper_transcript(query_embedding)

prompt = RAG_PROMPT_TEMPLATE.format(
    retrieved_papers_content=all_papers,
    user_query=query
)

# Call Gemini Flash
model = genai.GenerativeModel(model_name="gemini-2.5-flash")
response = model.generate_content(prompt)

print(response.text)
