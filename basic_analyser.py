import ollama
import requests
from io import BytesIO
from transformers import AutoTokenizer
from PyPDF2 import PdfReader
import torch
from sentence_transformers import SentenceTransformer
import json
import faiss
# ======== enabling gpu acceleration - cuz why the fuck not ======
# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ======== selecting our transformer. 
model_name = "all-MiniLM-L6-v2" # Or "sentence-transformers/allenai/specter2"
model = SentenceTransformer(model_name, device=device)

# ======== selecting our tokenizer. 
DEEPSEEK_TOKENIZER_NAME = "deepseek-ai/DeepSeek-R1"
deepseek_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_TOKENIZER_NAME)
print(f"Loaded tokenizer: {DEEPSEEK_TOKENIZER_NAME}")

# ======== Helper Functions ========
def download_pdf_to_memory(pdf_url):
    try:
        r = requests.get(pdf_url + '.pdf', timeout=30)
        if r.status_code == 200:
            return BytesIO(r.content)
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")
    return None



def get_relevant_urls(query_embedding):
    index = faiss.read_index('arxiv_faiss.index')
    D, I = index.search(query_embedding, 5)
    with open('arxiv_id_mapping.json','r') as f:
        mapping = json.load(f)
    relavent_papers = []
    for i in I[0]:
        relavent_papers.append(mapping[i]['id'])
    return [f"https://arxiv.org/pdf/{x}"for x in relavent_papers]

def get_paper_transcript(query_embedding):
    urls = get_relevant_urls(query_embedding)
    total_count = 0
    all_papers = ''
    for url in urls:
        reader = PdfReader(download_pdf_to_memory(url))
    text = "\n".join(page.extract_text() or '' for page in reader.pages)
    all_papers+=text
    total_count+=len(deepseek_tokenizer.encode(text))
    return all_papers



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


query =  "A new method for optimizing neural network training with limited data using meta-learning techniques."
query_embedding = model.encode([query]).astype('float32')
all_papers = get_paper_transcript(query_embedding)
response = ollama.chat(
    model=DEEPSEEK_MODEL_NAME,  # or 'deepseek-llm' if you're using the base LLM
    messages=[
        {"role": "system", "content": RAG_PROMPT_TEMPLATE.format(retrieved_papers_content=all_papers,user_query=query)}
    ]#,stream=True
)
print(response['message']['content'])