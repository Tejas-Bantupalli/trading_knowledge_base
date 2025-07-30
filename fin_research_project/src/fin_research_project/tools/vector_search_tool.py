from crewai.tools import BaseTool
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class VectorSearchInput(BaseModel):
    """Input schema for VectorSearchTool."""
    query: str = Field(..., description="The search query to find relevant papers")
    num_results: int = Field(default=5, description="Number of papers to retrieve")

class VectorSearchTool(BaseTool):
    name: str = "Vector Database Search Tool"
    description: str = (
        "Search the vector database to find relevant academic papers based on a query. "
        "Returns paper metadata and IDs for the most relevant papers."
    )
    args_schema: Type[BaseModel] = VectorSearchInput

    def __init__(self):
        super().__init__()
        # Initialize embedding model
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        # Load the FAISS index and mapping
        try:
            self.index = faiss.read_index('arxiv_faiss.index')
            self.mapping = self._load_ndjson_mapping('arxiv_id_mapping.jsonl')
        except Exception as e:
            print(f"Warning: Could not load FAISS index: {e}")
            self.index = None
            self.mapping = {}

    def _load_ndjson_mapping(self, file_path: str) -> Dict[str, Any]:
        """Load the paper ID mapping from JSONL file."""
        mapping = {}
        try:
            with open(file_path, 'r') as f:
                for idx, line in enumerate(f):
                    try:
                        obj = json.loads(line)
                        mapping[str(idx)] = obj
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Warning: Mapping file {file_path} not found")
        return mapping

    def _run(self, query: str, num_results: int = 5) -> str:
        """Search for relevant papers using vector similarity."""
        if not self.index or not self.mapping:
            return "Error: Vector database not properly initialized"
        
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode([query]).astype('float32')
            
            # Search the index
            D, I = self.index.search(query_embedding, num_results)
            
            # Get paper details
            results = []
            for i, (distance, idx) in enumerate(zip(D[0], I[0])):
                key = str(idx)
                if key in self.mapping:
                    paper_info = self.mapping[key]
                    results.append({
                        "rank": i + 1,
                        "arxiv_id": paper_info.get('id', 'Unknown'),
                        "title": paper_info.get('title', 'Unknown'),
                        "authors": paper_info.get('authors', 'Unknown'),
                        "abstract": paper_info.get('abstract', 'Unknown'),
                        "url": f"https://arxiv.org/pdf/{paper_info.get('id', '')}",
                        "similarity_score": float(1 - distance)  # Convert distance to similarity
                    })
            
            # Format results as JSON string
            return json.dumps({
                "query": query,
                "num_results": len(results),
                "papers": results
            }, indent=2)
            
        except Exception as e:
            return f"Error during vector search: {str(e)}" 