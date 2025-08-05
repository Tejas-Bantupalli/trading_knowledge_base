from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import requests
from io import BytesIO
import fitz  # PyMuPDF
import json

class ArxivRetrievalInput(BaseModel):
    """Input schema for ArxivRetrievalTool."""
    arxiv_id: str = Field(..., description="The ArXiv ID of the paper to retrieve")

class ArxivRetrievalTool(BaseTool):
    name: str = "ArXiv Retrieval Tool"
    description: str = (
        "Download and extract text from ArXiv papers using their ArXiv ID. "
        "Returns the full text content of the paper."
    )
    args_schema: Type[BaseModel] = ArxivRetrievalInput

    def _run(self, arxiv_id: str) -> str:
        """Download and extract text from an ArXiv paper."""
        try:
            # Clean the ArXiv ID (remove version suffix if present)
            clean_id = arxiv_id.split('v')[0]
            
            # Construct the PDF URL
            pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
            
            # Download the PDF
            response = requests.get(pdf_url, timeout=30)
            if response.status_code != 200:
                return f"Error: Failed to fetch PDF from {pdf_url} (status {response.status_code})"
            
            # Extract text from PDF
            pdf_data = BytesIO(response.content)
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            
            # Extract text from all pages
            text_content = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():  # Only add non-empty pages
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            
            # Return structured result
            result = {
                "arxiv_id": clean_id,
                "pdf_url": pdf_url,
                "text_length": len(text_content),
                "content": text_content  
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error retrieving paper {arxiv_id}: {str(e)}" 