from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import google.generativeai as genai
import os
import json

class QueryAnsweringInput(BaseModel):
    """Input schema for QueryAnsweringTool."""
    paper_text: str = Field(..., description="The text content of the paper(s) to search")
    user_query: str = Field(..., description="The specific question to answer")
    search_context: str = Field(default="", description="Additional context about the search")

class QueryAnsweringTool(BaseTool):
    name: str = "Query Answering Tool"
    description: str = (
        "Extract specific answers from research papers to match user queries using RAG techniques. "
        "Performs localized search within papers to find relevant information."
    )
    args_schema: Type[BaseModel] = QueryAnsweringInput

    def __init__(self):
        super().__init__()
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")

    def _run(self, paper_text: str, user_query: str, search_context: str = "") -> str:
        """Extract answers from papers using RAG approach."""
        try:
            # Create the RAG prompt template
            prompt = f"""
SYSTEM: You are an expert academic research assistant specializing in Quantitative Finance. 
Your goal is to provide comprehensive and accurate answers to user questions. 
Base your answers *strictly* on the provided research paper excerpts. 
Do not use external knowledge or invent information. 
If the answer cannot be found in the provided context, clearly state that you do not have enough information from the given papers.

CONTEXT:
---
{paper_text[:40000]}  # Limit text length for API call
---

SEARCH CONTEXT: {search_context if search_context else "General quantitative finance research"}

Present your findings in the following structured format:

1. **Summary:** A concise overview of the answer derived from the papers.
2. **Key Insights:** A bulleted list of significant findings or contributions. For each insight, cite the source paper(s) using their exact [ARXIV_ID] as found in the context.
3. **Limitations/Future Work:** Identify any limitations of the research or directions for future work explicitly mentioned in the provided papers, if relevant to the query.
4. **Paper Summaries:** Create a concise 1-line summary for each paper presented in the context.
5. **Confidence Level:** Rate your confidence in the answer (High/Medium/Low) and explain why.

USER QUESTION: {user_query}

ANSWER:"""

            # Generate answer using Gemini
            response = self.model.generate_content(prompt)
            
            # Structure the result
            result = {
                "user_query": user_query,
                "search_context": search_context,
                "answer": response.text,
                "paper_length": len(paper_text),
                "answer_length": len(response.text)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error generating answer: {str(e)}" 