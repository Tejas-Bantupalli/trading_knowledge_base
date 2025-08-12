from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import google.generativeai as genai
import os
import json

class SummarizationInput(BaseModel):
    """Input schema for SummarizationTool."""
    paper_text: str = Field(..., description="The full text of the paper to summarize")
    user_query: str = Field(..., description="The user's original query to focus the summary")
    research_context: str = Field(default="", description="Overall research context and direction")

class SummarizationTool(BaseTool):
    name: str = "Paper Summarization Tool"
    description: str = (
        "Generate concise summaries of research papers based on user queries and research context. "
        "Focuses on key methods, findings, and conclusions relevant to the user's needs."
    )
    args_schema: Type[BaseModel] = SummarizationInput

    def __init__(self):
        super().__init__()
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")

    def _run(self, paper_text: str, user_query: str, research_context: str = "") -> str:
        """Generate a focused summary of the paper."""
        try:
            # Create the summarization prompt
            prompt = f"""
You are an expert quantitative finance researcher tasked with summarizing academic papers.

USER QUERY: {user_query}

RESEARCH CONTEXT: {research_context if research_context else "General quantitative finance research"}

PAPER TEXT:
{paper_text[:30000]}  # Limit text length for API call

Please provide a comprehensive summary that includes:

1. **Key Research Question**: What problem does this paper address?
2. **Methodology**: What approach or methods does the paper use?
3. **Key Findings**: What are the main results and contributions?
4. **Relevance to Query**: How does this paper relate to the user's specific question?
5. **Limitations**: What are the paper's limitations or areas for improvement?
6. **Practical Implications**: What are the real-world applications or implications?

Focus on aspects that are most relevant to the user's query and the overall research direction.
Keep the summary concise but comprehensive, highlighting the most important insights.
"""

            # Generate summary using Gemini
            response = self.model.generate_content(prompt)
            
            # Structure the result
            result = {
                "user_query": user_query,
                "research_context": research_context,
                "summary": response.text,
                "paper_length": len(paper_text),
                "summary_length": len(response.text)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error generating summary: {str(e)}" 