from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import google.generativeai as genai
import os
import json

class CriticInput(BaseModel):
    """Input schema for CriticTool."""
    content_to_review: str = Field(..., description="The content to review (summary, answer, etc.)")
    original_papers: str = Field(..., description="The original paper content for verification")
    user_query: str = Field(..., description="The original user query for context")
    review_type: str = Field(default="general", description="Type of review: 'summary', 'answer', or 'general'")

class CriticTool(BaseTool):
    name: str = "Critic Review Tool"
    description: str = (
        "Review and validate outputs from other agents for accuracy, completeness, and relevance. "
        "Flags hallucinations and identifies areas needing more information."
    )
    args_schema: Type[BaseModel] = CriticInput

    def __init__(self):
        super().__init__()
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")

    def _run(self, content_to_review: str, original_papers: str, user_query: str, review_type: str = "general") -> str:
        """Review content for accuracy and completeness."""
        try:
            # Create the review prompt based on type
            if review_type == "summary":
                prompt = self._create_summary_review_prompt(content_to_review, original_papers, user_query)
            elif review_type == "answer":
                prompt = self._create_answer_review_prompt(content_to_review, original_papers, user_query)
            else:
                prompt = self._create_general_review_prompt(content_to_review, original_papers, user_query)

            # Generate review using Gemini
            response = self.model.generate_content(prompt)
            
            # Structure the result
            result = {
                "review_type": review_type,
                "user_query": user_query,
                "review": response.text,
                "content_length": len(content_to_review),
                "papers_length": len(original_papers)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error during review: {str(e)}"

    def _create_summary_review_prompt(self, summary: str, original_papers: str, user_query: str) -> str:
        return f"""
You are an expert academic reviewer specializing in quantitative finance research.

TASK: Review a paper summary for accuracy, completeness, and relevance to the user's query.

USER QUERY: {user_query}

ORIGINAL PAPER CONTENT:
{original_papers[:30000]}

GENERATED SUMMARY:
{summary}

Please evaluate the summary and provide:

1. **Accuracy Assessment**: 
   - Are all claims in the summary supported by the original paper?
   - Are there any factual errors or misrepresentations?
   - Rate accuracy (High/Medium/Low)

2. **Completeness Check**:
   - Does the summary cover the key aspects relevant to the user's query?
   - Are important findings or methods missing?
   - Rate completeness (High/Medium/Low)

3. **Relevance Evaluation**:
   - How well does the summary address the user's specific question?
   - Is the focus appropriate given the query?
   - Rate relevance (High/Medium/Low)

4. **Hallucination Detection**:
   - Identify any information that appears to be fabricated or not supported by the source
   - List specific instances if found

5. **Overall Verdict**: 
   - APPROVED: Summary is accurate, complete, and relevant
   - NEEDS REVISION: Minor issues that can be fixed
   - FLAGGED: Significant problems requiring major revision

6. **Specific Feedback**: Provide detailed comments on what needs improvement
"""

    def _create_answer_review_prompt(self, answer: str, original_papers: str, user_query: str) -> str:
        return f"""
You are an expert academic reviewer specializing in quantitative finance research.

TASK: Review a query answer for factual consistency, completeness, and proper sourcing.

USER QUERY: {user_query}

ORIGINAL PAPER CONTENT:
{original_papers[:30000]}

GENERATED ANSWER:
{answer}

Please evaluate the answer and provide:

1. **Factual Consistency**:
   - Are all claims directly supported by the provided papers?
   - Are citations accurate and properly referenced?
   - Rate consistency (High/Medium/Low)

2. **Completeness Assessment**:
   - Does the answer fully address the user's question?
   - Are there gaps that require additional information?
   - Rate completeness (High/Medium/Low)

3. **Source Verification**:
   - Can all key points be traced back to specific parts of the papers?
   - Are there unsourced claims or assumptions?
   - List any unverified claims

4. **Hallucination Detection**:
   - Identify any information that appears to be fabricated
   - Flag any claims that go beyond the provided sources

5. **Overall Verdict**:
   - APPROVED: Answer is accurate and well-sourced
   - NEEDS REVISION: Minor sourcing or completeness issues
   - FLAGGED: Significant problems requiring major revision or additional sources

6. **Recommendations**: Suggest specific improvements or additional sources needed
"""

    def _create_general_review_prompt(self, content: str, original_papers: str, user_query: str) -> str:
        return f"""
You are an expert academic reviewer specializing in quantitative finance research.

TASK: General review of content for quality, accuracy, and relevance.

USER QUERY: {user_query}

ORIGINAL PAPER CONTENT:
{original_papers[:30000]}

CONTENT TO REVIEW:
{content}

Please provide a comprehensive review covering:

1. **Quality Assessment**: Overall quality and coherence of the content
2. **Accuracy Check**: Verification against source materials
3. **Relevance Evaluation**: How well it addresses the user's needs
4. **Completeness**: Whether all necessary information is included
5. **Recommendations**: Specific suggestions for improvement

Provide a clear verdict: APPROVED, NEEDS REVISION, or FLAGGED with detailed reasoning.
""" 