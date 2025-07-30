from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from .tools import (
    VectorSearchTool, 
    ArxivRetrievalTool, 
    SummarizationTool, 
    QueryAnsweringTool, 
    CriticTool, 
    OutputGenerationTool
)

@CrewBase
class QuantitativeResearchCrew():
    """Quantitative Finance Research Crew with 6 specialized agents"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # ========== AGENT DEFINITIONS ==========
    
    @agent
    def research_orchestrator(self) -> Agent:
        """Research Orchestrator Agent - oversees operations and coordinates other agents"""
        return Agent(
            role="Research Orchestrator",
            goal="Oversee quantitative finance research operations, coordinate agents, and ensure high-quality outputs",
            backstory="""You are a senior quantitative finance research coordinator with extensive experience 
            in managing complex research projects. You excel at interpreting user queries, delegating tasks 
            to specialized agents, and synthesizing findings into comprehensive reports. You maintain the 
            research flow and ensure all outputs meet quality standards.""",
            tools=[
                OutputGenerationTool()
            ],
            verbose=True,
            allow_delegation=True
        )

    @agent
    def paper_retrieval_agent(self) -> Agent:
        """Paper Retrieval Agent - finds relevant papers using vector search"""
        return Agent(
            role="Paper Retrieval Specialist",
            goal="Efficiently locate and retrieve relevant academic papers from the vector database",
            backstory="""You are an expert in academic paper discovery and retrieval. You have deep knowledge 
            of quantitative finance literature and excel at finding the most relevant papers for any given query. 
            You use advanced vector search techniques to identify papers with high relevance scores.""",
            tools=[
                VectorSearchTool(),
                ArxivRetrievalTool()
            ],
            verbose=True
        )

    @agent
    def summarization_agent(self) -> Agent:
        """Summarization Agent - creates concise summaries of research papers"""
        return Agent(
            role="Research Summarization Expert",
            goal="Create comprehensive and accurate summaries of research papers focused on user queries",
            backstory="""You are a skilled research analyst with expertise in distilling complex academic papers 
            into clear, concise summaries. You understand quantitative finance concepts deeply and can identify 
            the key methods, findings, and implications that are most relevant to specific research questions.""",
            tools=[
                SummarizationTool()
            ],
            verbose=True
        )

    @agent
    def query_answering_agent(self) -> Agent:
        """Query Answering Agent - extracts specific answers from papers using RAG"""
        return Agent(
            role="Query Answering Specialist",
            goal="Extract precise answers to user queries from research papers using advanced RAG techniques",
            backstory="""You are an expert in information extraction and question answering. You excel at 
            finding specific information within research papers and presenting it in a clear, structured format. 
            You use retrieval-augmented generation to ensure answers are grounded in the source material.""",
            tools=[
                QueryAnsweringTool()
            ],
            verbose=True
        )

    @agent
    def critic_agent(self) -> Agent:
        """Critic Agent - reviews and validates outputs for accuracy and completeness"""
        return Agent(
            role="Research Quality Critic",
            goal="Review and validate research outputs for accuracy, completeness, and relevance",
            backstory="""You are a rigorous academic reviewer with expertise in quantitative finance. 
            You excel at identifying factual errors, detecting hallucinations, and ensuring that all 
            claims are properly supported by source materials. You maintain high standards for research quality.""",
            tools=[
                CriticTool()
            ],
            verbose=True
        )

    @agent
    def web_search_agent(self) -> Agent:
        """Web Search Agent - performs external searches when needed (placeholder for future implementation)"""
        return Agent(
            role="Web Research Specialist",
            goal="Perform external web searches to supplement academic paper research when needed",
            backstory="""You are an expert in web research and information gathering. You can find 
            current information, news, and additional context that may not be available in academic papers. 
            You help bridge gaps in research by finding supplementary information from reliable sources.""",
            tools=[],  # Placeholder - will be implemented with Serper later
            verbose=True
        )

    # ========== TASK DEFINITIONS ==========
    
    @task
    def initial_paper_retrieval(self) -> Task:
        """Task 1: Retrieve relevant papers based on user query"""
        return Task(
            description="""
            Given the user's research query, find the most relevant academic papers from the vector database.
            
            Steps:
            1. Use vector search to find papers with high relevance scores
            2. Retrieve the top 5 most relevant papers
            3. Download and extract text from each paper
            4. Return structured data with paper metadata and content
            
            Focus on papers that directly address the user's specific question.
            """,
            agent=self.paper_retrieval_agent,
            expected_output="""
            A JSON object containing:
            - List of 5 most relevant papers with metadata (title, authors, arxiv_id, url, similarity_score)
            - Full text content of each paper
            - Relevance scores and ranking
            """
        )

    @task
    def paper_summarization(self) -> Task:
        """Task 2: Generate focused summaries of retrieved papers"""
        return Task(
            description="""
            Create comprehensive summaries of the retrieved papers, focusing on aspects relevant to the user's query.
            
            For each paper:
            1. Identify the key research question and methodology
            2. Extract main findings and contributions
            3. Highlight relevance to the user's specific question
            4. Note limitations and future work directions
            
            Ensure summaries are accurate, complete, and tailored to the research context.
            """,
            agent=self.summarization_agent,
            expected_output="""
            A structured summary for each paper including:
            - Key research question and methodology
            - Main findings and contributions
            - Relevance to user query
            - Limitations and future work
            - Practical implications
            """
        )

    @task
    def query_answering(self) -> Task:
        """Task 3: Extract specific answers to user queries from papers"""
        return Task(
            description="""
            Extract precise answers to the user's specific question from the retrieved papers.
            
            Use RAG techniques to:
            1. Search within paper content for relevant information
            2. Extract specific answers with proper citations
            3. Structure the response with summary, key insights, and limitations
            4. Provide confidence levels for the answers
            
            Ensure all claims are directly supported by the source papers.
            """,
            agent=self.query_answering_agent,
            expected_output="""
            A comprehensive answer including:
            - Summary of findings
            - Key insights with citations
            - Limitations and future work
            - Paper summaries
            - Confidence level assessment
            """
        )

    @task
    def quality_review(self) -> Task:
        """Task 4: Review outputs for accuracy and completeness"""
        return Task(
            description="""
            Review all generated outputs (summaries and answers) for quality, accuracy, and completeness.
            
            For each output:
            1. Verify factual accuracy against source materials
            2. Check for hallucinations or unsupported claims
            3. Assess completeness and relevance
            4. Identify areas needing additional information
            5. Provide specific feedback for improvements
            
            Maintain high standards for research quality and reliability.
            """,
            agent=self.critic_agent,
            expected_output="""
            A detailed review report including:
            - Accuracy assessment (High/Medium/Low)
            - Completeness evaluation
            - Hallucination detection
            - Specific feedback and recommendations
            - Overall verdict (Approved/Needs Revision/Flagged)
            """
        )

    @task
    def report_generation(self) -> Task:
        """Task 5: Generate final research report and knowledge graph"""
        return Task(
            description="""
            Compile all research findings into comprehensive outputs.
            
            Generate:
            1. A detailed findings.md report with all research results
            2. A knowledge graph showing relationships between papers and concepts
            3. Executive summary of key insights
            4. Recommendations for further research
            
            Ensure the outputs are well-structured, comprehensive, and actionable.
            """,
            agent=self.research_orchestrator,
            expected_output="""
            Two main outputs:
            1. findings.md file with comprehensive research report
            2. knowledge_graph.json with structured relationship data
            Both files should be saved to the project directory.
            """
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Quantitative Research Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True  # Enable memory for conversation context
        ) 