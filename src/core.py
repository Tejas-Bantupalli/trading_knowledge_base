"""
Core Trading Knowledge Base System

This module provides a unified interface for all trading knowledge base functionality,
integrating crew-based research, memory management, and analysis tools.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from crewai import Agent, Crew, Process, Task

# Import tools and memory systems
try:
    from .memory.stm import STM
    from .memory.ltm import LTM
    from .tools.vector_search_tool import VectorSearchTool
    from .tools.arxiv_retrieval_tool import ArxivRetrievalTool
    from .tools.summarization_tool import SummarizationTool
    from .tools.query_answering_tool import QueryAnsweringTool
    from .tools.critic_tool import CriticTool
    from .tools.output_generation_tool import OutputGenerationTool
except ImportError:
    # Fallback for direct execution
    from memory.stm import STM
    from memory.ltm import LTM
    from tools.vector_search_tool import VectorSearchTool
    from tools.arxiv_retrieval_tool import ArxivRetrievalTool
    from tools.summarization_tool import SummarizationTool
    from tools.query_answering_tool import QueryAnsweringTool
    from tools.critic_tool import CriticTool
    from tools.output_generation_tool import OutputGenerationTool


class TradingKnowledgeBase:
    """
    Unified Trading Knowledge Base System
    
    This class integrates all functionality:
    - Crew-based research orchestration
    - Memory management (STM/LTM)
    - Vector search and retrieval
    - Paper analysis and summarization
    - Knowledge graph generation
    """
    
    def __init__(self, data_dir: str = "data", enable_memory: bool = True):
        """
        Initialize the Trading Knowledge Base.
        
        Args:
            data_dir: Directory for data storage
            enable_memory: Whether to enable memory management
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize memory systems
        self.enable_memory = enable_memory
        if enable_memory:
            self.stm = STM()
            try:
                self.ltm = LTM(str(self.data_dir))
            except Exception as e:
                print(f"Warning: LTM initialization failed: {e}")
                print("LTM features will be disabled")
                self.ltm = None
        
        # Research session data
        self.current_session = {}
        
        # Initialize agents
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize the research agents."""
        # Initialize tools
        tools = {}
        
        try:
            tools['output_generation'] = OutputGenerationTool()
        except Exception:
            tools['output_generation'] = None
        
        try:
            tools['vector_search'] = VectorSearchTool()
        except Exception:
            tools['vector_search'] = None
        
        try:
            tools['arxiv_retrieval'] = ArxivRetrievalTool()
        except Exception:
            tools['arxiv_retrieval'] = None
        
        try:
            tools['summarization'] = SummarizationTool()
        except Exception:
            tools['summarization'] = None
        
        try:
            tools['query_answering'] = QueryAnsweringTool()
        except Exception:
            tools['query_answering'] = None
        
        try:
            tools['critic'] = CriticTool()
        except Exception:
            tools['critic'] = None
        
        # Store tools for later use
        self.tools = tools
        
        # Initialize agents with available tools
        self.research_orchestrator = Agent(
            role="Research Orchestrator",
            goal="Oversee quantitative finance research operations, coordinate agents, and ensure high-quality outputs",
            backstory="""You are a senior quantitative finance research coordinator with extensive experience 
            in managing complex research projects. You excel at interpreting user queries, delegating tasks 
            to specialized agents, and synthesizing findings into comprehensive reports. You maintain the 
            research flow and ensure all outputs meet quality standards.""",
            tools=[tools['output_generation']] if tools['output_generation'] else [],
            verbose=True,
            allow_delegation=True
        )

        self.paper_retrieval_agent = Agent(
            role="Paper Retrieval Specialist",
            goal="Efficiently locate and retrieve relevant academic papers from the vector database",
            backstory="""You are an expert in academic paper discovery and retrieval. You have deep knowledge 
            of quantitative finance literature and excel at finding the most relevant papers for any given query. 
            You use advanced vector search techniques to identify papers with high relevance scores.""",
            tools=[t for t in [tools['vector_search'], tools['arxiv_retrieval']] if t is not None],
            verbose=True
        )

        self.summarization_agent = Agent(
            role="Research Summarization Expert",
            goal="Create comprehensive and accurate summaries of research papers focused on user queries",
            backstory="""You are a skilled research analyst with expertise in distilling complex academic papers 
            into clear, concise summaries. You understand quantitative finance concepts deeply and can identify 
            the key methods, findings, and implications that are most relevant to specific research questions.""",
            tools=[tools['summarization']] if tools['summarization'] else [],
            verbose=True
        )

        self.query_answering_agent = Agent(
            role="Query Answering Specialist",
            goal="Extract precise answers to user queries from research papers using advanced RAG techniques",
            backstory="""You are an expert in information extraction and question answering. You excel at 
            finding specific information within research papers and presenting it in a clear, structured format. 
            You use retrieval-augmented generation to ensure answers are grounded in the source material.""",
            tools=[tools['query_answering']] if tools['query_answering'] else [],
            verbose=True
        )

        self.critic_agent = Agent(
            role="Research Quality Critic",
            goal="Review and validate research outputs for accuracy, completeness, and relevance",
            backstory="""You are a rigorous academic reviewer with expertise in quantitative finance. 
            You excel at identifying factual errors, detecting hallucinations, and ensuring that all 
            claims are properly supported by source materials. You maintain high standards for research quality.""",
            tools=[tools['critic']] if tools['critic'] else [],
            verbose=True
        )
        
        # Check tool availability
        working_tools = sum(1 for tool in tools.values() if tool is not None)
        if working_tools < len(tools):
            print(f"⚠️  {len(tools) - working_tools} tools failed to initialize. Set GEMINI_API_KEY for full functionality.")
    
    def crew(self) -> Crew:
        """
        Create and return a CrewAI crew for research execution.
        
        Returns:
            Configured CrewAI crew instance
        """
        # Define research tasks
        retrieval_task = Task(
            description=f"Search for and retrieve relevant academic papers on the research topic",
            agent=self.paper_retrieval_agent,
            expected_output="List of relevant papers with relevance scores and brief descriptions"
        )
        
        summarization_task = Task(
            description=f"Analyze and summarize the retrieved papers, focusing on key findings and methodologies",
            agent=self.summarization_agent,
            expected_output="Comprehensive summaries of each paper highlighting key insights"
        )
        
        analysis_task = Task(
            description=f"Extract specific answers and insights from the papers based on the research query",
            agent=self.query_answering_agent,
            expected_output="Detailed analysis with specific answers and supporting evidence from papers"
        )
        
        review_task = Task(
            description=f"Review the research outputs for accuracy, completeness, and relevance",
            agent=self.critic_agent,
            expected_output="Quality assessment and validation of research findings"
        )
        
        synthesis_task = Task(
            description=f"Compile all research findings into a comprehensive report and knowledge graph",
            agent=self.research_orchestrator,
            expected_output="Final research report and structured knowledge representation"
        )
        
        # Create and return the crew
        return Crew(
            agents=[
                self.paper_retrieval_agent,
                self.summarization_agent,
                self.query_answering_agent,
                self.critic_agent,
                self.research_orchestrator
            ],
            tasks=[
                retrieval_task,
                summarization_task,
                analysis_task,
                review_task,
                synthesis_task
            ],
            process=Process.sequential,
            verbose=True
        )
    
    # ========== PUBLIC INTERFACE METHODS ==========
    
    def research(self, query: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a given query using CrewAI workflow.
        
        Args:
            query: Research query to investigate
            output_dir: Directory to save outputs (defaults to data_dir)
            
        Returns:
            Dictionary containing research results and metadata
        """
        if output_dir is None:
            output_dir = self.data_dir
        
        # Store session data
        self.current_session = {
            'user_query': query,
            'start_time': datetime.now().isoformat(),
            'output_dir': output_dir
        }
        
        # Add to STM if memory is enabled
        if self.enable_memory:
            self.stm.update_context('current_research_query', query)
        
        try:
            # Create and execute the crew workflow
            crew = self.crew()
            
            # Execute the research workflow
            results = crew.kickoff(inputs={'topic': query})
            
            # Store results in session
            self.current_session['results'] = results
            self.current_session['end_time'] = datetime.now().isoformat()
            
            # Save session data
            session_file = Path(output_dir) / f"research_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(session_file, 'w') as f:
                json.dump(self.current_session, f, indent=2, default=str)
            
            # Update memory with results
            if self.enable_memory:
                self.stm.update_context('last_research_results', results)
                self.stm.update_context('last_research_session', str(session_file))
            
            return {
                'status': 'success',
                'query': query,
                'results': results,
                'session_file': str(session_file),
                'memory_enabled': self.enable_memory,
                'message': 'Research completed successfully using CrewAI workflow'
            }
            
        except Exception as e:
            error_msg = f"Research workflow failed: {str(e)}"
            print(f"Error: {error_msg}")
            
            # Store error in session
            self.current_session['error'] = error_msg
            self.current_session['end_time'] = datetime.now().isoformat()
            
            # Save error session
            session_file = Path(output_dir) / f"research_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(session_file, 'w') as f:
                json.dump(self.current_session, f, indent=2, default=str)
            
            return {
                'status': 'error',
                'query': query,
                'error': error_msg,
                'session_file': str(session_file),
                'memory_enabled': self.enable_memory
            }
    
    def quick_analysis(self, query: str) -> Dict[str, Any]:
        """
        Perform a quick analysis without full crew execution.
        
        Args:
            query: Research query
            
        Returns:
            Quick analysis results
        """
        # Use vector search tool directly
        try:
            if 'vector_search' not in self.tools or self.tools['vector_search'] is None:
                return {
                    'status': 'error',
                    'query': query,
                    'error': 'VectorSearchTool not available. Check tool initialization.',
                    'memory_enabled': self.enable_memory
                }
            
            vector_tool = self.tools['vector_search']
            # Use the proper input schema for the tool
            result = vector_tool._run(query=query, num_results=5)
            
            return {
                'status': 'success',
                'query': query,
                'type': 'quick_analysis',
                'results': result,
                'memory_enabled': self.enable_memory
            }
        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': str(e),
                'memory_enabled': self.enable_memory
            }
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory system status."""
        if not self.enable_memory:
            return {'memory_enabled': False}
        
        return {
            'memory_enabled': True,
            'stm_entries': len(self.stm.conversation_buffer),
            'ltm_available': self.ltm is not None,
            'current_context': self.stm.get_context()
        }
    
    def clear_memory(self) -> None:
        """Clear all memory systems."""
        if self.enable_memory:
            self.stm.clear_context()
            self.stm.conversation_buffer.clear()
            print("Memory cleared successfully.")
        else:
            print("Memory system is not enabled.")
    
    def get_research_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of recent research sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of research session data
        """
        if not self.enable_memory:
            return []
        
        # Get recent context entries
        context = self.stm.get_context()
        history = []
        
        # Look for research-related context
        if 'last_research_results' in context:
            history.append({
                'type': 'research_results',
                'data': context['last_research_results'],
                'timestamp': context.get('last_research_session', 'unknown')
            })
        
        if 'current_research_query' in context:
            history.append({
                'type': 'current_query',
                'data': context['current_research_query'],
                'timestamp': 'current'
            })
        
        # Add conversation history
        for entry in self.stm.conversation_buffer[-limit:]:
            history.append({
                'type': 'conversation',
                'user_input': entry.get('user_input', ''),
                'ai_response': entry.get('ai_response', ''),
                'timestamp': entry.get('timestamp', 'unknown'),
                'metadata': entry.get('metadata', {})
            })
        
        return history[-limit:]
    
    def analyze_paper(self, paper_id: str, analysis_type: str = 'summary') -> Dict[str, Any]:
        """
        Analyze a specific paper by ID or content.
        
        Args:
            paper_id: ArXiv ID or identifier of the paper
            analysis_type: Type of analysis ('summary', 'key_findings', 'methodology')
            
        Returns:
            Analysis results
        """
        try:
            # Use appropriate tool based on analysis type
            if analysis_type == 'summary':
                if 'summarization' not in self.tools or self.tools['summarization'] is None:
                    return {
                        'status': 'error',
                        'paper_id': paper_id,
                        'error': 'SummarizationTool not available. Check tool initialization.',
                        'memory_enabled': self.enable_memory
                    }
                
                tool = self.tools['summarization']
                # Create dummy paper text for demonstration (in real use, this would fetch the actual paper)
                dummy_paper_text = f"Paper content for {paper_id} would be fetched here"
                result = tool._run(
                    paper_text=dummy_paper_text,
                    user_query=f"Summarize the key findings of {paper_id}",
                    research_context="Quantitative finance research analysis"
                )
            elif analysis_type == 'key_findings':
                if 'query_answering' not in self.tools or self.tools['query_answering'] is None:
                    return {
                        'status': 'error',
                        'paper_id': paper_id,
                        'error': 'QueryAnsweringTool not available. Check tool initialization.',
                        'memory_enabled': self.enable_memory
                    }
                
                tool = self.tools['query_answering']
                result = tool._run(f"What are the key findings in {paper_id}?")
            elif analysis_type == 'methodology':
                if 'query_answering' not in self.tools or self.tools['query_answering'] is None:
                    return {
                        'status': 'error',
                        'paper_id': paper_id,
                        'error': 'QueryAnsweringTool not available. Check tool initialization.',
                        'memory_enabled': self.enable_memory
                    }
                
                tool = self.tools['query_answering']
                result = tool._run(f"What methodology is used in {paper_id}?")
            else:
                return {'error': f'Unknown analysis type: {analysis_type}'}
            
            # Store in memory if enabled
            if self.enable_memory:
                self.stm.update_context(f'paper_analysis_{paper_id}', {
                    'type': analysis_type,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
            
            return {
                'status': 'success',
                'paper_id': paper_id,
                'analysis_type': analysis_type,
                'result': result,
                'memory_enabled': self.enable_memory
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'paper_id': paper_id,
                'error': str(e),
                'memory_enabled': self.enable_memory
            }
    
    def generate_knowledge_graph(self, session_id: Optional[str] = None, topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a knowledge graph from research results.
        
        Args:
            session_id: Research session ID to generate graph from
            topics: List of topics to include in the graph
            
        Returns:
            Knowledge graph data
        """
        try:
            # Check if output generation tool is available
            if 'output_generation' not in self.tools or self.tools['output_generation'] is None:
                return {
                    'status': 'error',
                    'type': 'knowledge_graph',
                    'error': 'OutputGenerationTool not available. Check tool initialization.',
                    'memory_enabled': self.enable_memory
                }
            
            # Use output generation tool
            tool = self.tools['output_generation']
            
            # Get context data for graph generation
            context_data = {}
            if self.enable_memory:
                context_data = self.stm.get_context()
            
            # Generate knowledge graph - create proper input for the tool
            graph_input = {
                'type': 'knowledge_graph',
                'context': context_data,
                'topics': topics or [],
                'session_id': session_id or 'current'
            }
            
            # Convert to string input as expected by the tool
            result = tool._run(json.dumps(graph_input))
            
            return {
                'status': 'success',
                'type': 'knowledge_graph',
                'result': result,
                'memory_enabled': self.enable_memory
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'type': 'knowledge_graph',
                'error': str(e),
                'memory_enabled': self.enable_memory
            }
