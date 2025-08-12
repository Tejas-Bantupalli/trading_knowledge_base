"""
Trading Knowledge Base - AI-powered quantitative finance research system

This package provides:
- Crew-based research orchestration
- Vector search and retrieval tools
- Memory management (STM/LTM)
- Paper analysis and summarization
- Knowledge graph generation
"""

__version__ = "1.0.0"
__author__ = "Trading Knowledge Base Team"

from .core import TradingKnowledgeBase
from .memory import STM, LTM
from .tools import *

__all__ = [
    "TradingKnowledgeBase",
    "STM", 
    "LTM",
    "VectorSearchTool",
    "ArxivRetrievalTool",
    "SummarizationTool",
    "QueryAnsweringTool",
    "CriticTool",
    "OutputGenerationTool"
]
