# Trading Knowledge Base

An AI-powered quantitative finance research system that combines crew-based orchestration with advanced memory management for comprehensive academic paper analysis.

## 🚀 Features

- **Crew-Based Research**: Multi-agent collaboration for comprehensive research
- **Memory Management**: STM (Short-Term Memory) and LTM (Long-Term Memory) systems
- **Vector Search**: Advanced paper retrieval using FAISS and embeddings
- **Paper Analysis**: AI-powered summarization and query answering
- **Quality Control**: Critical review and validation of outputs
- **Knowledge Graphs**: Visual representation of research relationships
- **Clean API**: Simple, direct access to all functionality

## 🏗️ Architecture

```
src/
├── core.py                 # Main TradingKnowledgeBase class
├── memory/                 # Memory management system
│   ├── stm.py             # Short-Term Memory
│   └── ltm.py             # Long-Term Memory (PostgreSQL + Redis)
├── tools/                  # Specialized research tools
│   ├── vector_search_tool.py
│   ├── arxiv_retrieval_tool.py
│   ├── summarization_tool.py
│   ├── query_answering_tool.py
│   ├── critic_tool.py
│   └── output_generation_tool.py
└── main.py                 # Command-line interface
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/trading-knowledge-base.git
   cd trading-knowledge-base
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Set up databases** (optional, for memory features):
   ```bash
   # PostgreSQL with pgvector extension
   docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg15
   
   # Redis
   docker run -d --name redis -p 6379:6379 redis:alpine
   ```

## 🚀 Quick Start

### Basic Research
```bash
# Run comprehensive research
python src/main.py research "Machine Learning in Quantitative Trading"

# Run quick analysis
python src/main.py quick "What are the latest developments in risk management?"
```

### Memory Management
```bash
# Check memory status
python src/main.py memory --status

# Clear memory
python src/main.py memory --clear
```

### System Status
```bash
# Check system health
python src/main.py status
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
GOOGLE_API_KEY=your_google_api_key

# Optional (for memory features)
POSTGRES_CONNECTION_STRING=postgresql://user:pass@localhost:5432/dbname
REDIS_URL=redis://localhost:6379
```

### Data Directory
The system stores data in the `data/` directory by default. You can customize this:
```python
from src import TradingKnowledgeBase

# Custom data directory
tkb = TradingKnowledgeBase(data_dir="my_data", enable_memory=True)
```

## 🧠 Memory System

### Short-Term Memory (STM)
- Manages conversation context and recent interactions
- Configurable buffer size (default: 10 entries)
- Temporary data storage with TTL

### Long-Term Memory (LTM)
- PostgreSQL with pgvector for vector similarity search
- Redis for fast paper caching
- Persistent storage of research sessions

## 🤖 Agents

1. **Research Orchestrator**: Coordinates research operations
2. **Paper Retrieval Agent**: Finds relevant papers using vector search
3. **Summarization Agent**: Creates focused paper summaries
4. **Query Answering Agent**: Extracts specific answers using RAG
5. **Critic Agent**: Validates outputs for accuracy and quality

## 📊 Outputs

### Research Reports
- `findings.md`: Comprehensive research findings
- `knowledge_graph.json`: Structured relationship data
- Session metadata and logs

### Knowledge Graphs
- Nodes: Research queries, papers, concepts
- Edges: Relationships and relevance scores
- Properties: Metadata and confidence levels

## 🔍 Usage Examples

### Python API
```python
from src import TradingKnowledgeBase

# Initialize system
tkb = TradingKnowledgeBase(enable_memory=True)

# Run research
results = tkb.research("Deep Learning in Finance")

# Quick analysis
quick_results = tkb.quick_analysis("Risk management strategies")

# Check memory
status = tkb.get_memory_status()
```

### Command Line
```bash
# Full research workflow
python src/main.py research "Quantitative Trading Strategies" --output-dir research_output

# Quick analysis without memory
python src/main.py quick "Market microstructure" --no-memory

# Memory management
python src/main.py memory --status --output-dir data
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📁 Project Structure

```
trading-knowledge-base/
├── src/                    # Source code
│   ├── core.py            # Main system class
│   ├── memory/            # Memory management
│   ├── tools/             # Research tools
│   └── main.py            # CLI interface
├── data/                   # Data storage
├── scripts/                # Utility scripts
├── tests/                  # Test suite
├── config/                 # Configuration files
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [PostgreSQL](https://www.postgresql.org/) and [Redis](https://redis.io/) for data storage

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/trading-knowledge-base/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/trading-knowledge-base/discussions)
- **Email**: team@tradingkb.com

---

**Built with ❤️ for the quantitative finance community**
