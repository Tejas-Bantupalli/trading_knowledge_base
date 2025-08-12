#!/usr/bin/env python3
"""
Trading Knowledge Base - Main Entry Point

This is the main entry point for the Trading Knowledge Base system.
It provides a unified interface for all functionality through the TradingKnowledgeBase class.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import TradingKnowledgeBase

def setup_environment():
    """Setup environment variables and check dependencies."""
    try:
        import dotenv
        dotenv.load_dotenv()
        
        # Check for required API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables.")
            print("   Please set it in your .env file or environment.")
            print("   You can get one from: https://makersuite.google.com/app/apikey")
    except ImportError:
        print("⚠️  Warning: python-dotenv not installed. Install with: pip install python-dotenv")

def run_research(topic: str, output_dir: str = "data", enable_memory: bool = True):
    """Run comprehensive research using the Trading Knowledge Base."""
    try:
        print(f"🚀 Starting research on: {topic}")
        print("=" * 60)
        
        # Initialize the system
        tkb = TradingKnowledgeBase(data_dir=output_dir, enable_memory=enable_memory)
        
        # Run research
        results = tkb.research(topic, output_dir)
        
        if results['status'] == 'success':
            print("✅ Research completed successfully!")
            print(f"📁 Results saved to: {results['session_file']}")
            print(f"🧠 Memory enabled: {results['memory_enabled']}")
        else:
            print(f"❌ Research failed: {results['error']}")
            
        return results
        
    except Exception as e:
        print(f"❌ Error during research: {e}")
        return {'status': 'error', 'error': str(e)}

def run_quick_analysis(query: str, output_dir: str = "data", enable_memory: bool = True):
    """Run quick analysis without full crew execution."""
    try:
        print(f"🔍 Running quick analysis for: {query}")
        print("=" * 60)
        
        # Initialize the system
        tkb = TradingKnowledgeBase(data_dir=output_dir, enable_memory=enable_memory)
        
        # Run quick analysis
        results = tkb.quick_analysis(query)
        
        if results['status'] == 'success':
            print("✅ Quick analysis completed!")
            print(f"📊 Type: {results['type']}")
            print(f"🧠 Memory enabled: {results['memory_enabled']}")
        else:
            print(f"❌ Quick analysis failed: {results['error']}")
            
        return results
        
    except Exception as e:
        print(f"❌ Error during quick analysis: {e}")
        return {'status': 'error', 'error': str(e)}

def show_memory_status(output_dir: str = "data"):
    """Show the current memory system status."""
    try:
        print("🧠 Memory System Status")
        print("=" * 60)
        
        tkb = TradingKnowledgeBase(data_dir=output_dir, enable_memory=True)
        status = tkb.get_memory_status()
        
        if status['memory_enabled']:
            print(f"✅ Memory system: ENABLED")
            print(f"📝 STM entries: {status['stm_entries']}")
            print(f"💾 LTM available: {status['ltm_available']}")
            print(f"🔍 Current context: {len(status['current_context'])} items")
        else:
            print("❌ Memory system: DISABLED")
            
        return status
        
    except Exception as e:
        print(f"❌ Error checking memory status: {e}")
        return {'status': 'error', 'error': str(e)}

def clear_memory(output_dir: str = "data"):
    """Clear all memory systems."""
    try:
        print("🧹 Clearing memory systems...")
        print("=" * 60)
        
        tkb = TradingKnowledgeBase(data_dir=output_dir, enable_memory=True)
        tkb.clear_memory()
        
        print("✅ Memory cleared successfully!")
        
    except Exception as e:
        print(f"❌ Error clearing memory: {e}")

def show_system_status():
    """Show the current status of the knowledge base."""
    print("📋 Trading Knowledge Base Status")
    print("=" * 60)
    
    # Check for required files
    required_files = [
        "data/arxiv_faiss.index",
        "data/arxiv_id_mapping.jsonl",
        "src/core.py",
        "src/memory/stm.py",
        "src/memory/ltm.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"❌ {file_path} (missing)")
    
    # Check for core system
    try:
        from src import TradingKnowledgeBase
        print("✅ Core system available")
    except ImportError as e:
        print(f"❌ Core system import failed: {e}")
    
    # Check environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("✅ GOOGLE_API_KEY configured")
    else:
        print("❌ GOOGLE_API_KEY not configured")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trading Knowledge Base - AI-powered research assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py research "What are the latest developments in quantitative trading?"
  python src/main.py quick "Machine Learning in Finance"
  python src/main.py memory --status
  python src/main.py memory --clear
  python src/main.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Research command
    research_parser = subparsers.add_parser('research', help='Run comprehensive research')
    research_parser.add_argument('topic', help='Research topic')
    research_parser.add_argument('--output-dir', default='data', help='Output directory (default: data)')
    research_parser.add_argument('--no-memory', action='store_true', help='Disable memory system')
    
    # Quick analysis command
    quick_parser = subparsers.add_parser('quick', help='Run quick analysis')
    quick_parser.add_argument('query', help='Research query')
    quick_parser.add_argument('--output-dir', default='data', help='Output directory (default: data)')
    quick_parser.add_argument('--no-memory', action='store_true', help='Disable memory system')
    
    # Memory management commands
    memory_parser = subparsers.add_parser('memory', help='Memory management')
    memory_group = memory_parser.add_mutually_exclusive_group(required=True)
    memory_group.add_argument('--status', action='store_true', help='Show memory status')
    memory_group.add_argument('--clear', action='store_true', help='Clear all memory')
    memory_parser.add_argument('--output-dir', default='data', help='Data directory (default: data)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup environment
    setup_environment()
    
    # Execute command
    if args.command == 'research':
        enable_memory = not args.no_memory
        run_research(args.topic, args.output_dir, enable_memory)
        
    elif args.command == 'quick':
        enable_memory = not args.no_memory
        run_quick_analysis(args.query, args.output_dir, enable_memory)
        
    elif args.command == 'memory':
        if args.status:
            show_memory_status(args.output_dir)
        elif args.clear:
            clear_memory(args.output_dir)
            
    elif args.command == 'status':
        show_system_status()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
