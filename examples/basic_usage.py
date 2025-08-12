#!/usr/bin/env python3
"""
Basic Usage Example for Trading Knowledge Base

This script demonstrates how to use the Trading Knowledge Base system
for quantitative finance research.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import TradingKnowledgeBase

def main():
    """Demonstrate basic usage of the Trading Knowledge Base."""
    
    print("🚀 Trading Knowledge Base - Basic Usage Example")
    print("=" * 60)
    
    # Initialize the system
    print("📋 Initializing Trading Knowledge Base...")
    tkb = TradingKnowledgeBase(
        data_dir="data", 
        enable_memory=True
    )
    
    # Check memory status
    print("\n🧠 Checking memory system...")
    memory_status = tkb.get_memory_status()
    print(f"   Memory enabled: {memory_status['memory_enabled']}")
    if memory_status['memory_enabled']:
        print(f"   STM entries: {memory_status['stm_entries']}")
        print(f"   LTM available: {memory_status['ltm_available']}")
    
    # Example research query
    query = "Machine Learning in Quantitative Trading"
    print(f"\n🔍 Example research query: {query}")
    
    # Run quick analysis
    print("\n⚡ Running quick analysis...")
    try:
        quick_results = tkb.quick_analysis(query)
        if quick_results['status'] == 'success':
            print("✅ Quick analysis completed successfully!")
            print(f"   Type: {quick_results['type']}")
            print(f"   Memory enabled: {quick_results['memory_enabled']}")
        else:
            print(f"❌ Quick analysis failed: {quick_results['error']}")
    except Exception as e:
        print(f"❌ Error during quick analysis: {e}")
    
    # Example of running full research (commented out to avoid long execution)
    print("\n📚 Full research example (commented out to avoid long execution):")
    print("   # results = tkb.research(query, output_dir='research_output')")
    print("   # This would run the full crew-based research workflow")
    
    # Show how to customize the system
    print("\n🔧 Customization examples:")
    print("   # Custom data directory")
    print("   # tkb = TradingKnowledgeBase(data_dir='my_research_data')")
    print("   # Disable memory")
    print("   # tkb = TradingKnowledgeBase(enable_memory=False)")
    
    print("\n✅ Basic usage example completed!")
    print("   Check the README.md for more detailed examples and API documentation.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
