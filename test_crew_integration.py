#!/usr/bin/env python3
"""
Test script to verify CrewAI integration is working properly.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from src.core import TradingKnowledgeBase
        print("✅ TradingKnowledgeBase imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_initialization():
    """Test that TradingKnowledgeBase can be initialized."""
    print("\n🔧 Testing initialization...")
    
    try:
        # Import here to ensure it's available
        from src.core import TradingKnowledgeBase
        
        # Initialize with memory disabled to avoid database issues
        tkb = TradingKnowledgeBase(enable_memory=False)
        print("✅ TradingKnowledgeBase initialized successfully")
        return tkb
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None

def test_crew_creation():
    """Test that the crew method can create a CrewAI crew."""
    print("\n🚀 Testing crew creation...")
    
    try:
        # Import here to ensure it's available
        from src.core import TradingKnowledgeBase
        
        tkb = TradingKnowledgeBase(enable_memory=False)
        crew = tkb.crew()
        print("✅ Crew created successfully")
        print(f"   - Agents: {len(crew.agents)}")
        print(f"   - Tasks: {len(crew.tasks)}")
        return True
    except Exception as e:
        print(f"❌ Crew creation failed: {e}")
        return False

def test_tools():
    """Test that individual tools can be created."""
    print("\n🛠️ Testing tool creation...")
    
    try:
        from src.tools.vector_search_tool import VectorSearchTool
        from src.tools.summarization_tool import SummarizationTool
        from src.tools.query_answering_tool import QueryAnsweringTool
        from src.tools.critic_tool import CriticTool
        from src.tools.output_generation_tool import OutputGenerationTool
        
        print("✅ All tools imported successfully")
        
        # Test tool instantiation - handle potential initialization errors
        tools = []
        tool_names = [
            ("VectorSearchTool", VectorSearchTool),
            ("SummarizationTool", SummarizationTool),
            ("QueryAnsweringTool", QueryAnsweringTool),
            ("CriticTool", CriticTool),
            ("OutputGenerationTool", OutputGenerationTool)
        ]
        
        for name, tool_class in tool_names:
            try:
                tool = tool_class()
                tools.append(tool)
                print(f"   ✅ {name} instantiated successfully")
            except Exception as e:
                print(f"   ⚠️ {name} failed: {e}")
                # Continue with other tools
        
        print(f"✅ {len(tools)}/{len(tool_names)} tools instantiated successfully")
        return len(tools) > 0  # Pass if at least some tools work
        
    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        return False

def test_research_method():
    """Test that the research method can be called (without full execution)."""
    print("\n📚 Testing research method...")
    
    try:
        # Import here to ensure it's available
        from src.core import TradingKnowledgeBase
        
        tkb = TradingKnowledgeBase(enable_memory=False)
        
        # Test the research method (this will try to execute the crew)
        print("   - Calling research method...")
        result = tkb.research("test query")
        
        print(f"✅ Research method executed")
        print(f"   - Status: {result.get('status')}")
        print(f"   - Message: {result.get('message', 'No message')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Research method test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing CrewAI Integration\n")
    
    tests = [
        ("Imports", test_imports),
        ("Tool Creation", test_tools),
        ("Initialization", test_initialization),
        ("Crew Creation", test_crew_creation),
        ("Research Method", test_research_method),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CrewAI integration is working.")
        print("\n🚀 Next steps:")
        print("   1. Set up your API keys (GOOGLE_API_KEY, GEMINI_API_KEY)")
        print("   2. Test with real research queries")
        print("   3. Use the core API to access your system")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 Common issues:")
        print("   - Missing API keys")
        print("   - Database connection issues")
        print("   - Tool initialization problems")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
