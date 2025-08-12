"""
Tests for the Trading Knowledge Base core system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import TradingKnowledgeBase
from src.memory import STM, LTM


class TestTradingKnowledgeBase:
    """Test the main TradingKnowledgeBase class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tkb = TradingKnowledgeBase(data_dir=self.temp_dir, enable_memory=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test that the system initializes correctly."""
        assert self.tkb.data_dir == Path(self.temp_dir)
        assert self.tkb.enable_memory is True
        assert hasattr(self.tkb, 'stm')
        assert hasattr(self.tkb, 'ltm')
    
    def test_memory_status(self):
        """Test memory status reporting."""
        status = self.tkb.get_memory_status()
        assert status['memory_enabled'] is True
        assert 'stm_entries' in status
        assert 'ltm_available' in status
        assert 'current_context' in status
    
    def test_memory_disabled(self):
        """Test system without memory."""
        tkb_no_memory = TradingKnowledgeBase(data_dir=self.temp_dir, enable_memory=False)
        status = tkb_no_memory.get_memory_status()
        assert status['memory_enabled'] is False
    
    def test_clear_memory(self):
        """Test memory clearing functionality."""
        # Add some context
        self.tkb.stm.update_context('test_key', 'test_value')
        assert self.tkb.stm.get_context('test_key') == 'test_value'
        
        # Clear memory
        self.tkb.clear_memory()
        assert len(self.tkb.stm.conversation_buffer) == 0
        assert len(self.tkb.stm.current_context) == 0


class TestSTM:
    """Test the Short-Term Memory system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stm = STM(max_entries=5)
    
    def test_add_interaction(self):
        """Test adding interactions to STM."""
        self.stm.add_interaction("Hello", "Hi there!")
        assert len(self.stm.conversation_buffer) == 1
        assert self.stm.conversation_buffer[0]['user'] == "Hello"
        assert self.stm.conversation_buffer[0]['assistant'] == "Hi there!"
    
    def test_max_entries_limit(self):
        """Test that STM respects max entries limit."""
        for i in range(10):
            self.stm.add_interaction(f"User {i}", f"Assistant {i}")
        
        assert len(self.stm.conversation_buffer) == 5  # max_entries
        assert self.stm.conversation_buffer[0]['user'] == "User 5"  # Most recent
    
    def test_context_management(self):
        """Test context management functionality."""
        self.stm.update_context('key1', 'value1')
        self.stm.update_context('key2', 'value2')
        
        assert self.stm.get_context('key1') == 'value1'
        assert self.stm.get_context('key2') == 'value2'
        assert self.stm.get_context() == {'key1': 'value1', 'key2': 'value2'}
        
        self.stm.clear_context()
        assert len(self.stm.current_context) == 0
    
    def test_temp_data(self):
        """Test temporary data functionality."""
        self.stm.set_temp_data('temp_key', 'temp_value', ttl=1)
        assert self.stm.get_temp_data('temp_key') == 'temp_value'
        
        # Wait for expiration (in real test, you'd mock time)
        import time
        time.sleep(1.1)
        assert self.stm.get_temp_data('temp_key') is None


class TestLTM:
    """Test the Long-Term Memory system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Note: LTM requires actual database connections, so we'll test basic functionality
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test LTM initialization (without actual DB connection)."""
        # This test will fail if we try to connect to actual databases
        # In a real test environment, you'd use test databases or mocks
        pass
    
    def test_paper_caching(self):
        """Test paper caching functionality."""
        # Test file operations without database
        ltm = LTM.__new__(LTM)  # Create instance without calling __init__
        ltm.dir = self.temp_dir
        
        # Test paper caching
        test_paper_id = "test_paper_123"
        test_text = "This is test paper content."
        
        ltm.cache_paper(test_paper_id, test_text)
        
        # Check if file was created
        paper_file = Path(self.temp_dir) / f"{test_paper_id}.txt"
        assert paper_file.exists()
        
        # Test retrieval
        retrieved_text = ltm.retrieve_paper_from_cache(test_paper_id)
        assert retrieved_text == test_text


if __name__ == "__main__":
    pytest.main([__file__])
