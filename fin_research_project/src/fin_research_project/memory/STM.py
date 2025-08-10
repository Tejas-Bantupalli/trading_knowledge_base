from datetime import datetime
import time
from typing import Any, Dict, List, Optional

class STM:
    """
    Short-Term Memory for managing conversation context and recent interactions.
    Designed to work alongside the LTM (Long-Term Memory) class.
    """
    
    def __init__(self, max_entries: int = 10):
        """
        Initialize Short-Term Memory.
        Args:
            max_entries: Maximum number of interactions to keep in memory
        """
        self.max_entries = max_entries
        self.conversation_buffer = []
        self.current_context = {}

    def add_interaction(self, user_input: str, ai_response: str, metadata: Optional[dict] = None) -> None:
        """
        Add a new interaction to the short-term memory.
        Args:
            user_input: The user's input/message
            ai_response: The assistant's response
            metadata: Additional metadata about the interaction
        """
        self.conversation_buffer.append({
            'user': user_input,
            'assistant': ai_response,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        })
        # Keep only the most recent N entries
        self.conversation_buffer = self.conversation_buffer[-self.max_entries:]

    def update_context(self, key: str, value: Any) -> None:
        """Update the current conversation context."""
        self.current_context[key] = value

    def get_context(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get specific context value or entire context.
        Args:
            key: Optional key to get a specific context value
            default: Default value to return if key is not found
        Returns:
            The context value, entire context, or default
        """
        if key is None:
            return self.current_context
        return self.current_context.get(key, default)

    def clear_context(self) -> None:
        """Clear the current context."""
        self.current_context = {}

    def get_recent_history(self, n: int = 3) -> List[dict]:
        """
        Get the last N interactions.
        Args:
            n: Number of recent interactions to return
        Returns:
            List of recent interactions
        """
        return self.conversation_buffer[-n:]

    def get_formatted_history(self) -> str:
        """
        Get conversation history in a formatted string.
        Returns:
            Formatted conversation history
        """
        return "\n".join(
            f"User: {item['user']}\nAssistant: {item['assistant']}"
            for item in self.conversation_buffer
        )

    def set_temp_data(self, key: str, value: Any, ttl: int = 300) -> None:
        """
        Store temporary data that expires after TTL seconds.
        Args:
            key: Data identifier
            value: Data to store
            ttl: Time to live in seconds
        """
        self.current_context[f"_temp_{key}"] = {
            'value': value,
            'expires_at': time.time() + ttl
        }

    def get_temp_data(self, key: str, default: Any = None) -> Any:
        """
        Retrieve temporary data if it exists and hasn't expired.
        Args:
            key: Data identifier
            default: Default value to return if data doesn't exist or has expired
        Returns:
            The stored value or default
        """
        temp_data = self.current_context.get(f"_temp_{key}")
        if temp_data and temp_data['expires_at'] > time.time():
            return temp_data['value']
        return default

    def promote_to_ltm(self, ltm, embedding: List[float], metadata: Optional[dict] = None) -> Optional[int]:
        """
        Promote the current conversation to long-term memory.
        Args:
            ltm: Instance of LTM class
            embedding: Vector embedding of the conversation
            metadata: Additional metadata to store
            
        Returns:
            ID of the created LTM entry or None if no recent interaction
        """
        if not self.conversation_buffer:
            return None
            
        # Get the last user input and assistant response
        last_interaction = self.conversation_buffer[-1]
        
        # Add to LTM
        return ltm.add_conversation_entry(
            question_embedding=embedding,
            question_text=last_interaction['user'],
            answer_text=last_interaction['assistant'],
            source_paper_ids=metadata.get('source_paper_ids', []) if metadata else []
        )

    def get_state(self) -> dict:
        """
        Get current STM state for serialization.
        Returns:
            Dictionary containing the current state
        """
        return {
            'conversation_buffer': self.conversation_buffer,
            'current_context': self.current_context
        }

    def load_state(self, state: dict) -> None:
        """
        Load STM state from serialized data.
        Args:
            state: Dictionary containing state to load
        """
        self.conversation_buffer = state.get('conversation_buffer', [])
        self.current_context = state.get('current_context', {})
