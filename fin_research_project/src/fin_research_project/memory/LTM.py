import os
import json
import psycopg2
import redis
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import numpy as np
from psycopg2.extras import register_json

# Load environment variables
load_dotenv()
db_connection_string = os.getenv('POSTGRES_CONNECTION_STRING')

class LTM:
    def __init__(self, data_directory: str, db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Long-Term Memory with Redis and PostgreSQL connections.
        
        Args:
            data_directory: Directory for local data storage
            db_config: PostgreSQL connection parameters. If None, uses defaults from environment.
        """
        self.data_directory = data_directory
        self.dir = os.path.dirname(data_directory)
        
        # Initialize Redis connection
        self.r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        
        # Initialize PostgreSQL connection
        self.db_config = db_config or {
            'dbname': 'postgres',
            'user': 'postgres',
            'password': 'postgres',
            'host': 'localhost',
            'port': '5432'
        }
        self.pg_conn = None
        self.connect_db()
        self._create_tables_if_not_exists()
        
    def connect_db(self) -> None:
        """Establish connection to PostgreSQL database."""
        try:
            if self.pg_conn is None or self.pg_conn.closed:
                self.pg_conn = psycopg2.connect(**self.db_config)
                register_json(oid=3802, array_oid=3807, globally=True)
                self.pg_conn.autocommit = True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def close_db(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'pg_conn') and self.pg_conn is not None:
            self.pg_conn.close()
            self.pg_conn = None

    def _create_tables_if_not_exists(self) -> None:
        """Create necessary tables if they don't exist."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id SERIAL PRIMARY KEY,
            question_embedding VECTOR(256), 
            question_text TEXT NOT NULL,
            answer_text TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            source_paper_ids TEXT[] 
        );
        """
        try:
            with self.pg_conn.cursor() as cur:
                # Enable pgvector extension if not enabled
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(create_table_query)
        except Exception as e:
            print(f"Error creating tables: {e}")
            raise

    # ======= CONVERSATION HISTORY METHODS ======= #

    def add_conversation_entry(self, question_embedding: list[float], question_text: str, 
                             answer_text: str, source_paper_ids: list[str]) -> int:
        """
        Add a new conversation entry to the database.
        
        Args:
            question_embedding: List of floats representing the question embedding
            question_text: The original question text
            answer_text: The generated answer text
            source_paper_ids: List of paper IDs that were used as sources
            
        Returns:
            int: ID of the newly created conversation entry
        """
        query = """
        INSERT INTO conversation_history 
        (question_embedding, question_text, answer_text, source_paper_ids)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute(query, (
                    question_embedding,
                    question_text,
                    answer_text,
                    source_paper_ids
                ))
                return cur.fetchone()[0]
        except Exception as e:
            print(f"Error adding conversation entry: {e}")
            raise

    def get_similar_conversations(self, question_embedding: list[float], 
                                max_distance: float = 0.8, 
                                limit: int = 5) -> list[dict]:
        """
        Find similar past conversations using Euclidean distance (L2) vector search.
        
        Args:
            question_embedding: The embedding vector to compare against
            max_distance: Maximum Euclidean distance (lower values mean more similar, 0 = identical)
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing similar conversations with their distances
        """
        query = """
        SELECT id, question_text, answer_text, source_paper_ids,
               question_embedding <-> %s as distance
        FROM conversation_history
        WHERE question_embedding <-> %s <= %s
        ORDER BY distance ASC
        LIMIT %s;
        """
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute(query, (
                    question_embedding,
                    question_embedding,
                    max_distance,
                    limit
                ))
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error finding similar conversations: {e}")
            return []

    def get_conversation_by_id(self, conversation_id: int) -> Optional[dict]:
        """
        Retrieve a specific conversation by its ID.
        
        Args:
            conversation_id: The ID of the conversation to retrieve
            
        Returns:
            Dictionary containing conversation data or None if not found
        """
        query = """
        SELECT id, question_text, answer_text, source_paper_ids, timestamp
        FROM conversation_history
        WHERE id = %s;
        """
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute(query, (conversation_id,))
                if cur.rowcount == 0:
                    return None
                columns = [desc[0] for desc in cur.description]
                return dict(zip(columns, cur.fetchone()))
        except Exception as e:
            print(f"Error retrieving conversation {conversation_id}: {e}")
            return None

    def get_recent_conversations(self, limit: int = 10) -> list[dict]:
        """
        Retrieve the most recent conversations.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of recent conversations, most recent first
        """
        query = """
        SELECT id, question_text, answer_text, timestamp
        FROM conversation_history
        ORDER BY timestamp DESC
        LIMIT %s;
        """
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute(query, (limit,))
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error retrieving recent conversations: {e}")
            return []

    def delete_conversation(self, conversation_id: int) -> bool:
        """
        Delete a conversation by its ID.
        
        Args:
            conversation_id: The ID of the conversation to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        query = "DELETE FROM conversation_history WHERE id = %s;"
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute(query, (conversation_id,))
                return cur.rowcount > 0
        except Exception as e:
            print(f"Error deleting conversation {conversation_id}: {e}")
            return False

    # ======= PAPER CACHE METHODS =============== #

    def cache_paper(self,paper_id,text):
        """
        cache seen papers into a folder for easy retreival.
        args:
            paper_id: arxiv id for a given research paper.
            text    : parsed text from the paper.
        """
        # create file:
        filepath = os.path.join(self,dir,paper_id+'.txt')
        with open(filepath,'w') as f:
            f.write(text)
        self.r.set(paper_id,filepath)

    
    def retreive_paper_from_cache(self,paper_id):
        filepath = os.path.join(self.dir,paper_id+'.txt')
        if not os.path.exists(filepath):
            return f"requested file: {paper_id} not found in cache"
        
        with open(filepath,'r') as f:
            text = ''.join(f.readlines())
        return text
            
    
    
    
    def test(self):
        cur = self.conn.cursor()
        cur.execute("SELECT current_database();")
        print(cur.fetchone())
        
        cur.close()
        self.conn.close()
        self.r.set('hello','world')
        print(self.r.get('hello'))
if __name__=="__main__":
    test = ltm()
    test.test()