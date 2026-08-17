import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.retrieval.vector_index import VectorIndex
from adaptive_rag.models.schema import Chunk, ChunkMetadata, ChunkType, ContentType

class Message(BaseModel):
    """Represents a single conversational turn."""
    role: str
    content: str
    token_count: int

class TwoTierMemory:
    """Manages short-term token-bounded memory and long-term semantic memory."""
    
    def __init__(self, token_manager: TokenBudgetManager, long_term_index: VectorIndex):
        self.short_term: List[Message] = []
        self.token_manager = token_manager
        self.long_term_index = long_term_index
        
    def add_interaction(self, user_text: str, assistant_text: str):
        """Stores the interaction in the short-term array and embeds it into long-term storage."""
        u_tokens = self.token_manager.count_tokens(user_text)
        a_tokens = self.token_manager.count_tokens(assistant_text)
        
        # 1. Update Short-Term Memory
        self.short_term.append(Message(role="user", content=user_text, token_count=u_tokens))
        self.short_term.append(Message(role="assistant", content=assistant_text, token_count=a_tokens))
        
        # 2. Update Long-Term Semantic Memory
        chunk_id = f"mem_{time.time()}"
        content = f"User previously asked: {user_text}\nSystem previously answered: {assistant_text}"
        
        meta = ChunkMetadata(
            document_id="conversation_memory", 
            document_name="memory", 
            page_number=0, 
            created_at=time.time(), 
            content_type=ContentType.TEXT
        )
        
        mem_chunk = Chunk(
            chunk_id=chunk_id, 
            chunk_type=ChunkType.CHILD, 
            content=content, 
            token_count=u_tokens + a_tokens, 
            content_hash=chunk_id, 
            metadata=meta
        )
        
        self.long_term_index.add_chunks([mem_chunk])

    def get_short_term_context(self, max_budget_tokens: int) -> str:
        """Retrieves the most recent messages that strictly fit within the token budget."""
        context_string = ""
        current_tokens = 0
        
        # Iterate backwards to prioritize the most recent context
        for msg in reversed(self.short_term):
            if current_tokens + msg.token_count <= max_budget_tokens:
                formatted_msg = f"{msg.role.capitalize()}: {msg.content}\n"
                context_string = formatted_msg + context_string
                current_tokens += msg.token_count
            else:
                break # Token budget exhausted
                
        return context_string.strip()

    def get_relevant_long_term(self, query: str, top_k: int = 2) -> List[dict]:
        """Semantically searches past conversations for relevant distant context."""
        return self.long_term_index.search(
            query=query, 
            top_k=top_k, 
            where={"document_id": "conversation_memory"}
        )
