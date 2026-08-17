import pytest
from adaptive_rag.memory.manager import TwoTierMemory
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.retrieval.vector_index import VectorIndex
from adaptive_rag.config import ModelContextConfig

@pytest.fixture
def memory_system(tmp_path):
    config = ModelContextConfig(
        context_window_tokens=32000, max_output_tokens=4000, 
        system_prompt_reserve=1000, conversation_memory_reserve=100, # Artificially small for testing
        query_reserve=500, safety_margin_tokens=500
    )
    token_manager = TokenBudgetManager(config)
    vector_index = VectorIndex(persist_directory=str(tmp_path / "chroma_memory"))
    return TwoTierMemory(token_manager, vector_index)

def test_short_term_memory_truncation(memory_system):
    """Verify that old messages are correctly truncated to respect the token budget."""
    
    # Add 10 dummy interactions
    for i in range(10):
        memory_system.add_interaction(f"Question {i}", f"Answer {i}")
        
    # We set a tight budget of ~30 tokens. It should only return the last 2-3 messages.
    short_term_context = memory_system.get_short_term_context(max_budget_tokens=30)
    
    # Ensure it captured the most recent interactions
    assert "Question 9" in short_term_context
    assert "Answer 9" in short_term_context
    
    # Ensure it dropped the oldest interactions to save tokens
    assert "Question 0" not in short_term_context
    assert "Question 1" not in short_term_context

def test_long_term_semantic_memory(memory_system):
    """Verify that a distant, truncated fact can be retrieved semantically."""
    
    # Inject a highly specific fact
    memory_system.add_interaction("What is my project codename?", "Your project codename is Project Titan.")
    
    # Flood the short term memory to push the fact out
    for i in range(20):
        memory_system.add_interaction(f"Filler {i}", f"Filler answer {i}")
        
    # Prove the fact is gone from the bounded short term memory
    short_term = memory_system.get_short_term_context(max_budget_tokens=100)
    assert "Project Titan" not in short_term
    
    # Prove the fact is successfully recovered via long-term semantic search
    recovered_memories = memory_system.get_relevant_long_term("project codename", top_k=1)
    assert len(recovered_memories) == 1
    
    # The chunk ID corresponds to what we pushed to Chroma
    assert recovered_memories[0]["chunk_id"].startswith("mem_")
