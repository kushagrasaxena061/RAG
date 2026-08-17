import pytest
from adaptive_rag.models.schema import Chunk, ChunkMetadata, ChunkType, ContentType
from adaptive_rag.reranking.cross_encoder import Reranker
from adaptive_rag.context.compressor import ContextCompressor
from adaptive_rag.context.token_budget import TokenBudgetManager, TokenBudgetReport
from adaptive_rag.config import ModelContextConfig

@pytest.fixture
def sample_chunks():
    meta = ChunkMetadata(
        document_id="doc_1", document_name="test.pdf", page_number=1, 
        created_at=0.0, content_type=ContentType.TEXT
    )
    
    c1 = Chunk(chunk_id="c_1", chunk_type=ChunkType.CHILD, content="The sky is blue today.", token_count=6, content_hash="h1", metadata=meta.model_copy())
    c2 = Chunk(chunk_id="c_2", chunk_type=ChunkType.CHILD, content="Company revenue increased by 15% in Q3 due to cloud computing sales.", token_count=13, content_hash="h2", metadata=meta.model_copy())
    c3 = Chunk(chunk_id="c_3", chunk_type=ChunkType.CHILD, content="The CEO mentioned that cloud computing is the future of the industry.", token_count=13, content_hash="h3", metadata=meta.model_copy())
    return [c1, c2, c3]

def test_cross_encoder_reranking(sample_chunks):
    """Verify that the Cross-Encoder pushes the most exactly relevant chunk to the top."""
    reranker = Reranker() 
    query = "Why did revenue go up in Q3?"
    
    reranked = reranker.rerank(query, sample_chunks)
    
    # c2 directly answers the financial question
    assert reranked[0].chunk_id == "c_2" 

def test_context_compression():
    """Verify that lower-ranked evidence is dropped when the token budget is reached."""
    config = ModelContextConfig(
        context_window_tokens=1000, max_output_tokens=100, 
        system_prompt_reserve=100, conversation_memory_reserve=0, 
        query_reserve=0, safety_margin_tokens=0
    )
    manager = TokenBudgetManager(config)
    compressor = ContextCompressor(manager)
    
    budget = TokenBudgetReport(
        context_window=1000, system_prompt_tokens=100, conversation_memory_tokens=0, 
        query_tokens=10, output_budget=100, safety_margin=0, 
        max_retrieval_tokens=50, available_tokens=50
    )
    
    meta = ChunkMetadata(document_id="d1", document_name="t.pdf", page_number=1, created_at=0.0)
    
    c1 = Chunk(chunk_id="1", chunk_type=ChunkType.CHILD, content="A"*50, token_count=30, content_hash="h1", metadata=meta.model_copy())
    c2 = Chunk(chunk_id="2", chunk_type=ChunkType.CHILD, content="B"*50, token_count=15, content_hash="h2", metadata=meta.model_copy())
    c3 = Chunk(chunk_id="3", chunk_type=ChunkType.CHILD, content="C"*50, token_count=20, content_hash="h3", metadata=meta.model_copy())
    
    compressed = compressor.compress_context([c1, c2, c3], budget)
    
    assert len(compressed) == 2
    assert compressed[0].chunk_id == "1"
    assert compressed[1].chunk_id == "2"
