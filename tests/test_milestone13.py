import pytest
from adaptive_rag.pipeline.orchestrator import RAGPipelineOrchestrator
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.retrieval.hybrid_search import HybridRetriever
from adaptive_rag.retrieval.vector_index import VectorIndex
from adaptive_rag.retrieval.bm25_index import BM25Index
from adaptive_rag.memory.manager import TwoTierMemory
from adaptive_rag.config import ModelContextConfig

def test_orchestrator_integration(tmp_path):
    """Verify the orchestrator successfully loads the new contradiction and temporal modules."""
    config = ModelContextConfig()
    token_manager = TokenBudgetManager(config)
    
    vec_idx = VectorIndex(persist_directory=str(tmp_path / "chroma"))
    bm25_idx = BM25Index(persist_path=str(tmp_path / "bm25.pkl"))
    hybrid_retriever = HybridRetriever(vec_idx, bm25_idx)
    
    mem_vec = VectorIndex(persist_directory=str(tmp_path / "chroma_mem"))
    memory = TwoTierMemory(token_manager, mem_vec)
    
    orchestrator = RAGPipelineOrchestrator(hybrid_retriever, token_manager, memory)
    
    # Assert modules instantiated correctly
    assert orchestrator.temporal_engine is not None
    assert orchestrator.contradiction_detector is not None
    
    # Fast path test
    res = orchestrator.process_query("hello")
    assert "Hello!" in res["answer"]
