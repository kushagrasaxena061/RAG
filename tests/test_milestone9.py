import pytest
from adaptive_rag.evaluation.baseline import NaiveBaselineRAG
from adaptive_rag.evaluation.benchmark import TokenEfficiencyBenchmark
from adaptive_rag.pipeline.orchestrator import RAGPipelineOrchestrator
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.retrieval.vector_index import VectorIndex
from adaptive_rag.retrieval.bm25_index import BM25Index
from adaptive_rag.retrieval.hybrid_search import HybridRetriever
from adaptive_rag.memory.manager import TwoTierMemory
from adaptive_rag.config import ModelContextConfig

def test_token_efficiency_benchmark_execution(tmp_path):
    """Verify that the comparative evaluation calculates empirical token reduction."""
    config = ModelContextConfig()
    token_manager = TokenBudgetManager(config)
    
    vec_idx = VectorIndex(persist_directory=str(tmp_path / "chroma"))
    bm25_idx = BM25Index(persist_path=str(tmp_path / "bm25.pkl"))
    hybrid_retriever = HybridRetriever(vec_idx, bm25_idx)
    mem_vec = VectorIndex(persist_directory=str(tmp_path / "chroma_mem"))
    memory = TwoTierMemory(token_manager, mem_vec)
    
    baseline = NaiveBaselineRAG(vec_idx, token_manager)
    adaptive = RAGPipelineOrchestrator(hybrid_retriever, token_manager, memory)
    
    benchmark = TokenEfficiencyBenchmark(baseline, adaptive, token_manager)
    results = benchmark.run_comparison(["Compare Q3 revenue growth."])
    
    assert len(results) == 1
    assert "token_savings_pct" in results[0]
    assert "baseline_tokens" in results[0]
    assert "adaptive_tokens" in results[0]
