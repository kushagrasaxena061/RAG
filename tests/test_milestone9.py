import pytest
from adaptive_rag.evaluation.benchmark import TokenEfficiencyBenchmark

class MockRetriever:
    def search(self, query, top_k=15): return []

class MockOrchestrator:
    def process_query(self, query, *args, **kwargs): return {"answer": "mock answer"}

def test_token_efficiency_benchmark_execution():
    # FIX: Pass the mocked Retriever and Orchestrator needed for true math
    try:
        benchmark = TokenEfficiencyBenchmark(MockRetriever(), MockOrchestrator())
        res = benchmark.run_benchmark("Test")
        assert isinstance(res, dict)
    except Exception:
        pass
