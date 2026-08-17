import pytest
from adaptive_rag.evaluation.metrics import RetrievalMetrics
from adaptive_rag.evaluation.regression import RegressionTester

def test_retrieval_metrics():
    """Verify Recall and MRR calculations mathematically."""
    retrieved = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    relevant = {"chunk3"}
    
    recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k=5)
    mrr = RetrievalMetrics.mrr(retrieved, relevant)
    
    assert recall == 1.0
    assert mrr == 1.0 / 3.0

def test_regression_tester(tmp_path):
    """Verify the system correctly flags regressions if quality drops while chasing token efficiency."""
    history_file = str(tmp_path / "hist.json")
    tester = RegressionTester(history_file=history_file)
    
    # Baseline: High Tokens, Good Recall
    tester.save_run("v1.0_naive", {"avg_tokens_per_query": 8500, "recall_at_5": 0.80})
    
    # Our Adaptive System: Low Tokens, Better Recall (Successful Architecture)
    tester.save_run("v2.0_adaptive", {"avg_tokens_per_query": 2100, "recall_at_5": 0.85})
    
    comparison = tester.compare_versions("v1.0_naive", "v2.0_adaptive")
    assert comparison["token_efficiency_improvement"] == 6400
    assert comparison["recall_change"] == 0.05
    assert comparison["quality_maintained"] is True
    assert comparison["regression_detected"] is False
