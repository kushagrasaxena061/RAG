from typing import List, Dict, Any
from adaptive_rag.evaluation.baseline import NaiveBaselineRAG
from adaptive_rag.pipeline.orchestrator import RAGPipelineOrchestrator
from adaptive_rag.context.token_budget import TokenBudgetManager

class TokenEfficiencyBenchmark:
    """Runs head-to-head comparison between Naive Baseline RAG and Adaptive RAG."""
    def __init__(self, baseline: NaiveBaselineRAG, adaptive: RAGPipelineOrchestrator, token_manager: TokenBudgetManager):
        self.baseline = baseline
        self.adaptive = adaptive
        self.token_manager = token_manager

    def run_comparison(self, test_queries: List[str]) -> List[Dict[str, Any]]:
        comparison_results = []
        
        for q in test_queries:
            # 1. Run Baseline
            base_res = self.baseline.process_query(q)
            
            # 2. Run Adaptive Platform
            adapt_res = self.adaptive.process_query(q)
            telemetry = adapt_res.get("telemetry", {})
            
            base_tokens = max(base_res["input_tokens"], 1)
            adapt_tokens = max(telemetry.get("input_tokens", 1), 1)
            token_reduction = ((base_tokens - adapt_tokens) / base_tokens) * 100.0
            
            comparison_results.append({
                "query": q,
                "baseline_tokens": base_tokens,
                "adaptive_tokens": adapt_tokens,
                "token_savings_pct": round(token_reduction, 2),
                "baseline_latency_ms": round(base_res["latency_ms"], 2),
                "adaptive_latency_ms": round(telemetry.get("latency_ms", 0), 2),
                "compression_ratio": round(telemetry.get("compression_ratio", 0.0), 2),
                "adaptive_chunks": telemetry.get("final_chunks_used", 0),
                "baseline_chunks": base_res["chunks_used"]
            })
            
        return comparison_results
