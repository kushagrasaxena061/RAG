import pytest
from adaptive_rag.models.schema import QueryPlan, QueryCategory, TableRepresentation, Chunk, ChunkMetadata, ChunkType, ContentType
from adaptive_rag.query.planner import LlamaQueryPlanner
from adaptive_rag.evaluation.metrics import RetrievalMetrics
from adaptive_rag.evaluation.benchmark import TokenEfficiencyBenchmark
from adaptive_rag.reasoning.evaluator import SelfEvaluator
from adaptive_rag.storage.cache import MultiTierCache
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.config import ModelContextConfig

def test_llama_query_intelligence_classification():
    planner = LlamaQueryPlanner()
    plan = planner._rule_based_fallback("Compare revenue growth between 2024 and 2025 table breakdown")
    
    assert plan.category == QueryCategory.MULTI_STEP or plan.category == QueryCategory.TABULAR
    assert "year" in plan.filters
    assert len(plan.sub_queries) >= 2
    assert plan.bm25_weight >= 0.6

def test_self_evaluation_and_claim_verification():
    evaluator = SelfEvaluator()
    meta = ChunkMetadata(document_id="doc1", document_name="doc.pdf", page_number=1, created_at=0.0)
    c1 = Chunk(chunk_id="c1", chunk_type=ChunkType.CHILD, content="Net revenue was $15M in 2024.", token_count=8, content_hash="h1", metadata=meta)
    
    report = evaluator.evaluate_response(
        query="What was net revenue in 2024?",
        chunks=[c1],
        answer="Net revenue was $15M in 2024 [doc.pdf, p.1]."
    )
    assert report.is_supported_by_evidence is True
    assert report.faithfulness_score >= 0.85

def test_retrieval_and_quality_token_metrics():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c5"}
    
    recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k=5)
    precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k=5)
    mrr = RetrievalMetrics.mrr(retrieved, relevant)
    ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=5)
    eff_ratio = RetrievalMetrics.calculate_efficiency_ratio(quality_score=0.95, tokens_used=1900)
    
    assert recall == 1.0
    assert precision == 0.4
    assert mrr == 0.5
    assert ndcg > 0.6
    assert eff_ratio == 0.5

def test_multitier_cache():
    cache = MultiTierCache(ttl_seconds=3600)
    cache.set_query_response("What is revenue?", "doc_hash_1", {"answer": "$10M"})
    
    assert cache.get_query_response("What is revenue?", "doc_hash_1") == {"answer": "$10M"}
    assert cache.get_query_response("What is revenue?", "doc_hash_2") is None
