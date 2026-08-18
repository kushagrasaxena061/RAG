import pytest
from adaptive_rag.query.planner import LlamaQueryPlanner

def test_query_decomposition_and_filtering():
    planner = LlamaQueryPlanner()
    mock_llm_output = '''{
        "intent": "Compare revenue growth and identify factors",
        "category": "multi_step",
        "rewritten_query": "Compare revenue",
        "expanded_queries": [],
        "sub_queries": ["rev 2024", "rev 2025"],
        "filters": {"year": 2024},
        "adaptive_top_k": 12,
        "bm25_weight": 0.5,
        "vector_weight": 0.5,
        "needs_multi_step": true,
        "confidence": 0.92
    }'''
    plan = planner.plan(user_query="Compare revenue", mock_response=mock_llm_output)
    assert plan.category.value == "multi_step"
    assert len(plan.sub_queries) == 2

def test_planner_fallback_mechanism():
    planner = LlamaQueryPlanner(base_url="http://localhost:9999/v1")
    plan = planner.plan(user_query="What is the CEO's name?")
    assert plan.category.value == "simple"
