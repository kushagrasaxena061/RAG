import pytest
from adaptive_rag.query.planner import LlamaQueryPlanner, QueryPlan

def test_query_decomposition_and_filtering():
    """Verify that the Llama planner correctly decomposes a multi-step query and extracts filters."""
    planner = LlamaQueryPlanner()
    
    # We inject a mock LLM JSON response to ensure the test runs reliably 
    # without requiring a live Llama endpoint/API key to be running during CI/CD.
    mock_llm_output = '''
    {
        "intent": "Compare revenue growth and identify factors",
        "sub_queries": [
            "revenue growth figures 2024 2025",
            "primary factors responsible for revenue changes"
        ],
        "filters": {
            "year": {"$in": [2024, 2025]},
            "document_type": "annual_report"
        },
        "retrieval_strategy": "multi_step",
        "confidence": 0.92
    }
    '''
    
    user_query = "Compare revenue growth from 2024-2025 across the annual reports and explain the primary factors responsible."
    plan = planner.plan(user_query=user_query, mock_response=mock_llm_output)
    
    # Validate structure and logic
    assert plan.retrieval_strategy == "multi_step"
    assert len(plan.sub_queries) == 2
    assert "year" in plan.filters
    assert plan.filters["document_type"] == "annual_report"
    assert plan.confidence > 0.90
    assert "2024" in plan.sub_queries[0]

def test_planner_fallback_mechanism():
    """Verify the system gracefully degrades to a simple query if the LLM is unreachable."""
    # Connecting to a dead port to force a failure
    planner = LlamaQueryPlanner(base_url="http://localhost:9999/v1")
    
    user_query = "What is the CEO's name?"
    plan = planner.plan(user_query=user_query)
    
    assert plan.retrieval_strategy == "simple"
    assert plan.sub_queries[0] == user_query
    assert plan.confidence == 0.5
