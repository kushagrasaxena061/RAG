import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from adaptive_rag.config import global_config

class QueryPlan(BaseModel):
    """Structured reasoning metadata output from the Llama planner."""
    intent: str = Field(description="The core intent of the user query.")
    sub_queries: List[str] = Field(description="1 or more rewritten queries optimized for vector/BM25 search.")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Extracted metadata filters (e.g., year, document_type).")
    retrieval_strategy: str = Field(description="Strategy: 'simple', 'multi_step', or 'conversational'.")
    confidence: float = Field(description="Confidence score (0.0 to 1.0) in the understanding of the query.")

class LlamaQueryPlanner:
    """Uses a Llama-family model to understand, rewrite, and route queries."""
    
    def __init__(self, api_key: str = "sk-mock", base_url: str = "http://localhost:11434/v1"):
        # Defaults to a local endpoint (e.g., Ollama), can be overridden for Together/Groq/vLLM
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = global_config.model.model_name

    def plan(self, user_query: str, mock_response: Optional[str] = None) -> QueryPlan:
        """Generates a structured query plan. Allows mock_response for deterministic testing."""
        if mock_response:
            return QueryPlan.model_validate_json(mock_response)

        system_prompt = """You are an expert AI Query Planner for an advanced RAG system.
Your job is to analyze the user query and output a strict JSON object representing the retrieval plan.
Extract any temporal (year) or entity filters. Break complex questions into sub-queries.
Determine if the strategy should be 'simple' (one fact), 'multi_step' (comparison/synthesis), or 'conversational'.

Output JSON schema:
{
    "intent": "string",
    "sub_queries": ["string"],
    "filters": {"key": "value"},
    "retrieval_strategy": "string",
    "confidence": float
}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw_json = response.choices[0].message.content
            return QueryPlan.model_validate_json(raw_json)
        except Exception as e:
            # Graceful failure handling (Requirement #18)
            print(f"[Warning] Llama Planner failed: {e}. Falling back to simple routing.")
            return QueryPlan(
                intent="fallback",
                sub_queries=[user_query],
                filters={},
                retrieval_strategy="simple",
                confidence=0.5
            )
