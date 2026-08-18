import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from adaptive_rag.models.schema import QueryPlan, QueryCategory
from adaptive_rag.config import global_config

class LlamaQueryPlanner:
    def __init__(self, api_key: str = "sk-mock", base_url: str = "http://localhost:11434/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=3.0)
        self.model_name = global_config.model.model_name

    def _rule_based_fallback(self, user_query: str) -> QueryPlan:
        query_lower = user_query.lower()
        
        tabular_keywords = ["table", "revenue", "profit", "ebitda", "salary", "balance", "cost", "breakdown", "numbers", "financials"]
        is_tabular = any(w in query_lower for w in tabular_keywords)
        
        comparative_keywords = ["compare", "difference", "vs", "versus", "between", "change from", "trend"]
        is_comparative = any(w in query_lower for w in comparative_keywords)
        
        visual_keywords = ["chart", "figure", "graph", "diagram", "image", "plot"]
        is_visual = any(w in query_lower for w in visual_keywords)
        
        category = QueryCategory.SIMPLE
        if is_visual: category = QueryCategory.VISUAL
        elif is_tabular and is_comparative: category = QueryCategory.MULTI_STEP
        elif is_tabular: category = QueryCategory.TABULAR
        elif is_comparative: category = QueryCategory.COMPARATIVE

        years = [int(y) for y in re.findall(r"\b(20\d\d|19\d\d)\b", user_query)]
        filters = {}
        if years:
            filters["year"] = years if len(years) > 1 else years[0]

        sub_queries = [user_query]
        if is_comparative and len(years) >= 2:
            sub_queries = [
                f"{user_query} for {years[0]}",
                f"{user_query} for {years[1]}",
                f"Comparison of factors between {years[0]} and {years[1]}"
            ]

        adaptive_top_k = 12 if (is_comparative or is_tabular) else 6
        bm25_weight = 0.7 if (is_tabular or years) else 0.5
        vector_weight = 0.3 if (is_tabular or years) else 0.5

        return QueryPlan(
            intent=user_query,
            category=category,
            rewritten_query=user_query,
            expanded_queries=[user_query, f"{user_query} details summary"],
            sub_queries=sub_queries,
            filters=filters,
            adaptive_top_k=adaptive_top_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            needs_multi_step=(len(sub_queries) > 1),
            confidence=0.85
        )

    def plan(self, user_query: str, chat_history: str = "", mock_response: Optional[str] = None) -> QueryPlan:
        if mock_response:
            return QueryPlan.model_validate_json(mock_response)

        system_prompt = (
            "You are the master AI Query Planner for an Adaptive Token-Efficient RAG platform.\n"
            "Analyze the user query in the context of recent chat history.\n"
            "Output strict JSON conforming to this schema:\n"
            "{\n"
            '    "intent": "Core query intent",\n'
            '    "category": "simple" | "tabular" | "multi_step" | "comparative" | "temporal" | "visual" | "conversational",\n'
            '    "rewritten_query": "Fully resolved, standalone query removing ambiguity",\n'
            '    "expanded_queries": ["synonym 1", "keyword variation 2"],\n'
            '    "sub_queries": ["sub query 1", "sub query 2"],\n'
            '    "filters": {"document_name": "...", "year": 2024},\n'
            '    "adaptive_top_k": 6,\n'
            '    "bm25_weight": 0.5,\n'
            '    "vector_weight": 0.5,\n'
            '    "needs_multi_step": false,\n'
            '    "confidence": 0.9\n'
            "}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Chat History:\n{chat_history}\n\nCurrent Query: {user_query}"}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw_json = response.choices[0].message.content
            return QueryPlan.model_validate_json(raw_json)
        except Exception:
            return self._rule_based_fallback(user_query)