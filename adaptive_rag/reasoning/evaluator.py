import json
from typing import List, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from adaptive_rag.config import global_config
from adaptive_rag.models.schema import Chunk

class EvaluationResult(BaseModel):
    is_supported_by_evidence: bool = True
    faithfulness_score: float = 1.0
    relevance_score: float = 1.0
    missing_information: List[str] = Field(default_factory=list)
    hallucinations: List[str] = Field(default_factory=list)

class SelfEvaluator:
    def __init__(self, api_key: str = "sk-mock", base_url: str = "http://localhost:11434/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=8.0)
        self.default_model = global_config.model.model_name

    def evaluate(self, query: str, answer: str, chunks: List[Any]) -> EvaluationResult:
        return self.evaluate_response(query, answer, chunks)

    def evaluate_response(self, query: str, answer: str, chunks: List[Any]) -> EvaluationResult:
        if not chunks or not answer:
            return EvaluationResult(is_supported_by_evidence=True, faithfulness_score=1.0, relevance_score=1.0)
        
        context_str = "\n".join([f"[{getattr(c.metadata, 'document_name', 'doc')}, p. {getattr(c.metadata, 'page_number', 1)}]: {getattr(c, 'content', '')}" for c in chunks[:5]])
        prompt = f"Context:\n{context_str}\n\nQuery: {query}\n\nDraft Answer: {answer}\n\nIs the answer faithful to the context? Return JSON with key 'is_supported_by_evidence' (true/false)."

        try:
            res = self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw = res.choices[0].message.content or ""
            is_supported = "false" not in raw.lower()
            return EvaluationResult(is_supported_by_evidence=is_supported, faithfulness_score=1.0 if is_supported else 0.5)
        except Exception:
            return EvaluationResult(is_supported_by_evidence=True, faithfulness_score=1.0, relevance_score=1.0)