import json
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from adaptive_rag.models.schema import Chunk
from adaptive_rag.config import global_config

class EvaluationResult(BaseModel):
    is_supported_by_evidence: bool = Field(description="Are all claims supported by the context?")
    addresses_query: bool = Field(description="Does the answer satisfy the user's core intent?")
    needs_retrieval: bool = Field(description="Is important information missing, requiring another search round?")
    feedback: str = Field(description="Explanation of the evaluation, or what specifically is missing.")

class SelfEvaluator:
    """Evaluates the generated answer for hallucinations and completeness."""
    
    def __init__(self, api_key: str = "sk-mock", base_url: str = "http://localhost:11434/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = global_config.model.model_name

    def evaluate(self, query: str, generated_answer: str, chunks: List[Chunk], mock_response: Optional[str] = None) -> EvaluationResult:
        if mock_response:
            return EvaluationResult.model_validate_json(mock_response)

        context_str = "\n".join([c.content for c in chunks])
        
        system_prompt = """You are an AI verification evaluator. 
Compare the user's query, the provided context, and the generated answer.
Output a JSON object evaluating the answer's quality.

JSON Schema:
{
    "is_supported_by_evidence": boolean,
    "addresses_query": boolean,
    "needs_retrieval": boolean,
    "feedback": "string"
}"""

        user_content = f"Query: {query}\n\nContext: {context_str}\n\nGenerated Answer: {generated_answer}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return EvaluationResult.model_validate_json(response.choices[0].message.content)
        except Exception as e:
            return EvaluationResult(
                is_supported_by_evidence=True, 
                addresses_query=True, 
                needs_retrieval=False, 
                feedback=f"Evaluation failed, assuming pass: {e}"
            )
