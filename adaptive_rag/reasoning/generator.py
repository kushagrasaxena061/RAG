from typing import List, Optional
from openai import OpenAI
from adaptive_rag.models.schema import Chunk
from adaptive_rag.config import global_config

class AnswerGenerator:
    """Generates final answers while strictly enforcing citations based on chunk metadata."""
    
    def __init__(self, api_key: str = "sk-mock", base_url: str = "http://localhost:11434/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = global_config.model.model_name

    def format_context(self, chunks: List[Chunk]) -> str:
        """Formats the compressed chunks into a numbered context block for the LLM."""
        context_str = ""
        for i, c in enumerate(chunks):
            doc_name = c.metadata.document_name
            page = c.metadata.page_number
            context_str += f"--- Source {i+1} ---\nOrigin: [{doc_name}, p. {page}]\nContent: {c.content}\n\n"
        return context_str

    def generate(self, query: str, chunks: List[Chunk], mock_response: Optional[str] = None) -> str:
        if mock_response:
            return mock_response
            
        context_str = self.format_context(chunks)
        
        system_prompt = """You are a highly precise AI reasoning system.
Your task is to answer the user's query using ONLY the provided Source context.
CRITICAL INSTRUCTION: Every factual claim MUST be followed by an exact citation using the Origin tag provided.
Example: 'Revenue grew by 15% [annual_report_2025.pdf, p. 14].'
If the provided sources do not contain enough information to fully answer the query, explicitly state what is missing. DO NOT guess or hallucinate."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Warning] Generator failed: {e}"
