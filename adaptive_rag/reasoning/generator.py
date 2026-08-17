from typing import List, Optional, Generator
from openai import OpenAI
from adaptive_rag.models.schema import Chunk
from adaptive_rag.config import global_config

class AnswerGenerator:
    """Generates answers with strict citations, supporting dynamic model swapping."""
    
    def __init__(self, api_key: str = "sk-mock", base_url: str = "http://localhost:11434/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = global_config.model.model_name

    def format_context(self, chunks: List[Chunk]) -> str:
        context_str = ""
        for i, c in enumerate(chunks):
            doc_name = c.metadata.document_name
            page = c.metadata.page_number
            context_str += f"--- Source {i+1} ---\nOrigin: [{doc_name}, p. {page}]\nContent: {c.content}\n\n"
        return context_str

    def _build_prompt_payload(self, query: str, chunks: List[Chunk]):
        context_str = self.format_context(chunks)
        system_prompt = (
            "You are a highly precise AI reasoning system.\n"
            "Your task is to answer the user's query using ONLY the provided Source context.\n"
            "CRITICAL INSTRUCTION: Every factual claim MUST be followed by an exact citation using the Origin tag provided.\n"
            "Example: 'Basic salary was $8,000 [financials.pdf, p. 1].'\n"
            "If the provided sources do not contain enough information, state what is missing. DO NOT hallucinate."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"}
        ]
        return messages

    def generate(self, query: str, chunks: List[Chunk], model_name: Optional[str] = None, temperature: float = 0.2, mock_response: Optional[str] = None) -> str:
        if mock_response:
            return mock_response
        try:
            messages = self._build_prompt_payload(query, chunks)
            response = self.client.chat.completions.create(
                model=model_name or self.default_model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Warning] Generator error: {e}"

    def generate_stream(self, query: str, chunks: List[Chunk], model_name: Optional[str] = None, temperature: float = 0.2) -> Generator[str, None, None]:
        """Yields LLM tokens one by one in real-time for zero-latency UX."""
        try:
            messages = self._build_prompt_payload(query, chunks)
            stream = self.client.chat.completions.create(
                model=model_name or self.default_model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield token
        except Exception as e:
            yield f"[Stream Error: {e}]"
