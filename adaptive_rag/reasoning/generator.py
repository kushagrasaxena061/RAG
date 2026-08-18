import time
from typing import List, Optional, Generator
from openai import OpenAI
from adaptive_rag.models.schema import Chunk
from adaptive_rag.config import global_config

class AnswerGenerator:
    def __init__(self, api_key: str = "sk-mock", base_url: str = "http://localhost:11434/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=25.0)
        self.default_model = global_config.model.model_name

    def format_context(self, chunks: List[Chunk]) -> str:
        context_str = ""
        for i, c in enumerate(chunks):
            doc_name = getattr(c.metadata, 'document_name', 'document.pdf')
            page = getattr(c.metadata, 'page_number', 1)
            context_str += f"--- Source {i+1} ---\nOrigin: [{doc_name}, p. {page}]\nContent: {c.content}\n\n"
        return context_str

    def _fallback_extractive_answer(self, query: str, chunks: List[Chunk]) -> str:
        if not chunks:
            return "No relevant context found in the uploaded documents to answer this question."
        extracted = []
        for c in chunks[:4]:
            doc_name = getattr(c.metadata, 'document_name', 'document.pdf')
            page = getattr(c.metadata, 'page_number', 1)
            content = c.content.strip()
            if content:
                extracted.append(f"{content} [{doc_name}, p. {page}]")
        return "Based on the retrieved document context:\n\n" + "\n\n".join(extracted)

    def _build_prompt_payload(self, query: str, chunks: List[Chunk]):
        context_str = self.format_context(chunks)
        system_prompt = (
            "You are a highly precise AI reasoning system.\n"
            "Your task is to answer the user's query using ONLY the provided Source context.\n"
            "CRITICAL INSTRUCTION: Every factual claim MUST be followed by an exact citation using the Origin tag provided.\n"
            "Example: 'Basic salary was $8,000 [financials.pdf, p. 1].'\n"
            "If the provided sources do not contain enough information, state what is missing. DO NOT hallucinate."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"}
        ]

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
            return response.choices[0].message.content or ""
        except Exception:
            return self._fallback_extractive_answer(query, chunks)

    def generate_stream(self, query: str, chunks: List[Chunk], model_name: Optional[str] = None, temperature: float = 0.2) -> Generator[str, None, None]:
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
        except Exception:
            fallback = self._fallback_extractive_answer(query, chunks)
            for word in fallback.split(" "):
                yield word + " "
                time.sleep(0.015)