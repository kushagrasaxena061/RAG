import time
from typing import Dict, Any, List
from adaptive_rag.retrieval.vector_index import VectorIndex
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.reasoning.generator import AnswerGenerator
from adaptive_rag.models.schema import Chunk

class NaiveBaselineRAG:
    """
    Conventional Baseline RAG:
    Blindly retrieves top-K chunks without query planning, reranking, or token compression.
    """
    def __init__(self, vector_index: VectorIndex, token_manager: TokenBudgetManager):
        self.vector_index = vector_index
        self.token_manager = token_manager
        self.generator = AnswerGenerator()

    def process_query(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Naive Vector Search (No BM25, No Reranker)
        results = self.vector_index.search(query, top_k=top_k)
        
        # In-memory document lookup simulation for baseline
        chunks = []
        for r in results:
            content = f"Retrieved chunk content for doc {r['chunk_id']}"
            chunks.append(Chunk(
                chunk_id=r["chunk_id"],
                chunk_type="child",
                content=content,
                token_count=self.token_manager.count_tokens(content),
                content_hash=r["chunk_id"],
                metadata={
                    "document_id": "doc",
                    "document_name": "doc.pdf",
                    "page_number": 1,
                    "created_at": 0.0
                }
            ))

        # 2. Blind Context Concatenation (Uncompressed)
        context_str = self.generator.format_context(chunks)
        input_tokens = self.token_manager.count_tokens(context_str) + self.token_manager.count_tokens(query) + 200

        # 3. Direct LLM Call
        answer = self.generator.generate(query, chunks)
        output_tokens = self.token_manager.count_tokens(answer)
        latency = (time.time() - start_time) * 1000

        return {
            "query": query,
            "answer": answer,
            "latency_ms": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "chunks_used": len(chunks)
        }
