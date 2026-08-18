from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from adaptive_rag.retrieval.hybrid_search import HybridRetriever
from adaptive_rag.models.schema import Chunk, ContentType, QueryPlan, QueryCategory

class ParallelTableAwareRetriever:
    def __init__(self, hybrid_retriever: HybridRetriever, max_workers: int = 4):
        self.hybrid_retriever = hybrid_retriever
        self.max_workers = max_workers

    def retrieve_single(self, sub_query: str, top_k: int, is_tabular: bool = False) -> List[Chunk]:
        raw_chunks = self.hybrid_retriever.search(sub_query, top_k=top_k)
        if is_tabular:
            table_chunks = [c for c in raw_chunks if c.metadata.content_type == ContentType.TABLE]
            other_chunks = [c for c in raw_chunks if c.metadata.content_type != ContentType.TABLE]
            return table_chunks + other_chunks
        return raw_chunks

    def parallel_multi_query_search(self, plan: QueryPlan, active_documents: List[str] = None) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        is_tabular = (plan.category == QueryCategory.TABULAR)
        search_targets = list(dict.fromkeys(plan.sub_queries + plan.expanded_queries[:2]))
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_query = {
                executor.submit(self.retrieve_single, q, plan.adaptive_top_k, is_tabular): q 
                for q in search_targets
            }
            for future in as_completed(future_to_query):
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"[Warning] Parallel retrieval error: {e}")

        if active_documents and "All Documents" not in active_documents:
            all_chunks = [c for c in all_chunks if c.metadata.document_name in active_documents]

        unique_map = {c.chunk_id: c for c in all_chunks}
        return list(unique_map.values())
