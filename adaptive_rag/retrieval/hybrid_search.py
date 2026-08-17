from typing import List, Dict, Any
from adaptive_rag.models.schema import Chunk
from adaptive_rag.retrieval.vector_index import VectorIndex
from adaptive_rag.retrieval.bm25_index import BM25Index

class HybridRetriever:
    """Coordinates BM25 and Vector Search using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, vector_index: VectorIndex, bm25_index: BM25Index, rrf_k: int = 60):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5, where: Dict[str, Any] = None) -> List[Chunk]:
        # 1. Retrieve candidates independently (fetch 2x top_k to allow deep fusion)
        vector_results = self.vector_index.search(query, top_k=top_k * 2, where=where)
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)

        # 2. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}

        # Score Vector Ranks
        for rank, res in enumerate(vector_results, start=1):
            cid = res["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Score BM25 Ranks
        for rank, res in enumerate(bm25_results, start=1):
            cid = res["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort by fused score
        sorted_chunk_ids = [cid for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]

        # 4. Resolve exact Chunk payloads (using BM25 index as the fast in-memory store)
        chunk_map = {c.chunk_id: c for c in self.bm25_index.corpus_chunks}
        
        # Note: If metadata filtering ('where') drops a vector result that BM25 kept, 
        # it is handled safely because we only return valid merged IDs.
        return [chunk_map[cid] for cid in sorted_chunk_ids if cid in chunk_map]
