from sentence_transformers import CrossEncoder
from typing import List
from adaptive_rag.models.schema import Chunk

class Reranker:
    """Uses a Cross-Encoder to highly accurately score query-document relevance."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # We use a fast, lightweight cross-encoder optimized for RAG.
        # It evaluates the query and chunk simultaneously rather than separately.
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = None) -> List[Chunk]:
        if not chunks:
            return []
        
        # Prepare [Query, Document] pairs for the cross-encoder
        pairs = [[query, chunk.content] for chunk in chunks]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)
        
        # Zip chunks with their new extreme-precision scores and sort descending
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the sorted chunks
        reranked = [chunk for chunk, score in scored_chunks]
        
        if top_k:
            reranked = reranked[:top_k]
            
        return reranked
