import torch
from sentence_transformers import CrossEncoder
from typing import List
from adaptive_rag.models.schema import Chunk

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Auto-detect Apple Silicon GPU (MPS) or CUDA
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        self.model = CrossEncoder(model_name, max_length=512, device=device)

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Chunk]:
        if not chunks:
            return []
        
        # Limit candidate pool to top 8 to prevent cross-encoder latency spikes
        candidates = chunks[:8]
        pairs = [[query, chunk.content] for chunk in candidates]
        
        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)
        scored_chunks = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in scored_chunks[:top_k]]
