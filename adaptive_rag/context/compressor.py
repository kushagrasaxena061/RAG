from typing import List
from adaptive_rag.models.schema import Chunk
from adaptive_rag.context.token_budget import TokenBudgetManager, TokenBudgetReport

class ContextCompressor:
    """Compresses the retrieved context to fit strictly within the Token Budget."""
    
    def __init__(self, token_manager: TokenBudgetManager):
        self.token_manager = token_manager

    def compress_context(self, chunks: List[Chunk], budget: TokenBudgetReport) -> List[Chunk]:
        """
        Greedily packs highest-ranked chunks until the token budget is reached.
        Assumes `chunks` is already sorted by relevance (e.g., by the Reranker).
        """
        selected_chunks = []
        current_tokens = 0

        for chunk in chunks:
            if self.token_manager.fits_in_budget(current_tokens, chunk.token_count, budget):
                selected_chunks.append(chunk)
                current_tokens += chunk.token_count
            else:
                # Token budget reached. 
                # Because the list is ordered by strict relevance, skipping the rest 
                # cuts out the lowest-quality noise while preserving the context window.
                break
                
        return selected_chunks
