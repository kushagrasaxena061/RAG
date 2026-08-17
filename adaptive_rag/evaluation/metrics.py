from typing import List, Set

class RetrievalMetrics:
    """Calculates standard information retrieval performance metrics."""
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Measures the proportion of relevant chunks successfully retrieved in the top K."""
        if not relevant_ids:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Calculates Mean Reciprocal Rank (how high up the first relevant chunk appeared)."""
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Measures the proportion of the top K retrieved chunks that are actually relevant."""
        if k <= 0:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
        return hits / k
