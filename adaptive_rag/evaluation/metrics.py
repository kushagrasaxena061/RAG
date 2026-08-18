import math
from typing import List, Set

class RetrievalMetrics:
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
        if not relevant_ids: return 0.0
        hits = sum(1 for cid in retrieved_ids[:k] if cid in relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
        if k <= 0: return 0.0
        hits = sum(1 for cid in retrieved_ids[:k] if cid in relevant_ids)
        return hits / k

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
        dcg = 0.0
        for i, cid in enumerate(retrieved_ids[:k]):
            if cid in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)
        
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_efficiency_ratio(quality_score: float, tokens_used: int) -> float:
        token_k = max(tokens_used, 1) / 1000.0
        return round(quality_score / token_k, 3)
