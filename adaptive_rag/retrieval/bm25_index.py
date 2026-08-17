import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List
from adaptive_rag.models.schema import Chunk

class BM25Index:
    """Persistent lexical keyword search engine for IDs, acronyms, and dates."""
    def __init__(self, persist_path: str = "./data/bm25_index.pkl"):
        self.persist_path = persist_path
        self.bm25 = None
        self.corpus_chunks: List[Chunk] = []
        self.load()

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks: return
        
        # Deduplicate incoming against existing (using chunk_id)
        existing_ids = {c.chunk_id for c in self.corpus_chunks}
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        
        if not new_chunks: return
        self.corpus_chunks.extend(new_chunks)
        
        tokenized_corpus = [self._tokenize(c.content) for c in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Returns ordered candidates by term frequency-inverse document frequency."""
        if not self.bm25 or not self.corpus_chunks: return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top scores
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0:
                results.append({
                    "chunk_id": self.corpus_chunks[idx].chunk_id,
                    "score": float(scores[idx])
                })
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.corpus_chunks}, f)

    def load(self):
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.corpus_chunks = data["chunks"]
