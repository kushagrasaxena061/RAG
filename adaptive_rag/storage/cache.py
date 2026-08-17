import hashlib
import time
from typing import Optional, Dict, Any

class QueryResponseCache:
    """Exact and semantic response cache with document-state invalidation."""
    def __init__(self, ttl_seconds: int = 86400):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def _generate_key(self, query: str, active_doc_hash: str) -> str:
        raw_key = f"{query.strip().lower()}:{active_doc_hash}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, query: str, active_doc_hash: str) -> Optional[Dict[str, Any]]:
        key = self._generate_key(query, active_doc_hash)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["data"]
            del self.cache[key]
        return None

    def set(self, query: str, active_doc_hash: str, response_data: Dict[str, Any]):
        key = self._generate_key(query, active_doc_hash)
        self.cache[key] = {
            "timestamp": time.time(),
            "data": response_data
        }

    def clear(self):
        self.cache.clear()
