import hashlib
import time
from typing import Optional, Dict, Any

class MultiTierCache:
    def __init__(self, ttl_seconds: int = 86400):
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.plan_cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds

    def _hash_key(self, key_str: str) -> str:
        return hashlib.sha256(key_str.strip().lower().encode("utf-8")).hexdigest()

    def get_query_response(self, query: str, active_doc_hash: str) -> Optional[Dict[str, Any]]:
        key = self._hash_key(f"{query}:{active_doc_hash}")
        if key in self.query_cache:
            entry = self.query_cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            del self.query_cache[key]
        return None

    def set_query_response(self, query: str, active_doc_hash: str, response: Dict[str, Any]):
        key = self._hash_key(f"{query}:{active_doc_hash}")
        self.query_cache[key] = {"timestamp": time.time(), "data": response}

    def get_plan(self, query: str) -> Optional[Any]:
        key = self._hash_key(query)
        if key in self.plan_cache:
            entry = self.plan_cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["plan"]
        return None

    def set_plan(self, query: str, plan: Any):
        key = self._hash_key(query)
        self.plan_cache[key] = {"timestamp": time.time(), "plan": plan}

    def clear(self):
        self.query_cache.clear()
        self.plan_cache.clear()
