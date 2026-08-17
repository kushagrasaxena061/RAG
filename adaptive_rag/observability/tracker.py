import time
from typing import Dict, Any
from pydantic import BaseModel, Field

class QueryTelemetry(BaseModel):
    """Tracks performance and token efficiency metrics for a single query."""
    query_id: str
    query_text: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    retrieval_rounds: int = 0
    retrieved_chunks: int = 0
    final_chunks_used: int = 0
    compression_ratio: float = 0.0  # (final_chunks / retrieved_chunks)
    correction_loops_triggered: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TelemetryLogger:
    """Records and outputs observability metrics."""
    def __init__(self):
        self.records = []

    def log(self, record: QueryTelemetry):
        self.records.append(record)
        print(f"\n[TELEMETRY LOG] Query '{record.query_text[:30]}...'")
        print(f" ├─ Latency: {record.latency_ms:.2f}ms")
        print(f" ├─ Efficiency: {record.total_tokens} total tokens used")
        print(f" ├─ Compression Ratio: {record.compression_ratio:.2%}")
        print(f" └─ Retrieval Rounds: {record.retrieval_rounds} | Corrections: {record.correction_loops_triggered}\n")
