import time
import uuid
from typing import Dict, Any
from adaptive_rag.observability.tracker import QueryTelemetry, TelemetryLogger
from adaptive_rag.query.planner import LlamaQueryPlanner
from adaptive_rag.retrieval.hybrid_search import HybridRetriever
from adaptive_rag.reranking.cross_encoder import Reranker
from adaptive_rag.context.compressor import ContextCompressor
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.reasoning.generator import AnswerGenerator
from adaptive_rag.memory.manager import TwoTierMemory
from adaptive_rag.reasoning.contradiction import ContradictionDetector
from adaptive_rag.reasoning.temporal import TemporalReasoningEngine

class RAGPipelineOrchestrator:
    def __init__(self, hybrid_retriever: HybridRetriever, token_manager: TokenBudgetManager, memory: TwoTierMemory, logger: TelemetryLogger = None):
        self.logger = logger or TelemetryLogger()
        self.retriever = hybrid_retriever
        self.token_manager = token_manager
        self.memory = memory
        
        self.planner = LlamaQueryPlanner()
        self.reranker = Reranker()
        self.compressor = ContextCompressor(token_manager)
        self.generator = AnswerGenerator()
        
        # New Reasoning Engines
        self.contradiction_detector = ContradictionDetector()
        self.temporal_engine = TemporalReasoningEngine()

    def _is_conversational(self, query: str) -> bool:
        greetings = {"hello", "hi", "hey", "help", "who are you", "what can you do", "thanks", "thank you"}
        return query.strip().lower() in greetings or len(query.split()) <= 1

    def process_query(self, user_query: str, mock_mode: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        
        if self._is_conversational(user_query):
            answer = "Hello! I am your Adaptive Token-Efficient AI Assistant. Upload documents via the sidebar or ask questions about your indexed files."
            self.memory.add_interaction(user_query, answer)
            telemetry = QueryTelemetry(query_id=query_id, query_text=user_query, latency_ms=(time.time() - start_time) * 1000)
            self.logger.log(telemetry)
            return {"query_id": query_id, "answer": answer, "telemetry": telemetry.model_dump()}

        chat_history = self.memory.get_short_term_context(max_budget_tokens=500)
        contextualized_query = f"Chat History:\n{chat_history}\n\nCurrent Query: {user_query}" if chat_history else user_query
        
        raw_chunks = self.retriever.search(user_query, top_k=8)
        reranked = self.reranker.rerank(user_query, raw_chunks, top_k=5)
        
        # Temporal Ordering: Sort evidence chronologically before compressing
        chronological_chunks = self.temporal_engine.sort_chunks_by_temporal_order(reranked)
        
        history_tokens = self.token_manager.count_tokens(chat_history) if chat_history else 0
        budget = self.token_manager.calculate_budget(query=user_query, conversation_history_tokens=history_tokens)
        compressed_chunks = self.compressor.compress_context(chronological_chunks, budget)
        
        # Contradiction Detection: Check the final context for conflicting evidence
        conflict_report = self.contradiction_detector.detect_conflicts(user_query, compressed_chunks)
        if conflict_report.has_contradiction:
            contextualized_query += f"\n\n[SYSTEM WARNING]: Conflicting data detected in sources. {conflict_report.discrepancy_summary} {conflict_report.resolution_advice}"

        final_answer = self.generator.generate(contextualized_query, compressed_chunks)
        self.memory.add_interaction(user_query, final_answer)

        latency = (time.time() - start_time) * 1000
        retrieved_count = len(raw_chunks)
        final_count = len(compressed_chunks)
        
        telemetry = QueryTelemetry(
            query_id=query_id, query_text=user_query, latency_ms=latency,
            retrieval_rounds=1, retrieved_chunks=retrieved_count,
            final_chunks_used=final_count, compression_ratio=final_count / retrieved_count if retrieved_count > 0 else 0,
            metadata={"contradiction_detected": conflict_report.has_contradiction}
        )
        self.logger.log(telemetry)
        
        return {"query_id": query_id, "answer": final_answer, "telemetry": telemetry.model_dump()}
