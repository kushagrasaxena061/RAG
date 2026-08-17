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
from adaptive_rag.reasoning.evaluator import SelfEvaluator
from adaptive_rag.reasoning.correction_loop import CorrectionEngine

class RAGPipelineOrchestrator:
    def __init__(self, hybrid_retriever: HybridRetriever, token_manager: TokenBudgetManager, logger: TelemetryLogger = None):
        self.logger = logger or TelemetryLogger()
        self.retriever = hybrid_retriever
        self.token_manager = token_manager
        
        self.planner = LlamaQueryPlanner()
        self.reranker = Reranker()
        self.compressor = ContextCompressor(token_manager)
        
        generator = AnswerGenerator()
        evaluator = SelfEvaluator()
        self.correction_engine = CorrectionEngine(generator, evaluator)

    def process_query(self, user_query: str, mock_mode: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        
        echo_log = []
        
        plan = self.planner.plan(user_query)
        
        raw_chunks = []
        for sub_q in plan.sub_queries:
            raw_chunks.extend(self.retriever.search(sub_q, top_k=10))
            
        unique_chunks = {c.chunk_id: c for c in raw_chunks}.values()
        
        reranked = self.reranker.rerank(user_query, list(unique_chunks))
        
        budget = self.token_manager.calculate_budget(query=user_query)
        compressed_chunks = self.compressor.compress_context(reranked, budget)
        
        def retrieve_more(feedback: str):
            return self.retriever.search(feedback, top_k=5)
            
        final_answer, final_context = self.correction_engine.execute(
            query=user_query,
            initial_chunks=compressed_chunks,
            retrieve_more_callback=retrieve_more
        )

        latency = (time.time() - start_time) * 1000
        retrieved_count = len(unique_chunks)
        final_count = len(final_context)
        compression_ratio = final_count / retrieved_count if retrieved_count > 0 else 0
        
        telemetry = QueryTelemetry(
            query_id=query_id, query_text=user_query, latency_ms=latency,
            retrieval_rounds=1, retrieved_chunks=retrieved_count,
            final_chunks_used=final_count, compression_ratio=compression_ratio
        )
        self.logger.log(telemetry)
        
        return {"query_id": query_id, "answer": final_answer, "telemetry": telemetry.model_dump()}
