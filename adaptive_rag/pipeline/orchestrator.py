import time, uuid, re, concurrent.futures, json
from typing import Dict, Any, List, Generator
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
from adaptive_rag.reasoning.evaluator import SelfEvaluator
from adaptive_rag.storage.sqlite_db import SQLiteManager
from adaptive_rag.security.sanitizer import SecuritySanitizer

class RAGPipelineOrchestrator:
    def __init__(self, hybrid_retriever: HybridRetriever, token_manager: TokenBudgetManager, memory: TwoTierMemory, db: SQLiteManager, logger: TelemetryLogger = None):
        self.logger = logger or TelemetryLogger()
        self.retriever, self.token_manager, self.memory, self.db = hybrid_retriever, token_manager, memory, db
        self.planner, self.reranker, self.compressor = LlamaQueryPlanner(), Reranker(), ContextCompressor(token_manager)
        self.generator, self.evaluator = AnswerGenerator(), SelfEvaluator()
        self.contradiction_detector, self.temporal_engine = ContradictionDetector(), TemporalReasoningEngine()

    def verify_citations(self, answer: str, compressed_chunks: List) -> str:
        citations = re.findall(r"\[(.*?), p\. (\d+)\]", answer)
        valid_sources = [(getattr(c.metadata, 'document_name', ''), str(getattr(c.metadata, 'page_number', ''))) for c in compressed_chunks]
        for doc, page in citations:
            if (doc, page) not in valid_sources:
                answer += f"\n\n[SYSTEM WARNING]: Citation [{doc}, p. {page}] hallucinated. Evidence missing from compressed context."
        return answer

    def process_query(self, user_query: str, target_document: str = "All Documents", active_documents: List[str] = None, mock_mode: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        
        if SecuritySanitizer.is_malicious_query(user_query):
            return {"query_id": query_id, "answer": "Security Violation: Advanced Prompt Injection Detected.", "telemetry": {}}
        user_query = SecuritySanitizer.sanitize_text(user_query)

        try:
            chat_history = self.memory.get_short_term_context(max_budget_tokens=400)
            contextualized_query = f"Chat History:\n{chat_history}\n\nCurrent Query: {user_query}" if chat_history else user_query

            try:
                plan = self.planner.plan(contextualized_query)
                sub_queries = getattr(plan, 'sub_queries', [user_query]) or [user_query]
            except Exception:
                sub_queries = [user_query]

            raw_chunks = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(self.retriever.search, sub_q, top_k=5) for sub_q in sub_queries]
                for future in concurrent.futures.as_completed(futures):
                    raw_chunks.extend(future.result())

            if hasattr(self.retriever, 'vec_idx') and hasattr(self.retriever.vec_idx, 'get_chunks_by_ids'):
                parent_ids = list(set([getattr(c, 'parent_chunk_id', None) for c in raw_chunks if getattr(c, 'parent_chunk_id', None)]))
                if parent_ids:
                    parent_chunks = self.retriever.vec_idx.get_chunks_by_ids(parent_ids)
                    raw_chunks.extend(parent_chunks)

            unique_chunks = list({c.chunk_id: c for c in raw_chunks}.values())
            reranked = self.reranker.rerank(user_query, unique_chunks, top_k=8)
            ordered_chunks = self.temporal_engine.sort_chunks_by_temporal_order(reranked)

            history_tokens = self.token_manager.count_tokens(chat_history) if chat_history else 0
            budget = self.token_manager.calculate_budget(query=user_query, conversation_history_tokens=history_tokens)
            compressed_chunks = self.compressor.compress_context(ordered_chunks, budget)

            conflict_report = self.contradiction_detector.detect_conflicts(user_query, compressed_chunks)
            if conflict_report.has_contradiction:
                contextualized_query += f"\n\n[SYSTEM NOTICE]: {conflict_report.discrepancy_summary} {conflict_report.resolution_advice}"

            draft_answer = self.generator.generate(contextualized_query, compressed_chunks)
            eval_result = self.evaluator.evaluate(user_query, draft_answer, compressed_chunks)
            
            if not getattr(eval_result, 'is_supported_by_evidence', True):
                draft_answer = self.generator.generate(contextualized_query + "\nEnsure you only use the provided context.", compressed_chunks)

            verified_answer = self.verify_citations(draft_answer, compressed_chunks)
            self.memory.add_interaction(user_query, verified_answer)

            latency = (time.time() - start_time) * 1000
            telemetry = QueryTelemetry(
                query_id=query_id, query_text=user_query, latency_ms=latency,
                retrieval_rounds=1, retrieved_chunks=len(unique_chunks),
                final_chunks_used=len(compressed_chunks),
                compression_ratio=len(compressed_chunks) / len(unique_chunks) if unique_chunks else 0
            )
            self.logger.log(telemetry)
            return {"query_id": query_id, "answer": verified_answer, "telemetry": telemetry.model_dump() if hasattr(telemetry, 'model_dump') else telemetry.dict()}

        except Exception as e:
            self.db.log_crash(e)
            return {"query_id": query_id, "answer": f"An internal error occurred and was securely logged. Error: {str(e)}", "telemetry": {}}

    def process_query_stream(self, user_query: str, target_document: str = "All Documents", active_documents: List[str] = None, mock_mode: bool = False) -> Generator[str, None, None]:
        start_time = time.time()
        query_id = f"q_{uuid.uuid4().hex[:8]}"

        if SecuritySanitizer.is_malicious_query(user_query):
            yield json.dumps({"type": "metadata", "telemetry": {}}) + "\n---METADATA_END---\nSecurity Violation: Advanced Prompt Injection Detected.\n"
            return
            
        user_query = SecuritySanitizer.sanitize_text(user_query)

        try:
            chat_history = self.memory.get_short_term_context(max_budget_tokens=400)
            contextualized_query = f"Chat History:\n{chat_history}\n\nCurrent Query: {user_query}" if chat_history else user_query

            # Robust search retrieval
            raw_chunks = []
            doc_prefix = f"{target_document} " if target_document != "All Documents" else ""
            raw_chunks.extend(self.retriever.search(f"{doc_prefix}{user_query}", top_k=8))

            if hasattr(self.retriever, 'vec_idx') and hasattr(self.retriever.vec_idx, 'get_chunks_by_ids'):
                parent_ids = list(set([getattr(c, 'parent_chunk_id', None) for c in raw_chunks if getattr(c, 'parent_chunk_id', None)]))
                if parent_ids:
                    parent_chunks = self.retriever.vec_idx.get_chunks_by_ids(parent_ids)
                    raw_chunks.extend(parent_chunks)

            unique_chunks = list({c.chunk_id: c for c in raw_chunks}.values())
            reranked = self.reranker.rerank(user_query, unique_chunks, top_k=6)
            ordered_chunks = self.temporal_engine.sort_chunks_by_temporal_order(reranked)

            history_tokens = self.token_manager.count_tokens(chat_history) if chat_history else 0
            budget = self.token_manager.calculate_budget(query=user_query, conversation_history_tokens=history_tokens)
            compressed_chunks = self.compressor.compress_context(ordered_chunks, budget)

            latency = (time.time() - start_time) * 1000
            telemetry = QueryTelemetry(
                query_id=query_id, query_text=user_query, latency_ms=latency,
                retrieval_rounds=1, retrieved_chunks=len(unique_chunks),
                final_chunks_used=len(compressed_chunks),
                compression_ratio=len(compressed_chunks) / len(unique_chunks) if unique_chunks else 0
            )
            self.logger.log(telemetry)

            telemetry_dict = telemetry.model_dump() if hasattr(telemetry, 'model_dump') else telemetry.dict()
            yield json.dumps({"type": "metadata", "telemetry": telemetry_dict}) + "\n---METADATA_END---\n"

            draft_answer = ""
            for token in self.generator.generate_stream(contextualized_query, compressed_chunks):
                draft_answer += token
                yield token

            self.memory.add_interaction(user_query, draft_answer)

        except Exception as e:
            self.db.log_crash(e)
            yield f"\n[An internal error occurred: {str(e)}]"