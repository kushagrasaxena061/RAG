import time, uuid, re, concurrent.futures, json, hashlib
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
        return {"answer": "Sync endpoint disabled. Please use stream."}

    def process_query_stream(self, user_query: str, target_document: str = "All Documents", active_documents: List[str] = None, mock_mode: bool = False) -> Generator[str, None, None]:
        start_time = time.time()
        query_id = f"q_{uuid.uuid4().hex[:8]}"

        if SecuritySanitizer.is_malicious_query(user_query):
            yield json.dumps({"type": "metadata", "telemetry": {}}) + "\n---METADATA_END---\nSecurity Violation: Advanced Prompt Injection Detected.\n"
            return
            
        user_query = SecuritySanitizer.sanitize_text(user_query)

        # 3. CACHE IMPLEMENTATION (Normalizes spaces and casing)
        normalized_query = " ".join(user_query.strip().lower().split())
        cache_key = hashlib.sha256(f"{normalized_query}_{target_document}".encode("utf-8")).hexdigest()
        
        try:
            cursor = self.db.conn.execute("SELECT response FROM cache WHERE hash_key = ?", (cache_key,))
            row = cursor.fetchone()
            if row:
                cached_data = json.loads(row[0])
                telemetry_dict = cached_data.get("telemetry", {})
                telemetry_dict["cache_hit"] = True
                telemetry_dict["latency_ms"] = (time.time() - start_time) * 1000
                
                yield json.dumps({"type": "metadata", "telemetry": telemetry_dict}) + "\n---METADATA_END---\n"
                
                cached_answer = cached_data.get("answer", "")
                for word in cached_answer.split(" "):
                    yield word + " "
                    time.sleep(0.015)
                
                self.memory.add_interaction(user_query, cached_answer)
                return
        except Exception:
            pass # Fail gracefully if cache missing

        # FULL PIPELINE EXECUTION
        try:
            chat_history = self.memory.get_short_term_context(max_budget_tokens=400)
            contextualized_query = f"Chat History:\n{chat_history}\n\nCurrent Query: {user_query}" if chat_history else user_query

            try:
                plan = self.planner.plan(contextualized_query)
                sub_queries = getattr(plan, 'sub_queries', [user_query]) or [user_query]
            except Exception:
                sub_queries = [user_query]

            raw_chunks = []
            doc_prefix = f"{target_document} " if target_document != "All Documents" else ""
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(self.retriever.search, f"{doc_prefix}{sub_q}", top_k=5) for sub_q in sub_queries]
                for future in concurrent.futures.as_completed(futures):
                    raw_chunks.extend(future.result())

            if hasattr(self.retriever, 'vec_idx') and hasattr(self.retriever.vec_idx, 'get_chunks_by_ids'):
                parent_ids = list(set([getattr(c, 'parent_chunk_id', None) for c in raw_chunks if getattr(c, 'parent_chunk_id', None)]))
                if parent_ids:
                    parent_chunks = self.retriever.vec_idx.get_chunks_by_ids(parent_ids)
                    raw_chunks.extend(parent_chunks)

            # Filter by Target Document (Fixing the PDF selection issue)
            if target_document != "All Documents":
                unique_chunks = list({c.chunk_id: c for c in raw_chunks if c.metadata.document_name == target_document}.values())
            else:
                unique_chunks = list({c.chunk_id: c for c in raw_chunks}.values())

            if not unique_chunks:
                yield json.dumps({"type": "metadata", "telemetry": {}}) + "\n---METADATA_END---\nNo context found in the selected document to answer your query. Please check if the document uploaded successfully."
                return

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
            telemetry_dict["cache_hit"] = False
            
            # Guarantees the metadata separator is sent
            yield json.dumps({"type": "metadata", "telemetry": telemetry_dict}) + "\n---METADATA_END---\n"

            draft_answer = ""
            for token in self.generator.generate_stream(contextualized_query, compressed_chunks):
                draft_answer += token
                yield token

            self.memory.add_interaction(user_query, draft_answer)

            # SAVE TO CACHE
            try:
                self.db.conn.execute(
                    "INSERT OR REPLACE INTO cache (hash_key, response) VALUES (?, ?)", 
                    (cache_key, json.dumps({"answer": draft_answer, "telemetry": telemetry_dict}))
                )
                self.db.conn.commit()
            except Exception:
                pass

        except Exception as e:
            self.db.log_crash(e)
            # 4. ERROR FIX: Always send the metadata separator before the error so it shows up in the UI!
            yield json.dumps({"type": "metadata", "telemetry": {"error": "Internal Error"}}) + f"\n---METADATA_END---\n[An internal error occurred: {str(e)}]"