import time
import uuid
from typing import Dict, Any, List, Optional
from adaptive_rag.observability.tracker import QueryTelemetry, TelemetryLogger
from adaptive_rag.query.planner import LlamaQueryPlanner
from adaptive_rag.retrieval.hybrid_search import HybridRetriever
from adaptive_rag.retrieval.multimodal_index import MultimodalVisualIndex
from adaptive_rag.reranking.cross_encoder import Reranker
from adaptive_rag.context.compressor import ContextCompressor
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.reasoning.generator import AnswerGenerator
from adaptive_rag.memory.manager import TwoTierMemory
from adaptive_rag.reasoning.contradiction import ContradictionDetector
from adaptive_rag.reasoning.temporal import TemporalReasoningEngine
from adaptive_rag.models.schema import Chunk, ChunkMetadata, ChunkType, ContentType

class RAGPipelineOrchestrator:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        visual_index: MultimodalVisualIndex,
        token_manager: TokenBudgetManager,
        memory: TwoTierMemory,
        logger: TelemetryLogger = None
    ):
        self.logger = logger or TelemetryLogger()
        self.retriever = hybrid_retriever
        self.visual_index = visual_index
        self.token_manager = token_manager
        self.memory = memory
        
        self.planner = LlamaQueryPlanner()
        self.reranker = Reranker()
        self.compressor = ContextCompressor(token_manager)
        self.generator = AnswerGenerator()
        self.contradiction_detector = ContradictionDetector()
        self.temporal_engine = TemporalReasoningEngine()

    def _is_conversational(self, query: str) -> bool:
        greetings = {"hello", "hi", "hey", "help", "who are you", "what can you do", "thanks", "thank you"}
        return query.strip().lower() in greetings

    def _is_visual_query(self, query: str) -> bool:
        visual_keywords = {"chart", "graph", "figure", "image", "diagram", "plot", "visual", "trend", "illustration"}
        tokens = set(query.lower().split())
        return len(tokens.intersection(visual_keywords)) > 0

    def process_query(
        self,
        user_query: str,
        target_document: str = "All Documents",
        active_documents: List[str] = None,
        mock_mode: bool = False
    ) -> Dict[str, Any]:
        start_time = time.time()
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        
        if self._is_conversational(user_query):
            answer = "Hello! I am your Adaptive Token-Efficient AI Assistant. Ask questions across your documents, tables, or charts."
            self.memory.add_interaction(user_query, answer)
            telemetry = QueryTelemetry(query_id=query_id, query_text=user_query, latency_ms=(time.time() - start_time) * 1000)
            self.logger.log(telemetry)
            return {"query_id": query_id, "answer": answer, "telemetry": telemetry.model_dump()}

        chat_history = self.memory.get_short_term_context(max_budget_tokens=400)
        contextualized_query = f"Chat History:\n{chat_history}\n\nCurrent Query: {user_query}" if chat_history else user_query

        raw_chunks: List[Chunk] = []

        if target_document == "All Documents" and active_documents and len(active_documents) > 1:
            per_doc_k = max(3, 12 // len(active_documents))
            for doc_name in active_documents:
                doc_query = f"Document {doc_name} overview table summary {user_query}"
                doc_hits = self.retriever.search(doc_query, top_k=per_doc_k)
                matched = [c for c in doc_hits if c.metadata.document_name == doc_name]
                if not matched:
                    matched = [c for c in self.retriever.search(user_query, top_k=per_doc_k) if c.metadata.document_name == doc_name]
                raw_chunks.extend(matched)
        elif target_document != "All Documents":
            doc_hits = self.retriever.search(f"{target_document} {user_query}", top_k=15)
            raw_chunks = [c for c in doc_hits if c.metadata.document_name == target_document]
            if len(raw_chunks) < 3:
                raw_chunks.extend([c for c in self.retriever.search(user_query, top_k=10) if c.metadata.document_name == target_document])
        else:
            raw_chunks = self.retriever.search(user_query, top_k=15)

        if self._is_visual_query(user_query) or len(raw_chunks) < 4:
            visual_hits = self.visual_index.search_visual(user_query, top_k=3, target_document=target_document)
            for v in visual_hits:
                meta = ChunkMetadata(
                    document_id="visual_doc",
                    document_name=v["metadata"].get("document_name", "document.pdf"),
                    page_number=v["metadata"].get("page_number", 1),
                    section_title=v["metadata"].get("section_title", "Visual Evidence"),
                    content_type=ContentType.IMAGE,
                    created_at=0.0
                )
                raw_chunks.append(Chunk(
                    chunk_id=v["chunk_id"],
                    chunk_type=ChunkType.CHILD,
                    content=v["content"],
                    token_count=len(v["content"].split()),
                    content_hash=v["chunk_id"],
                    metadata=meta
                ))

        unique_chunks_map = {c.chunk_id: c for c in raw_chunks}
        unique_chunks = list(unique_chunks_map.values())

        reranked = self.reranker.rerank(user_query, unique_chunks, top_k=8)
        ordered_chunks = self.temporal_engine.sort_chunks_by_temporal_order(reranked)

        history_tokens = self.token_manager.count_tokens(chat_history) if chat_history else 0
        budget = self.token_manager.calculate_budget(query=user_query, conversation_history_tokens=history_tokens)
        compressed_chunks = self.compressor.compress_context(ordered_chunks, budget)

        conflict_report = self.contradiction_detector.detect_conflicts(user_query, compressed_chunks)
        if conflict_report.has_contradiction:
            contextualized_query += f"\n\n[SYSTEM NOTICE - CONTRADICTION / DISCREPANCY]: {conflict_report.discrepancy_summary} {conflict_report.resolution_advice}"

        final_answer = self.generator.generate(contextualized_query, compressed_chunks)
        self.memory.add_interaction(user_query, final_answer)

        # Extract IDs of the images that actually survived the budget compression to send to the UI
        visual_assets = [c.chunk_id for c in compressed_chunks if c.metadata.content_type == ContentType.IMAGE]

        latency = (time.time() - start_time) * 1000
        retrieved_count = len(unique_chunks)
        final_count = len(compressed_chunks)
        
        telemetry = QueryTelemetry(
            query_id=query_id, query_text=user_query, latency_ms=latency,
            retrieval_rounds=1, retrieved_chunks=retrieved_count,
            final_chunks_used=final_count,
            compression_ratio=final_count / retrieved_count if retrieved_count > 0 else 0,
            metadata={
                "target_document": target_document, 
                "active_documents": active_documents or [],
                "visual_assets": visual_assets
            }
        )
        self.logger.log(telemetry)

        return {"query_id": query_id, "answer": final_answer, "telemetry": telemetry.model_dump()}
