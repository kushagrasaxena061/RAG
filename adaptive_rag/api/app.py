from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import StreamingResponse
from typing import List, Generator, Optional
from pydantic import BaseModel
import os
import json
import shutil
import time
import uuid
import traceback

from adaptive_rag.pipeline.orchestrator import RAGPipelineOrchestrator
from adaptive_rag.observability.tracker import TelemetryLogger, QueryTelemetry
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.retrieval.vector_index import VectorIndex
from adaptive_rag.retrieval.bm25_index import BM25Index
from adaptive_rag.retrieval.hybrid_search import HybridRetriever
from adaptive_rag.retrieval.multimodal_index import MultimodalVisualIndex
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.memory.manager import TwoTierMemory
from adaptive_rag.config import global_config
from adaptive_rag.models.schema import ContentType, Chunk

app = FastAPI(title="Adaptive Token-Efficient RAG API")

def initialize_indices():
    global vec_idx, bm25_idx, visual_idx, hybrid_retriever, memory_manager, orchestrator, ingestion_pipeline, token_manager, logger
    os.makedirs("./data", exist_ok=True)
    vec_idx = VectorIndex(persist_directory="./data/chroma")
    bm25_idx = BM25Index(persist_path="./data/bm25.pkl")
    visual_idx = MultimodalVisualIndex(persist_directory="./data/chroma_multimodal")
    
    hybrid_retriever = HybridRetriever(vec_idx, bm25_idx)
    token_manager = TokenBudgetManager(global_config.model)
    logger = TelemetryLogger()

    mem_vec_idx = VectorIndex(persist_directory="./data/chroma_memory", collection_name="memory")
    memory_manager = TwoTierMemory(token_manager, mem_vec_idx)

    orchestrator = RAGPipelineOrchestrator(hybrid_retriever, visual_idx, token_manager, memory_manager, logger)
    ingestion_pipeline = IngestionPipeline()

initialize_indices()

class QueryRequest(BaseModel):
    query: str
    target_document: str = "All Documents"
    active_documents: List[str] = []
    model_name: Optional[str] = None
    temperature: float = 0.2
    mock_mode: bool = False

@app.post("/reset")
def reset_database():
    try:
        shutil.rmtree("./data", ignore_errors=True)
        initialize_indices()
        return {"status": "ok", "message": "Database and memory reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            content = await file.read()
            doc, parents, children, image_bytes_map = ingestion_pipeline.ingest_pdf_bytes(content, file.filename)
            
            all_chunks = parents + children
            if all_chunks:
                unique_chunks = list({c.chunk_id: c for c in all_chunks}.values())
                vec_idx.add_chunks(unique_chunks)
                bm25_idx.add_chunks(unique_chunks)
                
                fig_chunks = [c for c in unique_chunks if c.metadata.content_type == ContentType.IMAGE]
                if fig_chunks and image_bytes_map:
                    visual_idx.add_figure_chunks(fig_chunks, image_bytes_map)
                
            results.append({
                "filename": file.filename,
                "chunks_indexed": len(all_chunks),
                "figures_indexed": len(image_bytes_map)
            })
        return {"uploaded": results}
    except Exception as e:
        print("--- UPLOAD ERROR ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-stream")
def ask_question_stream(request: QueryRequest):
    def stream_generator() -> Generator[str, None, None]:
        start_time = time.time()
        query_id = f"q_{uuid.uuid4().hex[:8]}"
        user_query = request.query
        target_document = request.target_document
        active_documents = request.active_documents

        if orchestrator._is_conversational(user_query):
            ans = "Hello! I am your Adaptive Token-Efficient AI Assistant. Ask questions across your documents, tables, or charts."
            orchestrator.memory.add_interaction(user_query, ans)
            payload = {
                "type": "metadata",
                "telemetry": {"latency_ms": round((time.time() - start_time) * 1000, 2), "visual_assets": []},
                "complete_text": ans
            }
            yield json.dumps(payload) + "\n---METADATA_END---\n"
            yield ans
            return

        chat_history = orchestrator.memory.get_short_term_context(max_budget_tokens=400)
        contextualized_query = f"Chat History:\n{chat_history}\n\nCurrent Query: {user_query}" if chat_history else user_query

        raw_chunks: List[Chunk] = []
        if target_document == "All Documents" and active_documents and len(active_documents) > 1:
            per_doc_k = max(3, 12 // len(active_documents))
            for doc_name in active_documents:
                doc_query = f"Document {doc_name} overview table summary {user_query}"
                doc_hits = orchestrator.retriever.search(doc_query, top_k=per_doc_k)
                matched = [c for c in doc_hits if c.metadata.document_name == doc_name]
                if not matched:
                    matched = [c for c in orchestrator.retriever.search(user_query, top_k=per_doc_k) if c.metadata.document_name == doc_name]
                raw_chunks.extend(matched)
        elif target_document != "All Documents":
            doc_hits = orchestrator.retriever.search(f"{target_document} {user_query}", top_k=15)
            raw_chunks = [c for c in doc_hits if c.metadata.document_name == target_document]
            if len(raw_chunks) < 3:
                raw_chunks.extend([c for c in orchestrator.retriever.search(user_query, top_k=10) if c.metadata.document_name == target_document])
        else:
            raw_chunks = orchestrator.retriever.search(user_query, top_k=15)

        if orchestrator._is_visual_query(user_query) or len(raw_chunks) < 4:
            visual_hits = orchestrator.visual_index.search_visual(user_query, top_k=3, target_document=target_document)
            for v in visual_hits:
                meta = Chunk(
                    chunk_id=v["chunk_id"],
                    chunk_type=ContentType.IMAGE,
                    content=v["content"],
                    token_count=len(v["content"].split()),
                    content_hash=v["chunk_id"],
                    metadata={
                        "document_name": v["metadata"].get("document_name", "document.pdf"),
                        "page_number": v["metadata"].get("page_number", 1),
                        "content_type": ContentType.IMAGE.value,
                        "section_title": "Visual Evidence",
                        "created_at": 0.0,
                        "document_id": "vis",
                        "version": "1.0"
                    }
                )
                raw_chunks.append(meta)

        unique_chunks = list({c.chunk_id: c for c in raw_chunks}.values())
        reranked = orchestrator.reranker.rerank(user_query, unique_chunks, top_k=8)
        ordered_chunks = orchestrator.temporal_engine.sort_chunks_by_temporal_order(reranked)

        history_tokens = orchestrator.token_manager.count_tokens(chat_history) if chat_history else 0
        budget = orchestrator.token_manager.calculate_budget(query=user_query, conversation_history_tokens=history_tokens)
        compressed_chunks = orchestrator.compressor.compress_context(ordered_chunks, budget)

        visual_assets = [c.chunk_id for c in compressed_chunks if c.metadata.content_type == ContentType.IMAGE]

        telemetry_info = {
            "query_id": query_id,
            "retrieved_chunks": len(unique_chunks),
            "final_chunks_used": len(compressed_chunks),
            "compression_ratio": round(len(compressed_chunks) / len(unique_chunks), 2) if unique_chunks else 0.0,
            "visual_assets": visual_assets,
            "model_used": request.model_name or "default"
        }

        yield json.dumps({"type": "metadata", "telemetry": telemetry_info}) + "\n---METADATA_END---\n"

        full_answer_accumulator = []
        for token in orchestrator.generator.generate_stream(
            contextualized_query, 
            compressed_chunks,
            model_name=request.model_name,
            temperature=request.temperature
        ):
            full_answer_accumulator.append(token)
            yield token

        full_answer = "".join(full_answer_accumulator)
        orchestrator.memory.add_interaction(user_query, full_answer)

    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.get("/image/{chunk_id}")
def get_image(chunk_id: str):
    img_bytes = visual_idx.image_store.get(chunk_id)
    if img_bytes:
        return Response(content=img_bytes, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")
