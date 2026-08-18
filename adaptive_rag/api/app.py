from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import StreamingResponse
import shutil, tempfile, os, json, time
from typing import List, Generator
from pydantic import BaseModel
from adaptive_rag.pipeline.orchestrator import RAGPipelineOrchestrator
from adaptive_rag.observability.tracker import TelemetryLogger
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.retrieval.vector_index import VectorIndex
from adaptive_rag.retrieval.bm25_index import BM25Index
from adaptive_rag.retrieval.hybrid_search import HybridRetriever
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.memory.manager import TwoTierMemory
from adaptive_rag.config import global_config
from adaptive_rag.storage.sqlite_db import SQLiteManager

app = FastAPI(title="Production RAG API")

os.makedirs("./data", exist_ok=True)
db = SQLiteManager("./data/rag_state.db")
vec_idx = VectorIndex(persist_directory="./data/chroma")
bm25_idx = BM25Index(persist_path="./data/bm25.pkl")

hybrid_retriever = HybridRetriever(vec_idx, bm25_idx)
token_manager = TokenBudgetManager(global_config.model)
logger = TelemetryLogger()

mem_vec_idx = VectorIndex(persist_directory="./data/chroma_memory", collection_name="memory")
memory_manager = TwoTierMemory(token_manager, mem_vec_idx)

orchestrator = RAGPipelineOrchestrator(hybrid_retriever, token_manager, memory_manager, db, logger)
ingestion_pipeline = IngestionPipeline()

class QueryRequest(BaseModel):
    query: str
    target_document: str = "All Documents"
    active_documents: List[str] = []
    mock_mode: bool = False

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        try:
            doc, parents, children = ingestion_pipeline.ingest_pdf_path(tmp_path, file.filename)
            all_chunks = parents + children
            if all_chunks:
                unique_chunks = list({c.chunk_id: c for c in all_chunks}.values())
                vec_idx.add_chunks(unique_chunks)
                bm25_idx.add_chunks(unique_chunks)
            db.save_document(getattr(doc, 'document_hash', f"hash_{file.filename}"), {"filename": file.filename, "chunks": len(all_chunks)})
            results.append({"filename": file.filename, "chunks_indexed": len(all_chunks), "status": "Streaming ingestion complete"})
        except Exception as e:
            db.log_crash(e)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return {"uploaded": results}

@app.post("/ask")
def ask_question(request: QueryRequest):
    return orchestrator.process_query(request.query, request.target_document, request.active_documents, request.mock_mode)

@app.post("/ask-stream")
def ask_question_stream(request: QueryRequest):
    # TRUE SPECULATIVE STREAMING ENDPOINT
    return StreamingResponse(
        orchestrator.process_query_stream(
            request.query, 
            request.target_document, 
            request.active_documents, 
            request.mock_mode
        ), 
        media_type="text/plain"
    )

@app.post("/reset")
def reset_system():
    db.conn.execute("DELETE FROM documents")
    db.conn.execute("DELETE FROM cache")
    db.conn.commit()
    return {"status": "ok"}