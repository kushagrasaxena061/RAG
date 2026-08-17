from fastapi import FastAPI, UploadFile, File
from typing import List
from pydantic import BaseModel
import os

from adaptive_rag.pipeline.orchestrator import RAGPipelineOrchestrator
from adaptive_rag.observability.tracker import TelemetryLogger
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.retrieval.vector_index import VectorIndex
from adaptive_rag.retrieval.bm25_index import BM25Index
from adaptive_rag.retrieval.hybrid_search import HybridRetriever
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.config import global_config

app = FastAPI(title="Adaptive Token-Efficient RAG API")

os.makedirs("./data", exist_ok=True)
vec_idx = VectorIndex(persist_directory="./data/chroma")
bm25_idx = BM25Index(persist_path="./data/bm25.pkl")
hybrid_retriever = HybridRetriever(vec_idx, bm25_idx)

token_manager = TokenBudgetManager(global_config.model)
logger = TelemetryLogger()
orchestrator = RAGPipelineOrchestrator(hybrid_retriever, token_manager, logger)
ingestion_pipeline = IngestionPipeline()

class QueryRequest(BaseModel):
    query: str
    mock_mode: bool = False

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        doc, parents, children = ingestion_pipeline.ingest_pdf_bytes(content, file.filename)
        
        all_chunks = parents + children
        if all_chunks:
            vec_idx.add_chunks(all_chunks)
            bm25_idx.add_chunks(all_chunks)
            
        results.append({"filename": file.filename, "chunks_indexed": len(all_chunks)})
    return {"uploaded": results}

@app.post("/ask")
def ask_question(request: QueryRequest):
    return orchestrator.process_query(request.query, mock_mode=request.mock_mode)
