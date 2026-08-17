from typing import Callable, Dict, List, Optional, Tuple
from adaptive_rag.config import AppConfig, global_config
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.ingestion.chunker import HierarchicalChunker
from adaptive_rag.ingestion.parser import PageAwarePDFParser
from adaptive_rag.models.schema import Chunk, Document, IngestionProgress, IngestionStage

class IngestionPipeline:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or global_config
        self.token_counter = TokenBudgetManager(self.config.model)
        self.parser = PageAwarePDFParser(self.token_counter)
        self.chunker = HierarchicalChunker(self.config.ingestion, self.token_counter)
        self.indexed_documents: Dict[str, Document] = {}

    def ingest_pdf_bytes(
        self,
        file_bytes: bytes,
        document_name: str,
        version: str = "1.0",
        document_type: str = "pdf",
        progress_callback: Optional[Callable[[IngestionProgress], None]] = None,
    ) -> Tuple[Document, List[Chunk], List[Chunk]]:
        doc_hash = self.parser.compute_sha256(file_bytes)
        cache_key = f"{doc_hash}_v{version}"

        if cache_key in self.indexed_documents:
            cached_doc = self.indexed_documents[cache_key]
            if progress_callback:
                progress_callback(IngestionProgress(document_id=cached_doc.document_id, stage=IngestionStage.COMPLETED, progress_percentage=100.0, message="Version already indexed. Retrieved from cache."))
            parents, children = self.chunker.chunk_document(cached_doc)
            return cached_doc, parents, children

        parsed_doc = self.parser.parse_pdf(
            file_bytes=file_bytes,
            document_name=document_name,
            version=version,
            document_type=document_type,
            progress_callback=progress_callback,
        )

        parents, children = self.chunker.chunk_document(parsed_doc)
        self.indexed_documents[cache_key] = parsed_doc

        if progress_callback:
            progress_callback(IngestionProgress(document_id=parsed_doc.document_id, stage=IngestionStage.COMPLETED, total_pages=parsed_doc.total_pages, chunks_created=len(parents) + len(children), progress_percentage=100.0, message="Indexing complete."))

        return parsed_doc, parents, children
