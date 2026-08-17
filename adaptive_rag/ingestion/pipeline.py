from typing import Callable, Dict, List, Optional, Tuple
from adaptive_rag.config import AppConfig, global_config
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.ingestion.parser import PageAwarePDFParser
from adaptive_rag.ingestion.diff_engine import IncrementalDiffEngine
from adaptive_rag.ingestion.multimodal import OCRProcessor, MultimodalExtractor
from adaptive_rag.models.schema import Chunk, Document, IngestionProgress, IngestionStage

class IngestionPipeline:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or global_config
        self.token_counter = TokenBudgetManager(self.config.model)
        self.parser = PageAwarePDFParser(self.token_counter)
        self.diff_engine = IncrementalDiffEngine(self.config.ingestion, self.token_counter)
        self.indexed_documents: Dict[str, Document] = {}

    def ingest_pdf_bytes(
        self,
        file_bytes: bytes,
        document_name: str,
        version: str = "1.0",
        document_type: str = "pdf",
        progress_callback: Optional[Callable[[IngestionProgress], None]] = None,
    ) -> Tuple[Document, List[Chunk], List[Chunk], Dict[str, bytes]]:
        
        parsed_doc, image_bytes_map = self.parser.parse_pdf(
            file_bytes=file_bytes,
            document_name=document_name,
            version=version,
            document_type=document_type,
            progress_callback=progress_callback,
        )

        parents, children, stats = self.diff_engine.diff_and_chunk(parsed_doc)
        doc_hash = parsed_doc.document_hash
        self.indexed_documents[f"{doc_hash}_v{version}"] = parsed_doc

        if progress_callback:
            msg = f"Indexed {stats['modified_pages']} pages ({len(parents) + len(children)} text/table chunks, {len(image_bytes_map)} visual figures)."
            progress_callback(IngestionProgress(
                document_id=parsed_doc.document_id,
                stage=IngestionStage.COMPLETED,
                total_pages=parsed_doc.total_pages,
                chunks_created=len(parents) + len(children),
                progress_percentage=100.0,
                message=msg
            ))

        return parsed_doc, parents, children, image_bytes_map
