import hashlib
from adaptive_rag.config import global_config
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.ingestion.chunker import HierarchicalChunker
from adaptive_rag.ingestion.parser import PageAwarePDFParser
from adaptive_rag.storage.sqlite_db import SQLiteManager
from adaptive_rag.security.sanitizer import SecuritySanitizer

class IngestionPipeline:
    def __init__(self):
        self.config = global_config
        self.token_counter = TokenBudgetManager(self.config.model)
        self.parser = PageAwarePDFParser(self.token_counter)
        self.chunker = HierarchicalChunker(self.config.ingestion, self.token_counter)
        self.db = SQLiteManager()

    def ingest_pdf_path(self, file_path: str, document_name: str, version: str = "1.0"):
        parsed_doc = self.parser.parse_pdf_path(file_path, document_name)

        # INCREMENTAL DIFFING & INDIRECT INJECTION DEFENSE
        modified_sections = []
        for section in parsed_doc.sections:
            section.content = SecuritySanitizer.sanitize_text(section.content)
            sec_hash = hashlib.sha256(section.content.encode('utf-8')).hexdigest()
            old_hash = self.db.get_page_hash(document_name, section.page_start)

            if old_hash != sec_hash:
                modified_sections.append(section)
                self.db.save_page_hash(document_name, section.page_start, sec_hash)

        if not modified_sections:
            return parsed_doc, [], [] # Skip processing if document is identical

        parsed_doc.sections = modified_sections
        parents, children = self.chunker.chunk_document(parsed_doc)
        return parsed_doc, parents, children