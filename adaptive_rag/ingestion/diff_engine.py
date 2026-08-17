import hashlib
from typing import List, Dict, Tuple, Set
from adaptive_rag.models.schema import Document, Chunk, ParsedPage
from adaptive_rag.ingestion.chunker import HierarchicalChunker
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.config import IngestionConfig

class IncrementalDiffEngine:
    def __init__(self, config: IngestionConfig, token_counter: TokenBudgetManager):
        self.chunker = HierarchicalChunker(config, token_counter)
        self.page_hashes_by_doc: Dict[str, Dict[int, str]] = {}
        self.cached_chunks_by_page: Dict[str, Dict[int, Tuple[List[Chunk], List[Chunk]]]] = {}

    def compute_page_hash(self, page: ParsedPage) -> str:
        content = f"{page.text}:{page.token_count}:{[t.raw_csv for t in page.tables]}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def diff_and_chunk(
        self,
        new_doc: Document
    ) -> Tuple[List[Chunk], List[Chunk], Dict[str, int]]:
        doc_name = new_doc.document_name
        prev_hashes = self.page_hashes_by_doc.get(doc_name, {})
        prev_chunks = self.cached_chunks_by_page.get(doc_name, {})

        # Use dictionaries to mathematically guarantee zero duplicate IDs across pages
        unique_parents: Dict[str, Chunk] = {}
        unique_children: Dict[str, Chunk] = {}
        
        reused_pages = 0
        modified_pages = 0

        new_hashes: Dict[int, str] = {}
        new_chunks: Dict[int, Tuple[List[Chunk], List[Chunk]]] = {}

        for page in new_doc.pages:
            p_num = page.page_number
            p_hash = self.compute_page_hash(page)
            new_hashes[p_num] = p_hash

            if p_num in prev_hashes and prev_hashes[p_num] == p_hash and p_num in prev_chunks:
                cached_p, cached_c = prev_chunks[p_num]
                for p in cached_p: unique_parents[p.chunk_id] = p
                for c in cached_c: unique_children[c.chunk_id] = c
                new_chunks[p_num] = (cached_p, cached_c)
                reused_pages += 1
            else:
                single_page_doc = new_doc.model_copy(update={
                    "pages": [page],
                    "sections": [s for s in new_doc.sections if s.page_start <= p_num <= s.page_end]
                })
                p_chunks, c_chunks = self.chunker.chunk_document(single_page_doc)
                
                # Append page number to chunk IDs to guarantee global uniqueness
                for p in p_chunks:
                    p.chunk_id = f"{p.chunk_id}_pg{p_num}"
                    unique_parents[p.chunk_id] = p
                for c in c_chunks:
                    c.chunk_id = f"{c.chunk_id}_pg{p_num}"
                    if c.parent_chunk_id:
                        c.parent_chunk_id = f"{c.parent_chunk_id}_pg{p_num}"
                    unique_children[c.chunk_id] = c
                
                new_chunks[p_num] = (p_chunks, c_chunks)
                modified_pages += 1

        self.page_hashes_by_doc[doc_name] = new_hashes
        self.cached_chunks_by_page[doc_name] = new_chunks

        stats = {
            "total_pages": len(new_doc.pages),
            "reused_pages": reused_pages,
            "modified_pages": modified_pages
        }
        return list(unique_parents.values()), list(unique_children.values()), stats
