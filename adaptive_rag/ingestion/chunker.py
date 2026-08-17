import hashlib
import time
from typing import List, Tuple
from adaptive_rag.config import IngestionConfig
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.models.schema import Chunk, ChunkMetadata, ChunkType, ContentType, Document, Section

class HierarchicalChunker:
    def __init__(self, config: IngestionConfig, token_counter: TokenBudgetManager):
        self.config = config
        self.token_counter = token_counter

    def chunk_document(self, document: Document) -> Tuple[List[Chunk], List[Chunk]]:
        parent_chunks, child_chunks = [], []
        seen_hashes = set()

        for section in document.sections:
            sec_parents, sec_children = self._chunk_section(section, document, seen_hashes)
            parent_chunks.extend(sec_parents)
            child_chunks.extend(sec_children)

        return parent_chunks, child_chunks

    def _chunk_section(self, section: Section, document: Document, seen_hashes: set) -> Tuple[List[Chunk], List[Chunk]]:
        parents, children = [], []
        parent_idx = 1

        # 1. Process Text Paragraphs
        paragraphs = [p.strip() for p in section.content.split("\n\n") if p.strip()]
        current_parent_text = ""

        for para in paragraphs:
            candidate = f"{current_parent_text}\n\n{para}".strip() if current_parent_text else para
            if self.token_counter.count_tokens(candidate) <= self.config.parent_chunk_size_tokens:
                current_parent_text = candidate
            else:
                if current_parent_text:
                    p_chunk, c_chunks = self._build_text_chunks(current_parent_text, parent_idx, section, document, seen_hashes)
                    if p_chunk:
                        parents.append(p_chunk)
                        children.extend(c_chunks)
                        parent_idx += 1
                current_parent_text = para

        if current_parent_text:
            p_chunk, c_chunks = self._build_text_chunks(current_parent_text, parent_idx, section, document, seen_hashes)
            if p_chunk:
                parents.append(p_chunk)
                children.extend(c_chunks)
                parent_idx += 1

        # 2. Process Tables (Tables bypass word-splitting, kept intact for Token Efficiency)
        for table in section.tables:
            table_id = f"{section.section_id}_p{parent_idx}_table"
            table_content = f"Table Headers: {', '.join(table.headers)}\nData:\n{table.raw_csv}"
            t_tokens = self.token_counter.count_tokens(table_content)
            t_hash = hashlib.sha256(table_content.encode("utf-8")).hexdigest()

            metadata = ChunkMetadata(
                document_id=document.document_id,
                document_name=document.document_name,
                version=document.version,
                page_number=section.page_start,
                section_title=section.title,
                section_path=[section.title],
                content_type=ContentType.TABLE,
                created_at=time.time()
            )

            # Table is its own parent and child simultaneously to preserve structure
            table_chunk = Chunk(
                chunk_id=table_id,
                chunk_type=ChunkType.CHILD,
                content=table_content,
                structured_data=table,
                token_count=t_tokens,
                content_hash=t_hash,
                metadata=metadata
            )
            children.append(table_chunk)
            parent_idx += 1

        return parents, children

    def _build_text_chunks(self, parent_text: str, parent_idx: int, section: Section, document: Document, seen_hashes: set) -> Tuple[Chunk, List[Chunk]]:
        parent_tokens = self.token_counter.count_tokens(parent_text)
        if parent_tokens < self.config.min_chunk_size_tokens:
            return None, []

        p_hash = hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
        parent_id = f"{section.section_id}_p{parent_idx}"

        metadata = ChunkMetadata(
            document_id=document.document_id,
            document_name=document.document_name,
            version=document.version,
            page_number=section.page_start,
            section_title=section.title,
            section_path=[section.title],
            content_type=ContentType.TEXT,
            created_at=time.time(),
        )

        parent_chunk = Chunk(
            chunk_id=parent_id,
            chunk_type=ChunkType.PARENT,
            content=parent_text,
            token_count=parent_tokens,
            content_hash=p_hash,
            metadata=metadata,
        )

        child_chunks = []
        words = parent_text.split()
        child_word_target = max(10, int(self.config.child_chunk_size_tokens * 0.75))
        child_word_overlap = max(2, int(self.config.chunk_overlap_tokens * 0.75))

        step = max(1, child_word_target - child_word_overlap)
        for child_idx, i in enumerate(range(0, len(words), step), start=1):
            window = words[i : i + child_word_target]
            if not window: continue

            child_text = " ".join(window)
            c_tokens = self.token_counter.count_tokens(child_text)
            if c_tokens < self.config.min_chunk_size_tokens: continue

            c_hash = hashlib.sha256(child_text.encode("utf-8")).hexdigest()
            if self.config.enable_deduplication and c_hash in seen_hashes: continue
            seen_hashes.add(c_hash)

            child_chunks.append(Chunk(
                chunk_id=f"{parent_id}_c{child_idx}",
                parent_chunk_id=parent_id,
                chunk_type=ChunkType.CHILD,
                content=child_text,
                token_count=c_tokens,
                content_hash=c_hash,
                metadata=metadata.model_copy(),
            ))

        return parent_chunk, child_chunks
