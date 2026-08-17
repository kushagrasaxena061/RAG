import hashlib
import io
import re
import time
from typing import Callable, List, Optional
import pdfplumber
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.models.schema import Document, IngestionProgress, IngestionStage, ParsedPage, Section, TableRepresentation

class PageAwarePDFParser:
    def __init__(self, token_counter: TokenBudgetManager):
        self.token_counter = token_counter

    def compute_sha256(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def _trigger_ocr(self, page_image) -> str:
        """Fallback OCR logic utilizing pytesseract."""
        try:
            import pytesseract
            return pytesseract.image_to_string(page_image)
        except Exception as e:
            return f"[OCR FAILED: {str(e)}]"

    def parse_pdf(
        self,
        file_bytes: bytes,
        document_name: str,
        version: str = "1.0",
        document_type: str = "pdf",
        progress_callback: Optional[Callable[[IngestionProgress], None]] = None,
    ) -> Document:
        doc_hash = self.compute_sha256(file_bytes)
        doc_id = f"doc_{doc_hash[:16]}_v{version.replace('.', '_')}"

        if progress_callback:
            progress_callback(IngestionProgress(document_id=doc_id, stage=IngestionStage.VALIDATING, message="Validating payload..."))

        parsed_pages: List[ParsedPage] = []
        total_doc_tokens = 0

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                raise ValueError("Document contains 0 pages.")

            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                
                # 1. Extract Text
                raw_text = page.extract_text() or ""
                requires_ocr = len(raw_text.strip()) < 50 and len(page.images) > 0
                
                if requires_ocr:
                    if progress_callback:
                        progress_callback(IngestionProgress(document_id=doc_id, stage=IngestionStage.OCR_FALLBACK, current_page=page_num, total_pages=total_pages, message="Triggering OCR..."))
                    try:
                        # Convert specific page to image for OCR
                        from pdf2image import convert_from_bytes
                        images = convert_from_bytes(file_bytes, first_page=page_num, last_page=page_num)
                        if images:
                            raw_text = self._trigger_ocr(images[0])
                    except Exception:
                        pass # Silently continue if poppler/tesseract not installed in test env
                
                cleaned_text = self._clean_text(raw_text)
                headings = self._extract_headings(cleaned_text)

                # 2. Extract Tables (Preserving Structure)
                structured_tables = []
                extracted_tables = page.extract_tables()
                for table in extracted_tables:
                    if not table or len(table) < 2: continue
                    cleaned_table = [[str(cell).strip() if cell else "" for cell in row] for row in table]
                    csv_repr = "\n".join([",".join(row) for row in cleaned_table])
                    structured_tables.append(TableRepresentation(
                        headers=cleaned_table[0],
                        rows=cleaned_table[1:],
                        raw_csv=csv_repr
                    ))
                    # Prevent table text from duplicating in the standard text chunk
                    for row in cleaned_table:
                        for cell in row:
                            cleaned_text = cleaned_text.replace(cell, "")

                page_tokens = self.token_counter.count_tokens(cleaned_text)
                total_doc_tokens += page_tokens

                parsed_pages.append(ParsedPage(
                    page_number=page_num,
                    text=cleaned_text.strip(),
                    tables=structured_tables,
                    headings=headings,
                    token_count=page_tokens,
                    requires_ocr=requires_ocr
                ))

                if progress_callback:
                    progress_callback(IngestionProgress(document_id=doc_id, stage=IngestionStage.PARSING, current_page=page_num, total_pages=total_pages, progress_percentage=round((page_num / total_pages) * 100, 2), message=f"Parsed page {page_num}/{total_pages}"))

        sections = self._build_sections(parsed_pages, doc_id)

        return Document(
            document_id=doc_id,
            document_name=document_name,
            version=version,
            document_hash=doc_hash,
            document_type=document_type,
            total_pages=total_pages,
            total_tokens=total_doc_tokens,
            pages=parsed_pages,
            sections=sections,
            user_metadata={"parsed_at": time.time()},
        )

    def _clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_headings(self, text: str) -> List[str]:
        lines = text.split("\n")
        heading_pattern = re.compile(r"^(\d+(\.\d+)*|[A-Z][A-Z\s]{2,40}|[A-Z][a-z0-9\s]{2,40}:)$")
        return [line.strip() for line in lines if 3 < len(line.strip()) < 60 and heading_pattern.match(line.strip())]

    def _build_sections(self, pages: List[ParsedPage], doc_id: str) -> List[Section]:
        sections, current_section_pages, current_tables = [], [], []
        current_section_title = "Introduction"
        section_idx = 1

        for page in pages:
            if page.headings:
                if current_section_pages:
                    sec_text = "\n\n".join([p.text for p in current_section_pages])
                    sections.append(Section(
                        section_id=f"{doc_id}_sec_{section_idx}",
                        title=current_section_title,
                        level=1,
                        page_start=current_section_pages[0].page_number,
                        page_end=current_section_pages[-1].page_number,
                        content=sec_text,
                        tables=current_tables,
                        token_count=self.token_counter.count_tokens(sec_text),
                    ))
                    section_idx += 1
                    current_section_pages, current_tables = [], []
                current_section_title = page.headings[0]
            current_section_pages.append(page)
            current_tables.extend(page.tables)

        if current_section_pages:
            sec_text = "\n\n".join([p.text for p in current_section_pages])
            sections.append(Section(
                section_id=f"{doc_id}_sec_{section_idx}",
                title=current_section_title,
                level=1,
                page_start=current_section_pages[0].page_number,
                page_end=current_section_pages[-1].page_number,
                content=sec_text,
                tables=current_tables,
                token_count=self.token_counter.count_tokens(sec_text),
            ))
        return sections
