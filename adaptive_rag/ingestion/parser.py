import hashlib
import io
import re
import time
from typing import Callable, List, Optional, Tuple, Dict, Any
import pdfplumber
from PIL import Image
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.models.schema import Document, IngestionProgress, IngestionStage, ParsedPage, Section, TableRepresentation, ImageRepresentation
from adaptive_rag.ingestion.multimodal import OCRProcessor, MultimodalExtractor

class PageAwarePDFParser:
    def __init__(self, token_counter: TokenBudgetManager):
        self.token_counter = token_counter
        self.ocr_processor = OCRProcessor()
        self.multimodal_extractor = MultimodalExtractor(self.ocr_processor)

    def compute_sha256(self, file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    def parse_pdf(
        self,
        file_bytes: bytes,
        document_name: str,
        version: str = "1.0",
        document_type: str = "pdf",
        progress_callback: Optional[Callable[[IngestionProgress], None]] = None,
    ) -> Tuple[Document, Dict[str, bytes]]:
        doc_hash = self.compute_sha256(file_bytes)
        doc_id = f"doc_{doc_hash[:16]}_v{version.replace('.', '_')}"

        if progress_callback:
            progress_callback(IngestionProgress(document_id=doc_id, stage=IngestionStage.VALIDATING, message="Validating PDF payload..."))

        parsed_pages: List[ParsedPage] = []
        total_doc_tokens = 0
        extracted_image_bytes_map: Dict[str, bytes] = {}

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                raise ValueError("Document contains 0 pages.")

            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                raw_text = page.extract_text() or ""
                image_count = len(page.images)
                requires_ocr = self.ocr_processor.is_scanned_page(raw_text, image_count)

                if requires_ocr:
                    if progress_callback:
                        progress_callback(IngestionProgress(document_id=doc_id, stage=IngestionStage.OCR_FALLBACK, current_page=page_num, total_pages=total_pages, message=f"Performing OCR on page {page_num}..."))
                    try:
                        from pdf2image import convert_from_bytes
                        pil_images = convert_from_bytes(file_bytes, first_page=page_num, last_page=page_num)
                        if pil_images:
                            img_byte_arr = io.BytesIO()
                            pil_images[0].save(img_byte_arr, format='PNG')
                            ocr_text, _ = self.ocr_processor.process_image_bytes(img_byte_arr.getvalue())
                            raw_text = ocr_text
                    except Exception:
                        pass

                cleaned_text = self._clean_text(raw_text)
                headings = self._extract_headings(cleaned_text)

                structured_tables = []
                extracted_tables = page.extract_tables()
                for table in extracted_tables:
                    if not table or len(table) < 2: continue
                    cleaned_table = [[str(cell).strip() if cell else "" for cell in row] for row in table]
                    csv_repr = "\n".join([",".join(row) for row in cleaned_table])
                    headers = cleaned_table[0]
                    rows = cleaned_table[1:]
                    structured_tables.append(TableRepresentation(
                        headers=headers,
                        rows=rows,
                        raw_csv=csv_repr
                    ))
                    for row in cleaned_table:
                        for cell in row:
                            if cell:
                                cleaned_text = cleaned_text.replace(cell, "")

                image_representations: List[ImageRepresentation] = []
                if page.images:
                    raw_page_images = []
                    for img_meta in page.images:
                        bbox = (
                            float(img_meta.get("x0", 0.0)),
                            float(img_meta.get("top", 0.0)),
                            float(img_meta.get("x1", 100.0)),
                            float(img_meta.get("bottom", 100.0))
                        )
                        img_bytes = b""
                        try:
                            cropped = page.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                            pil_crop = cropped.to_image(resolution=150).original
                            buf = io.BytesIO()
                            pil_crop.save(buf, format="PNG")
                            img_bytes = buf.getvalue()
                        except Exception:
                            dummy = io.BytesIO()
                            Image.new("RGB", (150, 150), color=(230, 230, 230)).save(dummy, format="PNG")
                            img_bytes = dummy.getvalue()

                        raw_page_images.append({
                            "x0": bbox[0], "top": bbox[1], "x1": bbox[2], "bottom": bbox[3],
                            "caption": f"Visual Graphic on Page {page_num}",
                            "bytes": img_bytes
                        })

                    extracted_tuples = self.multimodal_extractor.extract_image_representations(page_num, raw_page_images)
                    fig_chunks, fig_bytes_map = self.multimodal_extractor.create_figure_chunks(
                        doc_id, document_name, version, page_num, extracted_tuples
                    )
                    image_representations.extend([t[0] for t in extracted_tuples])
                    extracted_image_bytes_map.update(fig_bytes_map)

                page_tokens = self.token_counter.count_tokens(cleaned_text)
                total_doc_tokens += page_tokens

                parsed_pages.append(ParsedPage(
                    page_number=page_num,
                    text=cleaned_text.strip(),
                    tables=structured_tables,
                    images=image_representations,
                    headings=headings,
                    token_count=page_tokens,
                    requires_ocr=requires_ocr
                ))

                if progress_callback:
                    progress_callback(IngestionProgress(document_id=doc_id, stage=IngestionStage.PARSING, current_page=page_num, total_pages=total_pages, progress_percentage=round((page_num / total_pages) * 100, 2), message=f"Parsed page {page_num}/{total_pages}"))

        sections = self._build_sections(parsed_pages, doc_id)

        parsed_doc = Document(
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
        return parsed_doc, extracted_image_bytes_map

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
        current_section_title = "Overview"
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
