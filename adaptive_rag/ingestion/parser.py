import hashlib, time, pdfplumber, pytesseract
from pdf2image import convert_from_path
from adaptive_rag.models.schema import Document, ParsedPage, Section

class PageAwarePDFParser:
    def __init__(self, token_counter):
        self.token_counter = token_counter

    def parse_pdf_path(self, file_path: str, document_name: str) -> Document:
        doc_hash = hashlib.sha256(file_path.encode()).hexdigest()
        doc_id = f"doc_{doc_hash[:16]}"
        parsed_pages, total_tokens = [], 0
        
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                raw_tables = page.extract_tables() or []
                
                # FIX: Convert raw lists into structured Pydantic dictionaries
                structured_tables = []
                for t in raw_tables:
                    if not t or len(t) == 0: continue
                    headers = [str(c) if c is not None else "" for c in t[0]]
                    raw_csv = "\n".join([",".join([str(c) if c is not None else "" for c in row]) for row in t])
                    structured_tables.append({"headers": headers, "raw_csv": raw_csv})
                
                # OCR Fallback
                if len(text.strip()) < 50:
                    try:
                        images = convert_from_path(file_path, first_page=page_num, last_page=page_num)
                        if images: text += "\n" + pytesseract.image_to_string(images[0])
                    except Exception: pass
                
                page_tokens = self.token_counter.count_tokens(text)
                total_tokens += page_tokens
                parsed_pages.append(ParsedPage(
                    page_number=page_num, 
                    text=text, 
                    tables=structured_tables, 
                    headings=[], 
                    token_count=page_tokens
                ))

        sections = [Section(
            section_id=f"{doc_id}_s1", 
            title="Main Content", 
            level=1, 
            page_start=1, 
            page_end=total_pages, 
            content="\n\n".join([p.text for p in parsed_pages]), 
            token_count=total_tokens
        )]
        
        return Document(
            document_id=doc_id, 
            document_name=document_name, 
            document_hash=doc_hash, 
            document_type="pdf", 
            total_pages=total_pages, 
            total_tokens=total_tokens, 
            pages=parsed_pages, 
            sections=sections, 
            user_metadata={"parsed_at": time.time()}
        )