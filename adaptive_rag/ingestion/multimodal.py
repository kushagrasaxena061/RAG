import io
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
from adaptive_rag.models.schema import ImageRepresentation, Chunk, ChunkMetadata, ChunkType, ContentType

class OCRProcessor:
    """Handles scanned document detection and image OCR with graceful degradation."""
    def __init__(self, min_char_threshold: int = 50):
        self.min_char_threshold = min_char_threshold

    def is_scanned_page(self, text: str, image_count: int) -> bool:
        """Determines if a page is a scanned image lacking selectable text."""
        cleaned = text.strip() if text else ""
        return len(cleaned) < self.min_char_threshold and image_count > 0

    def process_image_bytes(self, img_bytes: bytes) -> Tuple[str, float]:
        """Performs OCR on raw image bytes. Returns (extracted_text, confidence)."""
        try:
            import pytesseract
            image = Image.open(io.BytesIO(img_bytes))
            extracted = pytesseract.image_to_string(image)
            return extracted.strip(), 0.85
        except Exception as e:
            return f"[OCR Fallback Active: Image text extraction unavailable ({str(e)})]", 0.0

class MultimodalExtractor:
    """Extracts, describes, and creates structured representations for figures and charts."""
    def __init__(self, ocr_processor: Optional[OCRProcessor] = None):
        self.ocr = ocr_processor or OCRProcessor()

    def extract_image_representations(self, page_num: int, pdf_page_images: List[Dict[str, Any]]) -> List[ImageRepresentation]:
        """Converts raw PDF image metadata into structured ImageRepresentation schemas."""
        representations = []
        for idx, img_info in enumerate(pdf_page_images):
            bbox = (
                float(img_info.get("x0", 0.0)),
                float(img_info.get("top", 0.0)),
                float(img_info.get("x1", 100.0)),
                float(img_info.get("bottom", 100.0))
            )
            img_hash = hashlib.sha256(f"p{page_num}_img{idx}_{bbox}".encode("utf-8")).hexdigest()[:16]
            caption = img_info.get("caption") or f"Figure {idx + 1} on Page {page_num}"
            
            representations.append(ImageRepresentation(
                image_hash=img_hash,
                bounding_box=bbox,
                caption=caption
            ))
        return representations

    def create_figure_chunks(
        self,
        document_id: str,
        document_name: str,
        version: str,
        page_num: int,
        images: List[ImageRepresentation]
    ) -> List[Chunk]:
        """Constructs token-efficient figure chunks for hybrid retrieval."""
        chunks = []
        for img in images:
            content = f"[FIGURE / CHART] {img.caption} (ID: {img.image_hash}, Region: {img.bounding_box})"
            meta = ChunkMetadata(
                document_id=document_id,
                document_name=document_name,
                version=version,
                page_number=page_num,
                section_title="Figures and Visual Evidence",
                section_path=["Visuals", f"Page {page_num}"],
                content_type=ContentType.IMAGE,
                created_at=0.0
            )
            chunks.append(Chunk(
                chunk_id=f"{document_id}_p{page_num}_fig_{img.image_hash}",
                chunk_type=ChunkType.CHILD,
                content=content,
                structured_data=img,
                token_count=len(content.split()),
                content_hash=img.image_hash,
                metadata=meta
            ))
        return chunks
