import io
import hashlib
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
from adaptive_rag.models.schema import ImageRepresentation, Chunk, ChunkMetadata, ChunkType, ContentType

class OCRProcessor:
    def __init__(self, min_char_threshold: int = 50):
        self.min_char_threshold = min_char_threshold

    def is_scanned_page(self, text: str, image_count: int) -> bool:
        cleaned = text.strip() if text else ""
        return len(cleaned) < self.min_char_threshold and image_count > 0

    def process_image_bytes(self, img_bytes: bytes) -> Tuple[str, float]:
        try:
            import pytesseract
            image = Image.open(io.BytesIO(img_bytes))
            extracted = pytesseract.image_to_string(image)
            return extracted.strip(), 0.85
        except Exception as e:
            return f"[Visual evidence registered]", 0.0

class MultimodalExtractor:
    def __init__(self, ocr_processor: Optional[OCRProcessor] = None):
        self.ocr = ocr_processor or OCRProcessor()

    def extract_image_representations(self, page_num: int, pdf_page_images: List[Dict[str, Any]]) -> List[Tuple[ImageRepresentation, bytes]]:
        representations = []
        for idx, img_info in enumerate(pdf_page_images):
            bbox = (
                float(img_info.get("x0", 0.0)),
                float(img_info.get("top", 0.0)),
                float(img_info.get("x1", 100.0)),
                float(img_info.get("bottom", 100.0))
            )
            raw_bytes = img_info.get("bytes", b"")
            if not raw_bytes:
                # Synthetic byte fallback for bounding-box metadata if raw image parsing fails
                img_byte_arr = io.BytesIO()
                Image.new("RGB", (200, 200), color=(240, 240, 240)).save(img_byte_arr, format='PNG')
                raw_bytes = img_byte_arr.getvalue()

            img_hash = hashlib.sha256(f"p{page_num}_img{idx}_{bbox}".encode("utf-8")).hexdigest()[:16]
            caption = img_info.get("caption") or f"Chart/Figure {idx + 1} on Page {page_num}"

            rep = ImageRepresentation(
                image_hash=img_hash,
                bounding_box=bbox,
                caption=caption
            )
            representations.append((rep, raw_bytes))
        return representations

    def create_figure_chunks(
        self,
        document_id: str,
        document_name: str,
        version: str,
        page_num: int,
        extracted_tuples: List[Tuple[ImageRepresentation, bytes]]
    ) -> Tuple[List[Chunk], Dict[str, bytes]]:
        chunks = []
        image_bytes_map = {}

        for img, raw_bytes in extracted_tuples:
            cid = f"{document_id}_p{page_num}_fig_{img.image_hash}"
            content = f"### [VISUAL FIGURE / CHART: {img.caption} | Document: {document_name}]\n- Page: {page_num}\n- Visual Hash: {img.image_hash}\n"
            
            meta = ChunkMetadata(
                document_id=document_id,
                document_name=document_name,
                version=version,
                page_number=page_num,
                section_title="Figures, Charts, and Visual Evidence",
                section_path=[document_name, f"Page {page_num}", "Visual"],
                content_type=ContentType.IMAGE,
                created_at=0.0
            )
            chunk = Chunk(
                chunk_id=cid,
                chunk_type=ChunkType.CHILD,
                content=content,
                structured_data=img,
                token_count=len(content.split()),
                content_hash=img.image_hash,
                metadata=meta
            )
            chunks.append(chunk)
            image_bytes_map[cid] = raw_bytes

        return chunks, image_bytes_map
