import pytest
from adaptive_rag.ingestion.multimodal import OCRProcessor, MultimodalExtractor
from adaptive_rag.models.schema import ContentType, ChunkType

def test_scanned_page_detection():
    """Verify that pages with insufficient selectable text and images trigger OCR classification."""
    ocr_proc = OCRProcessor(min_char_threshold=50)

    # Scanned page scenario: empty/near-empty text but embedded image present
    assert ocr_proc.is_scanned_page(text="   \n ", image_count=1) is True
    assert ocr_proc.is_scanned_page(text="Small text", image_count=2) is True

    # Standard selectable text page
    assert ocr_proc.is_scanned_page(text="This page contains plenty of selectable text describing financials." * 5, image_count=1) is False
    assert ocr_proc.is_scanned_page(text="", image_count=0) is False

def test_multimodal_figure_extraction_and_chunking():
    """Verify structured representation and chunk generation for figures and charts."""
    extractor = MultimodalExtractor()
    
    mock_pdf_images = [
        {"x0": 50.0, "top": 120.0, "x1": 450.0, "bottom": 350.0, "caption": "Quarterly Revenue Breakdown 2024 Chart"},
        {"x0": 50.0, "top": 400.0, "x1": 300.0, "bottom": 600.0}
    ]
    
    representations = extractor.extract_image_representations(page_num=3, pdf_page_images=mock_pdf_images)
    assert len(representations) == 2
    assert representations[0].caption == "Quarterly Revenue Breakdown 2024 Chart"
    assert len(representations[0].bounding_box) == 4

    chunks = extractor.create_figure_chunks(
        document_id="doc_vis",
        document_name="visual_report.pdf",
        version="1.0",
        page_num=3,
        images=representations
    )
    
    assert len(chunks) == 2
    assert chunks[0].metadata.content_type == ContentType.IMAGE
    assert chunks[0].chunk_type == ChunkType.CHILD
    assert "Quarterly Revenue Breakdown" in chunks[0].content

def test_ocr_fallback_handling():
    """Verify that OCR execution gracefully recovers without crashing if tesseract binary is absent."""
    ocr_proc = OCRProcessor()
    # Passing empty dummy bytes to verify error trapping and structured fallback
    text, confidence = ocr_proc.process_image_bytes(b"dummy_bytes")
    assert isinstance(text, str)
    assert confidence == 0.0
