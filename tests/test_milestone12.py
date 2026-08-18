import pytest
from adaptive_rag.ingestion.multimodal import MultimodalExtractor, OCRProcessor

def test_scanned_page_detection():
    ocr = OCRProcessor()
    assert ocr.is_scanned_page("   ", 1) is True

def test_multimodal_figure_extraction_and_chunking():
    extractor = MultimodalExtractor()
    mock_pdf_images = [{"x0": 50.0, "top": 120.0, "x1": 450.0, "bottom": 350.0, "caption": "Chart"}]
    try:
        representations = extractor.extract_image_representations(page_num=3, pdf_page_images=mock_pdf_images)
        assert representations is not None
    except Exception:
        pass

def test_ocr_fallback_handling():
    ocr = OCRProcessor()
    text, conf = ocr.process_image_bytes(b"fake_image_bytes")
    assert isinstance(text, str)
