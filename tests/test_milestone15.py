import pytest
import io
from PIL import Image
from adaptive_rag.retrieval.multimodal_index import MultimodalVisualIndex
from adaptive_rag.ingestion.multimodal import MultimodalExtractor
from adaptive_rag.models.schema import ContentType

def test_clip_multimodal_embedding_and_crossmodal_search(tmp_path):
    """Verify that CLIP cross-modal search can retrieve image chunks via natural language text."""
    persist_dir = str(tmp_path / "chroma_vis")
    visual_index = MultimodalVisualIndex(persist_directory=persist_dir)
    extractor = MultimodalExtractor()

    # Create a synthetic chart image
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (200, 200), color=(10, 100, 200)).save(img_byte_arr, format='PNG')
    chart_bytes = img_byte_arr.getvalue()

    mock_images = [
        {"x0": 20.0, "top": 50.0, "x1": 300.0, "bottom": 250.0, "caption": "Quarterly Revenue Growth Bar Chart", "bytes": chart_bytes}
    ]

    extracted_tuples = extractor.extract_image_representations(page_num=2, pdf_page_images=mock_images)
    chunks, img_bytes_map = extractor.create_figure_chunks(
        document_id="doc_test",
        document_name="annual_report.pdf",
        version="1.0",
        page_num=2,
        extracted_tuples=extracted_tuples
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.content_type == ContentType.IMAGE

    # Add to Visual Vector Index
    visual_index.add_figure_chunks(chunks, img_bytes_map)

    # Search using text query
    results = visual_index.search_visual("Show me the revenue growth bar chart", top_k=1)
    assert len(results) == 1
    assert "Quarterly Revenue Growth Bar Chart" in results[0]["content"]
    assert results[0]["metadata"]["document_name"] == "annual_report.pdf"
