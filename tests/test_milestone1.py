import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import io
from adaptive_rag.ingestion.pipeline import IngestionPipeline
from adaptive_rag.models.schema import ChunkType, ContentType

def create_advanced_synthetic_pdf() -> bytes:
    """Create a PDF with text and a structured table using ReportLab."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 750, "SECTION 1: FINANCIALS")
    c.drawString(100, 730, "The following table demonstrates revenue and profit across 2023-2025.")
    
    data = [['Year', 'Revenue', 'Profit'], ['2023', '$10B', '$2B'], ['2024', '$12B', '$3B'], ['2025', '$15B', '$4B']]
    t = Table(data)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    t.wrapOn(c, 400, 200)
    t.drawOn(c, 100, 600)
    c.showPage()
    c.save()
    return buf.getvalue()

def test_table_preservation_and_versioning():
    """Verify tables are extracted intact and versions are respected."""
    pipeline = IngestionPipeline()
    pdf_bytes = create_advanced_synthetic_pdf()
    
    # Test v1.0
    doc, parents, children = pipeline.ingest_pdf_bytes(pdf_bytes, "financials.pdf", version="1.0")
    
    assert doc.version == "1.0"
    
    table_chunks = [c for c in children if c.metadata.content_type == ContentType.TABLE]
    assert len(table_chunks) == 1
    
    table_chunk = table_chunks[0]
    assert table_chunk.structured_data is not None
    assert "Year" in table_chunk.structured_data.headers
    assert "2024" in table_chunk.structured_data.rows[1]
    assert "$12B" in table_chunk.structured_data.rows[1]
    
    # Test incremental version change cache miss
    doc_v2, _, _ = pipeline.ingest_pdf_bytes(pdf_bytes, "financials.pdf", version="2.0")
    assert doc_v2.document_id != doc.document_id
    assert "v2_0" in doc_v2.document_id
