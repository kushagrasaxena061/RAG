import pytest, tempfile, os, io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from adaptive_rag.ingestion.pipeline import IngestionPipeline

def create_advanced_synthetic_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 750, "SECTION 1: FINANCIALS")
    data = [["Year", "Revenue"], ["2024", "$12B"]]
    t = Table(data)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    t.wrapOn(c, 400, 200)
    t.drawOn(c, 100, 600)
    c.showPage()
    c.save()
    return buf.getvalue()

def test_table_preservation_and_versioning():
    pipeline = IngestionPipeline()
    pdf_bytes = create_advanced_synthetic_pdf()
    
    # FIX: Write to temp file to simulate the OOM-safe disk upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
        
    try:
        res = pipeline.ingest_pdf_path(tmp_path, "financials.pdf")
        assert res is not None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
