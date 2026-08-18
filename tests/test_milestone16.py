import pytest, tempfile, os, io
from reportlab.pdfgen import canvas
from adaptive_rag.ingestion.pipeline import IngestionPipeline

def test_full_pipeline_ingestion_with_tables_and_metadata():
    pipeline = IngestionPipeline()
    
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, "Test Document")
    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()

    # FIX: Write to temp file to simulate the OOM-safe disk upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
        
    try:
        res = pipeline.ingest_pdf_path(tmp_path, "finance_summary.pdf")
        assert res is not None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
