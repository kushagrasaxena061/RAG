import pytest
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from adaptive_rag.ingestion.pipeline import IngestionPipeline

def generate_multi_content_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(80, 750, "FINANCIAL AND ARCHITECTURAL SUMMARY")
    c.drawString(80, 730, "Quarterly report showing earnings, deductions, and network topology.")
    
    data = [
        ['Head', 'Amount', 'Type'],
        ['Basic Salary', '$8,000', 'Earning'],
        ['Provident Fund', '$800', 'Deduction'],
        ['Net Take Home', '$7,200', 'Total']
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)
    ]))
    t.wrapOn(c, 400, 200)
    t.drawOn(c, 80, 600)
    c.showPage()
    c.save()
    return buf.getvalue()

def test_full_pipeline_ingestion_with_tables_and_metadata():
    pipeline = IngestionPipeline()
    pdf_bytes = generate_multi_content_pdf()
    
    doc, parents, children, image_map = pipeline.ingest_pdf_bytes(pdf_bytes, "finance_summary.pdf")
    
    assert doc.total_pages == 1
    assert len(children) >= 1
    table_chunks = [c for c in children if c.metadata.content_type.value == "table"]
    assert len(table_chunks) >= 1
    assert "Basic Salary" in table_chunks[0].content
    assert "Deduction" in table_chunks[0].content
