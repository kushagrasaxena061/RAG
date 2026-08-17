import pytest
import time
from adaptive_rag.ingestion.diff_engine import IncrementalDiffEngine
from adaptive_rag.storage.cache import QueryResponseCache
from adaptive_rag.models.schema import Document, ParsedPage, Section
from adaptive_rag.context.token_budget import TokenBudgetManager
from adaptive_rag.config import IngestionConfig, ModelContextConfig

@pytest.fixture
def diff_setup():
    config = IngestionConfig()
    token_counter = TokenBudgetManager(ModelContextConfig())
    engine = IncrementalDiffEngine(config, token_counter)
    return engine, token_counter

def test_incremental_page_diffing(diff_setup):
    """Verify unchanged pages are skipped during re-indexing."""
    engine, token_counter = diff_setup

    p1 = ParsedPage(page_number=1, text="Page 1 static overview content.", tables=[], headings=["Section 1"], token_count=6)
    p2 = ParsedPage(page_number=2, text="Page 2 financial metrics Q1.", tables=[], headings=["Section 2"], token_count=6)
    
    sec1 = Section(section_id="s1", title="Section 1", level=1, page_start=1, page_end=1, content="Page 1 static overview content.", tables=[], token_count=6)
    sec2 = Section(section_id="s2", title="Section 2", level=1, page_start=2, page_end=2, content="Page 2 financial metrics Q1.", tables=[], token_count=6)

    doc_v1 = Document(
        document_id="doc_v1", document_name="report.pdf", version="1.0",
        document_hash="h1", document_type="pdf", total_pages=2, total_tokens=12,
        pages=[p1, p2], sections=[sec1, sec2]
    )

    parents_1, children_1, stats_1 = engine.diff_and_chunk(doc_v1)
    assert stats_1["modified_pages"] == 2
    assert stats_1["reused_pages"] == 0

    # Version 2: Page 1 remains identical, Page 2 is modified
    p2_modified = ParsedPage(page_number=2, text="Page 2 financial metrics updated Q2.", tables=[], headings=["Section 2"], token_count=7)
    sec2_modified = Section(section_id="s2_mod", title="Section 2", level=1, page_start=2, page_end=2, content="Page 2 financial metrics updated Q2.", tables=[], token_count=7)

    doc_v2 = Document(
        document_id="doc_v2", document_name="report.pdf", version="2.0",
        document_hash="h2", document_type="pdf", total_pages=2, total_tokens=13,
        pages=[p1, p2_modified], sections=[sec1, sec2_modified]
    )

    parents_2, children_2, stats_2 = engine.diff_and_chunk(doc_v2)
    assert stats_2["reused_pages"] == 1
    assert stats_2["modified_pages"] == 1

def test_query_cache_invalidation():
    """Verify cache hits and document hash state invalidation."""
    cache = QueryResponseCache(ttl_seconds=3600)
    query = "What is the revenue?"
    doc_hash_v1 = "hash_v1_abc"
    doc_hash_v2 = "hash_v2_xyz"

    response_payload = {"answer": "Revenue was $10M."}

    cache.set(query, doc_hash_v1, response_payload)
    
    # Exact hit with matching doc state
    assert cache.get(query, doc_hash_v1) == response_payload

    # Cache miss when document hash changes
    assert cache.get(query, doc_hash_v2) is None
